from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from wav2aug.cpu._noise_addition import _sample_noise_like, add_noise


def load_audio_tc(src: str, *, sample_rate: int | None = None):
    """Load audio using torchcodec. Returns (waveform[C,T], sample_rate)."""
    from torchcodec.decoders import AudioDecoder
    dec = AudioDecoder(src, sample_rate=sample_rate)
    samp = dec.get_all_samples()
    wav = samp.data.contiguous().to(torch.float32)
    return wav, int(samp.sample_rate)


def test_empty_waveform_noop():
    """Empty waveform should return unchanged."""
    x = torch.empty(0)
    out = add_noise(x, sample_rate=16000)
    assert out.numel() == 0
    assert out.data_ptr() == x.data_ptr()


def test_inplace_modification():
    """Function should modify tensor in place."""
    x = torch.randn(1000)
    x_ptr = x.data_ptr()
    x_orig = x.clone()
    out = add_noise(x, sample_rate=16000)
    
    assert out.data_ptr() == x_ptr
    assert not torch.allclose(out, x_orig, rtol=1e-6)


def test_shape_preservation():
    """Output shape should match input shape."""
    x_mono = torch.randn(500)
    out_mono = add_noise(x_mono.clone(), sample_rate=16000)
    assert out_mono.shape == (500,)
    
    x_stereo = torch.randn(2, 500)
    out_stereo = add_noise(x_stereo.clone(), sample_rate=16000)
    assert out_stereo.shape == (2, 500)


def test_snr_affects_noise_level(monkeypatch):
    """Higher SNR should result in lower noise, lower SNR in higher noise."""
    x1 = torch.ones(1000) * 0.5
    x2 = x1.clone()
    
    def mock_sample_noise_like(waveform, sr, noise_dir):
        return torch.randn_like(waveform) * 0.1
    
    monkeypatch.setattr("wav2aug.cpu._noise_addition._sample_noise_like", mock_sample_noise_like)
    
    snr_calls = [20.0, 5.0]
    call_count = 0
    def mock_rand(*args, **kwargs):
        nonlocal call_count
        result = torch.tensor(0.0) if call_count == 0 else torch.tensor(1.0)
        call_count += 1
        return result
    monkeypatch.setattr(torch, "rand", mock_rand)
    
    out1 = add_noise(x1, sample_rate=16000, snr_low=20.0, snr_high=20.0)
    out2 = add_noise(x2, sample_rate=16000, snr_low=5.0, snr_high=5.0)
    
    energy1 = (out1 - torch.ones(1000) * 0.5).pow(2).mean()
    energy2 = (out2 - torch.ones(1000) * 0.5).pow(2).mean()
    
    assert energy2 > energy1


def test_noise_dir_fallback_to_random():
    """When noise_dir is None or empty, should fall back to random noise."""
    x = torch.randn(100)
    
    out1 = add_noise(x.clone(), sample_rate=16000, noise_dir=None, download=False)
    assert out1.shape == x.shape
    
    assert not torch.equal(out1, x)


@pytest.mark.skipif(not Path("tests/test.mp3").exists(), reason="test.mp3 not found")  
def test_real_audio_noise_addition():
    """Test noise addition on real MP3 audio."""
    wav, sr = load_audio_tc("tests/test.mp3")
    wav_orig = wav.clone()
    
    out = add_noise(wav, sample_rate=sr, snr_low=10.0, snr_high=15.0, download=False)
    
    assert out is wav
    assert out.shape == wav_orig.shape
    assert not torch.allclose(out, wav_orig, rtol=1e-3)
    
    orig_rms = wav_orig.pow(2).mean().sqrt()
    new_rms = out.pow(2).mean().sqrt()
    assert new_rms < orig_rms * 2


def test_multichannel_consistent_noise():
    """For multichannel input, same noise pattern should be applied across channels when noise is mono."""
    x = torch.randn(2, 200)
    
    with patch('wav2aug.cpu._noise_addition._sample_noise_like') as mock_noise:
        mono_noise = torch.randn(1, 200) * 0.1
        mock_noise.return_value = mono_noise.repeat(2, 1)
        
        out = add_noise(x.clone(), sample_rate=16000, download=False)
        
        assert not torch.equal(out[0], x[0])
        assert not torch.equal(out[1], x[1])


def test_cpu_only_assertion():
    """Function should reject GPU tensors."""
    if torch.cuda.is_available():
        x = torch.randn(10).cuda()
        with pytest.raises(AssertionError):
            add_noise(x, sample_rate=16000)


def test_wrong_ndim_assertion():
    """Function should reject tensors with wrong number of dimensions."""
    x = torch.randn(2, 3, 4)  # 3D tensor
    with pytest.raises(AssertionError):
        add_noise(x, sample_rate=16000)


def test_sample_noise_like_no_noise_dir():
    """_sample_noise_like should return random noise when noise_dir is None."""
    x = torch.randn(2, 100)
    noise = _sample_noise_like(x, sr=16000, noise_dir=None)
    
    assert noise.shape == x.shape
    assert noise.dtype == x.dtype
    assert torch.abs(noise.mean()) < 0.2
    assert 0.5 < noise.std() < 2.0


def test_sample_noise_like_empty_noise_dir(tmp_path):
    """_sample_noise_like should return random noise when noise_dir is empty."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    x = torch.randn(1, 100)
    noise = _sample_noise_like(x, sr=16000, noise_dir=str(empty_dir))
    
    assert noise.shape == x.shape
    assert noise.dtype == x.dtype


def test_snr_boundary_values():
    """Test edge cases for SNR values."""
    x = torch.ones(100) * 0.5
    
    out_high = add_noise(x.clone(), sample_rate=16000, snr_low=50.0, snr_high=50.0, download=False)
    energy_high = (out_high - x).pow(2).mean()
    
    out_low = add_noise(x.clone(), sample_rate=16000, snr_low=-10.0, snr_high=-10.0, download=False)
    energy_low = (out_low - x).pow(2).mean()
    
    assert energy_low > energy_high * 10


@patch('wav2aug.cpu._noise_addition._list_audio_files')
@patch('torchcodec.decoders.AudioDecoder')
def test_sample_noise_like_with_audio_files(mock_decoder, mock_list_files):
    """Test _sample_noise_like when audio files are available."""
    # Mock file list
    mock_list_files.return_value = ['/fake/noise1.wav', '/fake/noise2.wav']
    
    # Mock AudioDecoder
    mock_samples = MagicMock()
    mock_samples.data = torch.randn(1, 500).contiguous()
    mock_decoder_instance = MagicMock()
    mock_decoder_instance.get_all_samples.return_value = mock_samples
    mock_decoder.return_value = mock_decoder_instance
    
    x = torch.randn(2, 200)
    noise = _sample_noise_like(x, sr=16000, noise_dir='/fake/noise_dir')
    
    assert noise.shape == x.shape
    assert noise.dtype == x.dtype
    mock_list_files.assert_called_once_with('/fake/noise_dir')
    mock_decoder.assert_called_once()