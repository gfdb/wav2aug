from pathlib import Path

import pytest
import torch

from wav2aug.cpu import add_babble_noise


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
    batch_sum = torch.randn(100)  # batch_sum can be any size
    out = add_babble_noise(x, batch_sum)
    assert out.numel() == 0
    assert out.data_ptr() == x.data_ptr()


def test_inplace_modification():
    """Function should modify waveform tensor in place."""
    x = torch.randn(500)
    batch_sum = torch.randn(500)
    x_ptr = x.data_ptr()
    x_orig = x.clone()
    
    out = add_babble_noise(x, batch_sum)
    
    assert out.data_ptr() == x_ptr
    assert not torch.allclose(out, x_orig, rtol=1e-6)


def test_shape_preservation():
    """Output shape should match input waveform shape."""
    x_mono = torch.randn(300)
    batch_mono = torch.randn(300)
    out_mono = add_babble_noise(x_mono.clone(), batch_mono)
    assert out_mono.shape == (300,)
    
    x_stereo = torch.randn(2, 300)
    batch_stereo = torch.randn(2, 300)
    out_stereo = add_babble_noise(x_stereo.clone(), batch_stereo)
    assert out_stereo.shape == (2, 300)


def test_batch_sum_length_adjustment():
    """batch_sum should be padded or cropped to match waveform length."""
    x = torch.randn(100)
    
    batch_long = torch.randn(200)
    out_long = add_babble_noise(x.clone(), batch_long)
    assert out_long.shape == (100,)
    
    batch_short = torch.randn(50)
    out_short = add_babble_noise(x.clone(), batch_short)
    assert out_short.shape == (100,)


def test_batch_sum_channel_adjustment():
    """batch_sum channels should be adjusted to match waveform channels."""
    x = torch.randn(2, 100)
    
    batch_mono = torch.randn(1, 100)
    out = add_babble_noise(x.clone(), batch_mono)
    assert out.shape == (2, 100)
    
    batch_quad = torch.randn(4, 100)  
    out2 = add_babble_noise(x.clone(), batch_quad)
    assert out2.shape == (2, 100)


def test_snr_affects_babble_level(monkeypatch):
    """Higher SNR should result in lower babble noise level."""
    x1 = torch.ones(1000) * 0.5
    x2 = x1.clone()
    batch_sum = torch.randn(1000) * 0.2
    
    snr_calls = [20.0, 5.0]
    call_count = 0
    def mock_rand(*args, **kwargs):
        nonlocal call_count
        result = torch.tensor(0.0) if call_count == 0 else torch.tensor(1.0)
        call_count += 1
        return result
    monkeypatch.setattr(torch, "rand", mock_rand)
    
    out1 = add_babble_noise(x1, batch_sum.clone(), snr_low=20.0, snr_high=20.0)
    out2 = add_babble_noise(x2, batch_sum.clone(), snr_low=5.0, snr_high=5.0)
    
    babble_energy1 = (out1 - torch.ones(1000) * 0.5).pow(2).mean()
    babble_energy2 = (out2 - torch.ones(1000) * 0.5).pow(2).mean()
    
    assert babble_energy2 > babble_energy1


def test_different_from_original():
    """Adding babble noise should change the waveform."""
    x = torch.randn(200)
    batch_sum = torch.randn(200) * 0.5
    x_orig = x.clone()
    
    out = add_babble_noise(x, batch_sum)
    
    assert not torch.allclose(out, x_orig, rtol=1e-6)


def test_uses_shared_snr_mixing_function(monkeypatch):
    """Should use the shared apply_snr_and_mix function from noise_addition."""
    from wav2aug.cpu._noise_addition import apply_snr_and_mix

    x = torch.randn(100)
    batch_sum = torch.randn(100)

    call_count = 0
    original_func = apply_snr_and_mix

    def mock_apply_snr_and_mix(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_func(*args, **kwargs)

    monkeypatch.setattr("wav2aug.cpu._babble_noise_addition.apply_snr_and_mix", mock_apply_snr_and_mix)

    add_babble_noise(x, batch_sum)

    assert call_count == 1
@pytest.mark.skipif(not Path("tests/test.mp3").exists(), reason="test.mp3 not found")
def test_real_audio_babble_addition():
    """Test babble noise addition on real MP3 audio."""
    wav, sr = load_audio_tc("tests/test.mp3")
    wav_orig = wav.clone()
    
    batch_sum = torch.randn_like(wav) * 0.3
    
    out = add_babble_noise(wav, batch_sum, snr_low=10.0, snr_high=15.0)
    
    assert out is wav
    assert out.shape == wav_orig.shape
    assert not torch.allclose(out, wav_orig, rtol=1e-3)
    
    # Should not blow up the magnitude
    orig_rms = wav_orig.pow(2).mean().sqrt()
    new_rms = out.pow(2).mean().sqrt()
    assert new_rms < orig_rms * 3


def test_cpu_only_assertion():
    """Function should reject GPU tensors."""
    if torch.cuda.is_available():
        x = torch.randn(10).cuda()
        batch_sum = torch.randn(10).cuda()
        with pytest.raises(AssertionError, match="Input waveform must be on CPU"):
            add_babble_noise(x, batch_sum)


def test_batch_sum_cpu_assertion():
    """Function should reject GPU batch_sum.""" 
    if torch.cuda.is_available():
        x = torch.randn(10)
        batch_sum = torch.randn(10).cuda()
        with pytest.raises(AssertionError, match="Batch sum must be on CPU"):
            add_babble_noise(x, batch_sum)


def test_wrong_waveform_ndim_assertion():
    """Function should reject waveforms with wrong number of dimensions."""
    x = torch.randn(2, 3, 4)  # 3D tensor
    batch_sum = torch.randn(2, 3, 4)
    with pytest.raises(AssertionError, match="Expected waveform shape"):
        add_babble_noise(x, batch_sum)


def test_wrong_batch_sum_ndim_assertion():
    """Function should reject batch_sum with wrong number of dimensions."""
    x = torch.randn(100)
    batch_sum = torch.randn(2, 3, 4)
    with pytest.raises(AssertionError, match="Expected babble_waveform shape"):
        add_babble_noise(x, batch_sum)


def test_snr_range_boundary_values():
    """Test edge cases for SNR values."""
    x = torch.ones(100) * 0.5
    batch_sum = torch.randn(100) * 0.1
    
    out_high = add_babble_noise(x.clone(), batch_sum.clone(), snr_low=50.0, snr_high=50.0)
    energy_high = (out_high - x).pow(2).mean()
    
    out_low = add_babble_noise(x.clone(), batch_sum.clone(), snr_low=-10.0, snr_high=-10.0)
    energy_low = (out_low - x).pow(2).mean()
    
    assert energy_low > energy_high * 5


def test_default_snr_range():
    """Test that default SNR range (0-20 dB) works correctly."""
    x = torch.randn(200)
    batch_sum = torch.randn(200) * 0.2
    
    out = add_babble_noise(x.clone(), batch_sum)
    assert out.shape == x.shape
    assert not torch.equal(out, x)


def test_mono_to_stereo_conversion():
    """Test converting mono batch_sum to stereo to match waveform."""
    x_stereo = torch.randn(2, 150)
    batch_mono = torch.randn(1, 150)
    
    out = add_babble_noise(x_stereo.clone(), batch_mono)
    
    assert out.shape == (2, 150)
    assert not torch.equal(out[0], x_stereo[0])
    assert not torch.equal(out[1], x_stereo[1])