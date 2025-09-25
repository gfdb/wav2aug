from pathlib import Path

import pytest
import torch

from wav2aug.cpu import rand_amp_scale


def load_audio_tc(src: str, *, sample_rate: int | None = None):
    """Load audio using torchcodec. Returns (waveform[C,T], sample_rate)."""
    from torchcodec.decoders import AudioDecoder
    dec = AudioDecoder(src, sample_rate=sample_rate)
    samp = dec.get_all_samples()
    wav = samp.data.contiguous().to(torch.float32)
    return wav, int(samp.sample_rate)


def test_peak_normalization_then_scaling(monkeypatch):
    """Function should normalize to peak=1, then scale by amp factor."""
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.5))
    
    x = torch.tensor([2.0, -4.0, 1.0, -0.5])
    
    out = rand_amp_scale(x, amp_low=0.05, amp_high=0.5)
    
    expected_amp = 0.5 * (0.5 - 0.05) + 0.05
    expected_peak = expected_amp
    
    assert out.abs().max().item() == pytest.approx(expected_peak, abs=1e-5)


def test_multichannel_per_channel_normalization(monkeypatch):
    """Each channel should be normalized by its own peak."""
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    
    x = torch.tensor([
        [1.0, -2.0, 0.5],
        [4.0, -1.0, 2.0],
    ])
    
    out = rand_amp_scale(x, amp_low=0.05, amp_high=0.05)
    
    assert out[0].abs().max().item() == pytest.approx(0.05, abs=1e-6)
    assert out[1].abs().max().item() == pytest.approx(0.05, abs=1e-6)
    assert out[0, 0].item() == pytest.approx(0.025, abs=1e-6)
    assert out[0, 1].item() == pytest.approx(-0.05, abs=1e-6)
    assert out[0, 2].item() == pytest.approx(0.0125, abs=1e-6)


def test_mono_vector_handling(monkeypatch):
    """Mono [T] input should work correctly."""
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.111111))
    
    x = torch.tensor([3.0, -6.0, 1.5])
    
    out = rand_amp_scale(x, amp_low=0.05, amp_high=0.5)
    
    expected_amp = (0.5 - 0.05) * 0.111111 + 0.05
    assert out.abs().max().item() == pytest.approx(expected_amp, abs=1e-5)
    assert out.shape == (3,)


def test_different_amp_ranges_produce_different_outputs(monkeypatch):
    """Different amp ranges should result in different peak amplitudes."""
    x1 = torch.tensor([1.0, -2.0, 0.5])
    x2 = x1.clone()
    
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.5))
    
    out1 = rand_amp_scale(x1, amp_low=0.1, amp_high=0.3)
    out2 = rand_amp_scale(x2, amp_low=0.6, amp_high=0.8)
    
    peak1 = out1.abs().max().item()
    peak2 = out2.abs().max().item()
    
    assert peak1 == pytest.approx(0.2, abs=1e-5)
    assert peak2 == pytest.approx(0.7, abs=1e-5)
    assert abs(peak1 - peak2) > 0.4


def test_empty_waveform_noop():
    """Empty waveform should return unchanged."""
    x = torch.empty(0)
    out = rand_amp_scale(x)
    assert out.numel() == 0
    assert out.data_ptr() == x.data_ptr()


def test_preserves_zero_values():
    """Zero values should remain zero after scaling."""
    x = torch.tensor([1.0, 0.0, -2.0, 0.0])
    out = rand_amp_scale(x)
    
    assert out[1].item() == 0.0
    assert out[3].item() == 0.0


def test_inplace_modification():
    """Function should modify tensor in place."""
    x = torch.randn(50) * 5
    x_ptr = x.data_ptr()
    out = rand_amp_scale(x)
    
    assert out.data_ptr() == x_ptr
    assert out is x


@pytest.mark.skipif(not Path("tests/test.mp3").exists(), reason="test.mp3 not found")
def test_real_audio_scaling():
    """Test amplitude scaling on real MP3 audio."""
    wav, sr = load_audio_tc("tests/test.mp3")
    wav_orig = wav.clone()
    orig_shape = wav.shape
    
    out = rand_amp_scale(wav, amp_low=0.1, amp_high=0.4)

    assert out is wav
    assert out.shape == orig_shape

    peak = out.abs().max().item()

    assert peak >= 0.04
    assert peak <= 0.5
    orig_peak = wav_orig.abs().max().item()
    if orig_peak > 1.1:
        assert not torch.allclose(out, wav_orig, rtol=1e-3)


def test_minimum_peak_clamping():
    """Very small peaks should be clamped to 1.0 to avoid division issues."""
    x = torch.tensor([1e-8, -5e-9, 2e-8])
    
    out = rand_amp_scale(x)
    
    assert out.abs().max().item() < 1.0
    assert torch.isfinite(out).all()


def test_cpu_only_assertion():
    """Function should reject GPU tensors."""
    if torch.cuda.is_available():
        x = torch.randn(10).cuda()
        with pytest.raises(AssertionError):
            rand_amp_scale(x)


def test_wrong_ndim_assertion():
    """Function should reject tensors with wrong number of dimensions."""
    x = torch.randn(2, 3, 4)
    with pytest.raises(AssertionError):
        rand_amp_scale(x)