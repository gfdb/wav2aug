from pathlib import Path

import pytest
import torch

from wav2aug.cpu import rand_amp_clip


def load_audio_tc(src: str, *, sample_rate: int | None = None):
    """Load audio using torchcodec. Returns (waveform[C,T], sample_rate)."""
    from torchcodec.decoders import AudioDecoder
    dec = AudioDecoder(src, sample_rate=sample_rate)
    samp = dec.get_all_samples()
    wav = samp.data.contiguous().to(torch.float32)
    return wav, int(samp.sample_rate)


def test_identity_when_clip_is_one(monkeypatch):
    """When clip value is 1.0, waveform should remain unchanged."""
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(1.0))
    
    x = torch.tensor([0.5, -0.8, 0.3, -0.1])
    x_orig = x.clone()
    
    out = rand_amp_clip(x, clip_low=0, clip_high=1.0)
    
    assert out is x
    assert torch.allclose(out, x_orig, rtol=1e-6, atol=1e-6)


def test_clipping_reduces_peaks(monkeypatch):
    """Clipping should reduce peak values while preserving relative scaling."""
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    
    x = torch.tensor([1.0, -2.0, 0.5, -0.25])
    
    out = rand_amp_clip(x, clip_low=0.5, clip_high=0.5)
    
    assert out.abs().max().item() == pytest.approx(2.0, abs=1e-5)
    assert out[0].item() == pytest.approx(2.0, abs=1e-5)
    assert out[1].item() == pytest.approx(-2.0, abs=1e-5)
    assert out[2].item() == pytest.approx(1.0, abs=1e-5)
    assert out[3].item() == pytest.approx(-0.5, abs=1e-5)


def test_multichannel_preserves_shape_and_balance(monkeypatch):
    """Multichannel audio should preserve channel balance."""
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    
    x = torch.tensor([
        [1.0, -0.8, 0.4],
        [2.0, -1.6, 0.8],
    ])
    
    out = rand_amp_clip(x, clip_low=0.6, clip_high=0.6)
    
    assert out.shape == (2, 3)
    assert out[0].abs().max().item() == pytest.approx(1.0, abs=1e-5)
    assert out[1].abs().max().item() == pytest.approx(2.0, abs=1e-5)


def test_clip_range_affects_output(monkeypatch):
    """Different clip values should produce different outputs."""
    x1 = torch.tensor([2.0, -1.0, 0.5])
    x2 = x1.clone()
    
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    out1 = rand_amp_clip(x1, clip_low=0.3, clip_high=0.3)
    
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    out2 = rand_amp_clip(x2, clip_low=0.8, clip_high=0.8)
    
    assert not torch.allclose(out1, out2, rtol=1e-3)


def test_empty_waveform_noop():
    """Empty waveform should return unchanged."""
    x = torch.empty(0)
    out = rand_amp_clip(x)
    assert out.numel() == 0
    assert out.data_ptr() == x.data_ptr()


def test_mono_and_multichannel_shapes():
    """Function should handle both mono [T] and multichannel [C,T] shapes."""
    x_mono = torch.randn(100)
    out_mono = rand_amp_clip(x_mono.clone())
    assert out_mono.shape == (100,)
    
    x_multi = torch.randn(2, 100)
    out_multi = rand_amp_clip(x_multi.clone())
    assert out_multi.shape == (2, 100)
    x_multi = torch.randn(2, 100)
    out_multi = rand_amp_clip(x_multi.clone())
    assert out_multi.shape == (2, 100)


@pytest.mark.skipif(not Path("tests/test.mp3").exists(), reason="test.mp3 not found")
def test_real_audio_clipping():
    """Test amplitude clipping on real MP3 audio."""
    wav, _ = load_audio_tc("tests/test.mp3")
    wav_orig = wav.clone()
    
    out = rand_amp_clip(wav, clip_low=0.3, clip_high=0.7)

    assert out is wav
    assert out.shape == wav_orig.shape

    new_peak = out.abs().max()
    assert new_peak > 0.01
    assert new_peak < 10.0

def test_inplace_modification():
    """Function should modify tensor in place."""
    x = torch.randn(50)
    x_ptr = x.data_ptr()
    out = rand_amp_clip(x)
    
    assert out.data_ptr() == x_ptr
    assert out is x


def test_cpu_only_assertion():
    """Function should reject GPU tensors."""
    if torch.cuda.is_available():
        x = torch.randn(10).cuda()
        with pytest.raises(AssertionError, match="CPU only"):
            rand_amp_clip(x)


def test_wrong_ndim_assertion():
    """Function should reject tensors with wrong number of dimensions."""
    x = torch.randn(2, 3, 4)
    with pytest.raises(AssertionError, match="expected \\[T\\] or \\[C, T\\]"):
        rand_amp_clip(x)