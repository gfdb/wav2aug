import os
import pathlib

import pytest
import torch

# adjust import to your module path
import wav2aug.cpu._speed_perturbation as mod

try:
    from torchcodec.decoders import AudioDecoder
    _HAS_TORCHCODEC = True
except Exception:
    _HAS_TORCHCODEC = False


def load_audio_tc(src: str, *, sample_rate: int | None = None):
    """Load audio using torchcodec. Returns (waveform[C,T], sample_rate)."""
    from torchcodec.decoders import AudioDecoder
    dec = AudioDecoder(src, sample_rate=sample_rate)
    samp = dec.get_all_samples()
    wav = samp.data.contiguous().to(torch.float32)
    return wav, int(samp.sample_rate)


def _stub_randint_return(idx: int):
    """Stub torch.randint to return a fixed index once per call (scalar draws only)."""
    def _stub(low, high=None, size=(), **kwargs):
        assert size == () or size is None
        return torch.tensor(int(idx), dtype=torch.long)
    return _stub


@pytest.mark.skipif(not _HAS_TORCHCODEC, reason="torchcodec not available")
def test_file_exists():
    """Sanity: test asset must exist next to repo root or set AUDIO_FILE env."""
    p = pathlib.Path(os.environ.get("AUDIO_FILE", "tests/test.mp3"))
    if not p.exists():
        pytest.skip("tests/test.mp3 not found; set AUDIO_FILE env to an existing file")
    assert p.exists()


@pytest.mark.skipif(not _HAS_TORCHCODEC, reason="torchcodec not available")
def test_speed_up_shortens_real_audio(monkeypatch):
    """Given test.mp3, speed=1.1 → fewer samples, same channels."""
    p = pathlib.Path(os.environ.get("AUDIO_FILE", "tests/test.mp3"))
    if not p.exists():
        pytest.skip("tests/test.mp3 not found; set AUDIO_FILE env to an existing file")

    wav, sr = load_audio_tc(str(p))
    C, T = wav.shape
    assert T > 1

    monkeypatch.setattr(mod.torch, "randint", _stub_randint_return(2))
    out = mod.speed_perturb(wav)

    expected_T = max(1, int(round(T / 1.1)))
    assert out.shape == (C, expected_T)


@pytest.mark.skipif(not _HAS_TORCHCODEC, reason="torchcodec not available")
def test_slow_down_lengthens_real_audio(monkeypatch):
    """Given test.mp3, speed=0.9 → more samples."""
    p = pathlib.Path(os.environ.get("AUDIO_FILE", "tests/test.mp3"))
    if not p.exists():
        pytest.skip("tests/test.mp3 not found; set AUDIO_FILE env to an existing file")

    wav, sr = load_audio_tc(str(p))
    C, T = wav.shape
    assert T > 1

    monkeypatch.setattr(mod.torch, "randint", _stub_randint_return(0))
    out = mod.speed_perturb(wav)

    expected_T = max(1, int(round(T / 0.9)))
    assert out.shape == (C, expected_T)


@pytest.mark.skipif(not _HAS_TORCHCODEC, reason="torchcodec not available")
def test_identity_noop_pointer_and_shape(monkeypatch):
    """speed=1.0 returns the same tensor object and sample count."""
    p = pathlib.Path(os.environ.get("AUDIO_FILE", "tests/test.mp3"))
    if not p.exists():
        pytest.skip("tests/test.mp3 not found; set AUDIO_FILE env to an existing file")

    wav, sr = load_audio_tc(str(p))
    ptr = wav.data_ptr()

    monkeypatch.setattr(mod.torch, "randint", _stub_randint_return(1))
    out = mod.speed_perturb(wav)

    assert out.data_ptr() == ptr
    assert out.shape == wav.shape


@pytest.mark.skipif(not _HAS_TORCHCODEC, reason="torchcodec not available")
def test_supports_mono_vector_input(monkeypatch):
    """Selecting a single channel [T] is supported and resampled."""
    p = pathlib.Path(os.environ.get("AUDIO_FILE", "tests/test.mp3"))
    if not p.exists():
        pytest.skip("tests/test.mp3 not found; set AUDIO_FILE env to an existing file")

    wav, sr = load_audio_tc(str(p))
    x = wav[0].contiguous()
    T = x.numel()
    assert T > 1

    monkeypatch.setattr(mod.torch, "randint", _stub_randint_return(2))
    y = mod.speed_perturb(x)
    expected_T = max(1, int(round(T / 1.1)))
    assert y.ndim == 1 and y.numel() == expected_T