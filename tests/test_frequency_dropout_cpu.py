import os

import pytest
import torch

# Import the SB-aligned implementation
import wav2aug.cpu._frequency_dropout as mod


def _stub_rand_and_randint(center_u01: float, band_count: int):
    """Stubs for torch.rand (scalar U[0,1]) and torch.randint (scalar band_count)."""
    def rand_stub(*args, **kwargs):
        return torch.tensor(center_u01, dtype=torch.float32)

    scalars = [band_count]
    def randint_stub(low, high=None, size=(), **kwargs):
        assert size == () or size is None
        assert scalars, "unexpected extra randint call"
        return torch.tensor(int(scalars.pop(0)), dtype=torch.long)

    return rand_stub, randint_stub


def _rfft_mag(x: torch.Tensor, sr: int):
    X = torch.fft.rfft(x)
    mag = X.abs()
    f = torch.fft.rfftfreq(x.numel(), d=1.0 / sr)
    return f, mag


def test_notch_removes_3khz_preserves_1khz_at_16k(monkeypatch):
    """Center ~0.375 → ~3 kHz at 16 kHz. One notch. 3 kHz drops, 1 kHz mostly keeps."""
    sr = 16_000
    T = sr
    t = torch.arange(T, dtype=torch.float32) / sr
    x = torch.sin(2 * torch.pi * 1000 * t) + 0.7 * torch.sin(2 * torch.pi * 3000 * t)

    rand_stub, randint_stub = _stub_rand_and_randint(center_u01=0.375, band_count=1)
    monkeypatch.setattr(mod.torch, "rand", rand_stub)
    monkeypatch.setattr(mod.torch, "randint", randint_stub)

    y = mod.freq_drop(x.clone())

    f, Xmag = _rfft_mag(x, sr)
    _, Ymag = _rfft_mag(y, sr)
    k1 = (f > 950) & (f < 1050)
    k3 = (f > 2900) & (f < 3100)

    drop_ratio = (Ymag[k3].mean() / Xmag[k3].mean()).item()
    keep_ratio = (Ymag[k1].mean() / Xmag[k1].mean()).item()
    assert drop_ratio < 0.2
    assert keep_ratio > 0.8


def test_same_mask_applied_to_all_channels(monkeypatch):
    """Grouped conv with one shared kernel per channel, identical outputs if inputs equal."""
    sr = 16_000
    T = sr // 2
    t = torch.arange(T, dtype=torch.float32) / sr
    mono = torch.sin(2 * torch.pi * 1000 * t) + torch.sin(2 * torch.pi * 3000 * t)
    x = torch.stack([mono, mono], dim=0)

    rand_stub, randint_stub = _stub_rand_and_randint(center_u01=0.375, band_count=1)
    monkeypatch.setattr(mod.torch, "rand", rand_stub)
    monkeypatch.setattr(mod.torch, "randint", randint_stub)

    y = mod.freq_drop(x.clone())
    assert torch.allclose(y[0], y[1], atol=0, rtol=0)


def test_accepts_T_and_CT_shapes(monkeypatch):
    """Function must accept [T] and [C, T] and preserve shape."""
    sr = 16_000
    T = sr // 4
    t = torch.arange(T, dtype=torch.float32) / sr
    sig = torch.sin(2 * torch.pi * 2000 * t)

    rand_stub, randint_stub = _stub_rand_and_randint(center_u01=0.3, band_count=1)
    monkeypatch.setattr(mod.torch, "rand", rand_stub)
    monkeypatch.setattr(mod.torch, "randint", randint_stub)

    y1 = mod.freq_drop(sig.clone())
    
    rand_stub2, randint_stub2 = _stub_rand_and_randint(center_u01=0.3, band_count=1)
    monkeypatch.setattr(mod.torch, "rand", rand_stub2)
    monkeypatch.setattr(mod.torch, "randint", randint_stub2)
    y2 = mod.freq_drop(torch.stack([sig, sig]))
    
    assert y1.shape == sig.shape
    assert y2.shape == (2, T)


def test_inplace_modification(monkeypatch):
    """Function modifies the passed tensor in place."""
    sr = 16_000
    T = 8000
    t = torch.arange(T, dtype=torch.float32) / sr
    x = torch.sin(2 * torch.pi * 3000 * t)

    rand_stub, randint_stub = _stub_rand_and_randint(center_u01=0.375, band_count=1)
    monkeypatch.setattr(mod.torch, "rand", rand_stub)
    monkeypatch.setattr(mod.torch, "randint", randint_stub)

    x_ptr = x.data_ptr()
    y = mod.freq_drop(x)
    assert y.data_ptr() == x_ptr
    assert not torch.allclose(y, torch.sin(2 * torch.pi * 3000 * t))


def test_zero_length_noop():
    """Zero-length input returned unchanged."""
    x = torch.empty(0)
    y = mod.freq_drop(x)
    assert y.numel() == 0
    assert y.data_ptr() == x.data_ptr()


