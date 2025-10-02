from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

_FILTER_LEN: Final[int] = 101
_PAD: Final[int] = _FILTER_LEN // 2
_T_IDX = torch.arange(_FILTER_LEN, dtype=torch.float32) - ((_FILTER_LEN - 1) / 2.0)
_BLACKMAN = torch.blackman_window(_FILTER_LEN, periodic=True, dtype=torch.float32)


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(x) / x)


@torch.no_grad()
def freq_drop(
    waveforms: torch.Tensor,
    *,
    bound_low: float = 1e-12,
    bound_high: float = 1.0,
    band_count_low: int = 1,
    band_count_high: int = 8,
    band_width: float = 0.10,
    clamp_abs: float = 8.0,
) -> torch.Tensor:
    """Frequency dropout with per-sample independent notch filter stacks.

    Each waveform builds its own randomized multi-notch kernel instead of
    reusing a single shared filter for the batch.

    Stabilization additions retained:
    - Normalize each intermediate kernel
    - Final kernel L1 normalization + nan_to_num
    - Clamp output amplitude to +/- clamp_abs
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("freq_drop expects CUDA tensors")

    batch, total_time = waveforms.shape
    if batch == 0 or total_time == 0:
        return waveforms

    if band_count_high <= 0:
        return waveforms

    bound_low = max(0.0, min(1.0, float(bound_low)))
    bound_high = max(bound_low, min(1.0, float(bound_high)))
    width = max(0.0, min(1.0, float(band_width)))
    rng = bound_high - bound_low
    if rng <= 0.0 or width <= 0.0:
        return waveforms

    device = waveforms.device
    dtype = waveforms.dtype

    t = _T_IDX.to(device=device, dtype=dtype)
    window = _BLACKMAN.to(device=device, dtype=dtype)

    for b in range(batch):
        band_count = int(
            torch.randint(
                band_count_low,
                band_count_high + 1,
                (),
                device=device,
            ).item()
        )
        if band_count <= 0:
            continue
        drop = torch.zeros(_FILTER_LEN, device=device, dtype=dtype)
        drop[_PAD] = 1.0
        for _ in range(band_count):
            freq = torch.rand((), device=device, dtype=dtype)
            freq = (freq * rng + bound_low).clamp(1e-12, 1.0 - 1e-8)
            minus = (freq - width).clamp(1e-12, 1.0)
            plus = (freq + width).clamp(1e-12, 1.0)
            hlpf = _sinc(3.0 * minus * t) * window
            hlpf = hlpf / hlpf.sum().abs().clamp_min(1e-8)
            hhpf = _sinc(3.0 * plus * t) * window
            hhpf = hhpf / -hhpf.sum().abs().clamp_min(1e-8)
            hhpf[_PAD] += 1.0
            kernel = hlpf + hhpf
            kernel = kernel / kernel.abs().sum().clamp_min(1e-8)
            drop = F.conv1d(
                drop.view(1, 1, _FILTER_LEN),
                kernel.view(1, 1, _FILTER_LEN),
                padding=_PAD,
            ).view(_FILTER_LEN)
        if drop.abs().sum() > 0:
            drop = drop / drop.abs().sum().clamp_min(1e-8)
        drop = torch.nan_to_num(drop, nan=0.0, posinf=0.0, neginf=0.0)
        y = F.conv1d(waveforms[b:b+1].unsqueeze(1), drop.view(1,1,_FILTER_LEN), padding=_PAD).squeeze(1)
        if clamp_abs is not None and clamp_abs > 0:
            y = y.clamp_(-clamp_abs, clamp_abs)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        waveforms[b].copy_(y[0])
    return waveforms


__all__ = ["freq_drop"]
