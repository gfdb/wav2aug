from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

# FIR filter parameters, pre computed at module load for performance
_FILTER_LEN: Final[int] = 101
_PAD: Final[int] = _FILTER_LEN // 2
# cache time indices and Blackman window
_T_IDX_F32 = torch.arange(_FILTER_LEN, dtype=torch.float32) - ((_FILTER_LEN - 1) / 2.0)
_BLACKMAN_F32 = torch.blackman_window(_FILTER_LEN, periodic=True, dtype=torch.float32)

@torch.no_grad()
def _sinc(x: torch.Tensor) -> torch.Tensor:
    """Compute sinc function with safe division by zero handling.

    Implements sinc(x) = sin(x)/x with special case handling for x=0.
    
    Args:
        x: Input tensor with arbitrary shape.
        
    Returns:
        Sinc values with same shape as input.
    """

    y = torch.empty_like(x)
    z = (x == 0)
    y[~z] = torch.sin(x[~z]) / x[~z]
    y[z] = 1.0
    return y

@torch.no_grad()
def _sb_notch_kernel(nyq_frac: float, width: float, *, device, dtype) -> torch.Tensor:
    """Create notch filter kernel.

    Combines low-pass and high-pass filters with Blackman windowing to create
    a notch filter that attenuates frequencies around a center frequency.
    
    Args:
        nyq_frac: Center frequency as fraction of Nyquist rate (0, 1].
        width: Bandwidth of the notch as fraction of Nyquist rate.
        device: Target device for the kernel tensor.
        dtype: Target dtype for the kernel tensor.
        
    Returns:
        Filter kernel tensor of shape [1, filter_length, 1].
    """
    assert 0 < nyq_frac <= 1.0
    pad = _PAD
    t = _T_IDX_F32.to(device=device, dtype=dtype)
    w = _BLACKMAN_F32.to(device=device, dtype=dtype)

    hlpf = _sinc(3 * (nyq_frac - width) * t) * w
    hlpf = hlpf / hlpf.sum()

    hhpf = _sinc(3 * (nyq_frac + width) * t) * w
    hhpf = hhpf / -hhpf.sum()
    hhpf[pad] += 1

    h = (hlpf + hhpf).view(1, -1, 1)
    return h

@torch.no_grad()
def freq_drop(
    waveform: torch.Tensor,
    bound_low: float = 1e-12,
    bound_high: float = 1.0,
    band_count_low: int = 1,
    band_count_high: int = 8,
    band_width: float = 0.10
) -> torch.Tensor:
    """Apply frequency dropout augmentation.

    Randomly drops frequency bands using notch filters. Each notch
    removes a narrow band of frequencies while preserving adjacent content.
    
    Args:
        waveform: Audio tensor in [T] or [C, T] format. Must be on CPU.
        bound_low: Lower bound for frequency band selection (fraction of Nyquist). Default: 1e-12.
        bound_high: Upper bound for frequency band selection (fraction of Nyquist). Default: 1.0.
        band_count_low: Minimum number of bands to suppress. Default: 1.
        band_count_high: Maximum number of bands to suppress. Default: 8.
        band_width: Width of each suppressed band (fraction of Nyquist). Default: 0.10.
        
    Returns:
        The input tensor modified in-place.
        
    Raises:
        AssertionError: If waveform is not on CPU or has wrong dimensions.
    """
    assert waveform.device.type == "cpu"
    assert waveform.ndim in (1, 2), "expected [T] or [C, T]"

    if waveform.ndim == 1:
        T = int(waveform.numel())
        view = waveform.view(T, 1)
    else:
        C, T = waveform.shape[0], waveform.shape[1]
        view = waveform.transpose(0, 1)

    if T == 0:
        return waveform

    bound_low = max(0.0, min(1.0, bound_low))
    bound_high = max(bound_low, min(1.0, bound_high))
    width = max(0.0, min(1.0, band_width))
    rng = bound_high - bound_low
    if rng <= 0 or width <= 0:
        return waveform

    # build composite filter in filter domain, starting from delta impulse
    drop = torch.zeros(1, _FILTER_LEN, 1, device=waveform.device, dtype=waveform.dtype)
    drop[0, _PAD, 0] = 1

    band_count = int(torch.randint(band_count_low, band_count_high + 1, ()))
    for _ in range(band_count):
        f = float(torch.rand(()) * rng + bound_low)
        f = max(1e-12, min(1.0, f))
        k = _sb_notch_kernel(f, width, device=waveform.device, dtype=waveform.dtype)
        # convolve kernel with current composite kernel
        drop = F.conv1d(drop.transpose(2, 1), k.transpose(2, 1), padding=_PAD).transpose(2, 1)

    x = view.transpose(0, 1).unsqueeze(0)
    C = x.size(1)
    w = drop.transpose(2, 1).expand(C, 1, _FILTER_LEN)
    y = F.conv1d(x, w, padding=_PAD, groups=C)

    view[:] = y.squeeze(0).transpose(0, 1)
    return waveform
