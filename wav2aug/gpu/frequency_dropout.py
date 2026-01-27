from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F


def _notch_filter(
    notch_freq: float,
    filter_length: int,
    notch_width: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a notch filter that removes a specific frequency.

    This is a direct port of SpeechBrain's notch_filter implementation.
    Reference: speechbrain/processing/signal_processing.py

    Args:
        notch_freq: Normalized frequency to notch out (0-1, fraction of Nyquist).
        filter_length: Length of the filter (should be odd).
        notch_width: Width of the notch as fraction of Nyquist.
        device: Device to create tensors on.
        dtype: Data type for tensors.

    Returns:
        Notch filter kernel of shape [1, filter_length, 1].
    """
    assert 0 < notch_freq <= 1
    assert filter_length % 2 != 0  # must be odd

    pad = filter_length // 2
    inputs = torch.arange(filter_length, device=device, dtype=dtype) - pad

    # Avoid frequencies that are too low
    notch_freq = notch_freq + notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x: torch.Tensor) -> torch.Tensor:
        # The zero is at the middle index
        return torch.cat(
            [
                torch.sin(x[:pad]) / x[:pad],
                torch.ones(1, device=device, dtype=dtype),
                torch.sin(x[pad + 1 :]) / x[pad + 1 :],
            ]
        )

    # Compute a low-pass filter with cutoff frequency notch_freq - notch_width
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf = hlpf * torch.blackman_window(
        filter_length, periodic=False, device=device, dtype=dtype
    )
    hlpf = hlpf / torch.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq + notch_width
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf = hhpf * torch.blackman_window(
        filter_length, periodic=False, device=device, dtype=dtype
    )
    hhpf = hhpf / -torch.sum(hhpf)
    hhpf[pad] = hhpf[pad] + 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).view(1, filter_length, 1)


def _convolve1d(
    waveform: torch.Tensor,
    kernel: torch.Tensor,
    padding: int,
) -> torch.Tensor:
    """1D convolution matching speechbrain's convolve1d behavior.

    Args:
        waveform: Input tensor of shape [batch, time, 1] or [1, filter_len, 1].
        kernel: Kernel tensor of shape [1, filter_len, 1].
        padding: Amount of zero-padding on each side.

    Returns:
        Convolved tensor of same shape as waveform.
    """
    # waveform: [B, T, 1] -> [B, 1, T]
    # kernel: [1, K, 1] -> [1, 1, K]
    x = waveform.transpose(1, 2)
    k = kernel.transpose(1, 2)

    y = F.conv1d(x, k, padding=padding)

    # [B, 1, T] -> [B, T, 1]
    return y.transpose(1, 2)


@torch.no_grad()
def freq_drop(
    waveforms: torch.Tensor,
    *,
    bound_low: float = 0.0,
    bound_high: float = 1.0,
    band_count_low: int = 1,
    band_count_high: int = 8,
    band_width: float = 0.1,
    clamp_abs: float = 8.0,
) -> torch.Tensor:
    """Frequency dropout using speechbrain's notch filter approach.

    This implementation builds a composite notch filter by convolving
    individual notch kernels together, then applies it to the entire batch
    at once, avoiding per-sample Python loops and GPU/CPU synchronization.

    A single random band count is drawn for the entire batch (uniform in
    [band_count_low, band_count_high]). The same frequencies are dropped
    across all samples, matching speechbrain's batch-wide approach.

    Args:
        waveforms: Input waveforms. Shape [batch, time].
        bound_low: Lower bound for normalized frequency (0-1). Defaults to 0.0.
        bound_high: Upper bound for normalized frequency (0-1). Defaults to 1.0.
        band_count_low: Minimum number of frequency bands to drop. Defaults to 1.
        band_count_high: Maximum number of frequency bands to drop. Defaults to 8.
        band_width: Width of each notch filter as fraction of Nyquist. Defaults to 0.1.
        clamp_abs: Absolute clamp value for output. Defaults to 8.0.

    Returns:
        Waveforms with frequency dropout applied.

    Raises:
        AssertionError: If waveforms are not 2D.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

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

    # Draw a single band count for the entire batch (enables vectorization)
    band_count = torch.randint(
        band_count_low, band_count_high + 1, (1,), device=device
    ).item()
    if band_count <= 0:
        return waveforms

    _FILTER_LEN: Final[int] = 101
    _PAD: Final[int] = _FILTER_LEN // 2

    # Start with delta function (identity filter)
    drop_filter = torch.zeros(1, _FILTER_LEN, 1, device=device, dtype=dtype)
    drop_filter[0, _PAD, 0] = 1.0

    # Sample frequencies and build composite filter by convolving notch kernels
    drop_frequencies = (
        torch.rand(band_count, device=device, dtype=dtype) * rng + bound_low
    )
    # Ensure freq + 2*width <= 1 to satisfy _notch_filter internals
    max_freq = max(1e-12, 1.0 - 2.0 * width - 1e-6)
    drop_frequencies = drop_frequencies.clamp(min=1e-12, max=max_freq)

    for i in range(band_count):
        freq = drop_frequencies[i].item()
        notch_kernel = _notch_filter(freq, _FILTER_LEN, width, device, dtype)
        drop_filter = _convolve1d(drop_filter, notch_kernel, _PAD)

    # Apply filter to entire batch at once
    # waveforms: [B, T] -> [B, T, 1] for _convolve1d
    dropped = waveforms.unsqueeze(-1)
    dropped = _convolve1d(dropped, drop_filter, _PAD)
    dropped = dropped.squeeze(-1)

    if clamp_abs is not None and clamp_abs > 0:
        dropped = dropped.clamp_(-clamp_abs, clamp_abs)

    dropped = torch.nan_to_num(dropped, nan=0.0, posinf=0.0, neginf=0.0)

    # Copy result back (in-place semantics)
    waveforms.copy_(dropped)
    return waveforms


__all__ = ["freq_drop"]
