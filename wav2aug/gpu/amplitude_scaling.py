from __future__ import annotations

import torch


@torch.no_grad()
def rand_amp_scale(
    waveforms: torch.Tensor,
    *,
    amp_low: float = 0.05,
    amp_high: float = 0.5,
) -> torch.Tensor:
    """Random amplitude scaling for batched waveforms.

    Normalizes each waveform to [-1, 1] then applies a random amplitude
    scale factor.

    Args:
        waveforms: Tensor of shape [batch, time].
        amp_low: Minimum amplitude scale factor.
        amp_high: Maximum amplitude scale factor.

    Returns:
        The input ``waveforms`` tensor, modified in-place.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    if waveforms.numel() == 0:
        return waveforms

    device = waveforms.device
    dtype = waveforms.dtype

    # Normalize to [-1, 1] by dividing by absolute max 
    abs_max = waveforms.abs().amax(dim=1, keepdim=True)
    # Avoid division by zero for silent signals
    abs_max = abs_max.clamp_min(1e-14)
    waveforms.div_(abs_max)

    # Per-sample scaling factors
    scales = torch.rand((waveforms.size(0), 1), device=device, dtype=dtype)
    scales = scales * (amp_high - amp_low) + amp_low
    waveforms.mul_(scales)
    return waveforms


__all__ = ["rand_amp_scale"]
