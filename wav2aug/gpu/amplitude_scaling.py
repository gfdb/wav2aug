from __future__ import annotations

import torch


@torch.no_grad()
def rand_amp_scale(
    waveforms: torch.Tensor,
    amp_low: float = 0.05,
    amp_high: float = 0.5,
) -> torch.Tensor:
    """Random amplitude scaling for batched CUDA waveforms.

    Args:
        waveforms: Tensor of shape [batch, time] on CUDA device.
        amp_low: Minimum amplitude scale factor.
        amp_high: Maximum amplitude scale factor.

    Returns:
        The input ``waveforms`` tensor, modified in-place.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("rand_amp_scale expects CUDA tensors")
    if waveforms.numel() == 0:
        return waveforms

    device = waveforms.device
    dtype = waveforms.dtype
    denom = waveforms.abs().amax(dim=1, keepdim=True).clamp_min(1.0)

    scales = torch.rand((), device=device, dtype=dtype)
    scales = scales * (amp_high - amp_low) + amp_low

    waveforms.mul_(scales / denom)
    return waveforms


__all__ = ["rand_amp_scale"]
