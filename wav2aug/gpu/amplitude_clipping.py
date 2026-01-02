from __future__ import annotations

import torch


@torch.no_grad()
def rand_amp_clip(
    waveforms: torch.Tensor,
    *,
    clip_low: float = 0.0,
    clip_high: float = 0.75,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Random amplitude clipping for batched waveforms.

    Normalizes each waveform to [-1, 1], applies clipping, then restores
    the original amplitude scaled by the clip factor.

    Args:
        waveforms: Tensor of shape [batch, time].
        clip_low: Minimum clipping threshold as a fraction of peak.
        clip_high: Maximum clipping threshold as a fraction of peak.
        eps: Numerical floor to avoid division by zero.

    Returns:
        The input ``waveforms`` tensor, modified in-place.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    if waveforms.numel() == 0:
        return waveforms

    device = waveforms.device
    dtype = waveforms.dtype

    # Normalize to [-1, 1] by absolute max
    abs_max = waveforms.abs().amax(dim=1, keepdim=True)
    abs_max = abs_max.clamp_min(eps)
    waveforms.div_(abs_max)

    # Single clip value for entire batch (matches SpeechBrain)
    clip = torch.rand(1, device=device, dtype=dtype)
    clip = clip * (clip_high - clip_low) + clip_low
    clip = clip.clamp_min(eps)

    # Apply clipping
    waveforms.clamp_(-clip, clip)

    # Restore amplitude scaled by clip factor 
    waveforms.mul_(abs_max / clip)
    return waveforms


__all__ = ["rand_amp_clip"]
