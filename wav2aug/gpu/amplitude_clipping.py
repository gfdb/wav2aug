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
    peaks = waveforms.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    normalized = waveforms / peaks

    # Per-sample clip thresholds
    clip = torch.rand((waveforms.size(0), 1), device=device, dtype=dtype)
    clip = clip * (clip_high - clip_low) + clip_low
    clip = clip.clamp_min(eps)

    normalized = torch.minimum(normalized, clip)
    normalized = torch.maximum(normalized, -clip)

    scale = peaks / clip
    waveforms.copy_(normalized * scale)
    return waveforms


__all__ = ["rand_amp_clip"]
