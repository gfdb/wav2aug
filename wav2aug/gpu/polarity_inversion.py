from __future__ import annotations

import torch


@torch.no_grad()
def invert_polarity(
    waveforms: torch.Tensor,
    prob: float = 0.6,
) -> torch.Tensor:
    """Random polarity inversion with per-sample Bernoulli decisions.

    Each waveform is independently flipped with probability ``prob``.
    Previously a single batch-wide decision could flip all samples.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("invert_polarity expects CUDA tensors")
    if waveforms.numel() == 0:
        return waveforms

    batch = waveforms.size(0)

    flips = torch.rand(batch, device=waveforms.device) < prob
    if flips.any():
        waveforms[flips] *= -1
    return waveforms


__all__ = ["invert_polarity"]
