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

    Args:
        waveforms (torch.Tensor): The input waveforms. Shape [batch, time].
        prob (float, optional): The probability of flipping each waveform. Defaults to 0.6.

    Raises:
        AssertionError: If waveforms are not 2D shaped [batch, time].
        AssertionError: If waveforms are not on the CUDA device.
        AssertionError: If waveforms are empty.

    Returns:
        torch.Tensor: The waveforms with inverted polarity.
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
