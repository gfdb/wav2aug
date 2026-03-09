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
        AssertionError: If waveforms are empty.

    Returns:
        torch.Tensor: The waveforms with inverted polarity.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    if waveforms.numel() == 0:
        return waveforms

    batch = waveforms.size(0)

    # Build a per-sample sign multiplier: -1 for flipped, +1 for kept
    flips = torch.rand(batch, device=waveforms.device) < prob
    signs = torch.where(flips, -1.0, 1.0).unsqueeze(1)
    return waveforms * signs


__all__ = ["invert_polarity"]
