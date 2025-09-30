from __future__ import annotations

import torch


@torch.no_grad()
def invert_polarity(
    waveforms: torch.Tensor,
    prob: float = 0.6,
) -> torch.Tensor:
    """Random polarity inversion per waveform in the batch."""
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("invert_polarity expects CUDA tensors")
    if waveforms.numel() == 0:
        return waveforms

    flip = torch.rand((), device=waveforms.device) < prob
    if bool(flip):
        waveforms.neg_()
    return waveforms


__all__ = ["invert_polarity"]
