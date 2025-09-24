from __future__ import annotations

import torch


def invert_polarity(waveform: torch.Tensor, prob: float = 0.6) -> torch.Tensor:
    """Randomly invert waveform polarity.

    Flips the sign of audio samples as a regularization technique.
    Models typically don't require absolute polarity, making this
    a computationally cheap augmentation.

    Args:
        waveform: Input tensor of any shape containing audio samples.
        prob: Probability of inverting polarity. Default: 0.6.

    Returns:
        Waveform with inverted polarity (prob chance) or original.
    """
    if torch.rand(()) < (1.0 - prob):
        return waveform
    return -waveform