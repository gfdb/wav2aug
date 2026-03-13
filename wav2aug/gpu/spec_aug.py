from .time_dropout import time_dropout
from .frequency_dropout import freq_drop
from .speed_perturbtation import speed_perturb

import torch

@torch.no_grad()
def spec_aug(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    """
    waveforms = time_dropout(waveforms, lengths)
    waveforms = freq_drop(waveforms)
    return speed_perturb(waveforms)


