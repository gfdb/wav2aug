from .time_dropout import time_dropout
from .frequency_dropout import freq_drop
from .speed_perturbation import speed_perturb

import torch

class SpecAugment:

    def __init__(self, sample_rate) -> None:
        self.sample_rate = sample_rate
    
    @torch.no_grad()
    def __call__(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: 
        time_dropped = time_dropout(waveforms, self.sample_rate, lengths = lengths)
        freq_dropped = freq_drop(time_dropped)
        return speed_perturb(freq_dropped, self.sample_rate) 
    
    def replicate_labels(
        self,
        labels: torch.Tensor,
        deep: bool = True,
    ) -> torch.Tensor:
        """Replicate labels to match the augmented batch.

        Since Wav2Aug does not change the batch size, this simply returns
        the labels as-is, mainly for API compatibility with Wav2AugViews.

        Args:
            labels: Labels tensor of shape [batch, ...].
            deep: Ignored for Wav2Aug (kept for API compatibility).

        Returns:
            The same labels tensor unchanged.
        """
        return labels

__all__ = ["SpecAugment"]
