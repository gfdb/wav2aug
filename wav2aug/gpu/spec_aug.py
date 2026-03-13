import torch

from .frequency_dropout import freq_drop
from .speed_perturbation import speed_perturb
from .time_dropout import time_dropout


class SpecAugment:

    def __init__(self, sample_rate) -> None:
        self.sample_rate = sample_rate
    
    @torch.no_grad()
    def __call__(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: 
        waveforms = time_dropout(waveforms, self.sample_rate, lengths = lengths)
        waveforms = freq_drop(waveforms)
        waveforms = speed_perturb(waveforms, self.sample_rate)
        return waveforms if lengths is None else (waveforms, lengths)
    
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
