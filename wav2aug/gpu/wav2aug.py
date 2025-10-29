from __future__ import annotations

from typing import Callable, List

import torch

from wav2aug.gpu import (
    add_babble_noise,
    add_noise,
    chunk_swap,
    freq_drop,
    invert_polarity,
    rand_amp_clip,
    rand_amp_scale,
    speed_perturb,
    time_dropout,
)


class Wav2Aug:
    """Applies two random augmentations to a batch of waveforms when called."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        self._base_ops: List[
            Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
        ] = [
            lambda x, lengths: add_noise(x, self.sample_rate),
            lambda x, lengths: add_babble_noise(x),
            lambda x, lengths: chunk_swap(x),
            lambda x, lengths: freq_drop(x),
            lambda x, lengths: invert_polarity(x),
            lambda x, lengths: rand_amp_clip(x),
            lambda x, lengths: rand_amp_scale(x),
            lambda x, lengths: speed_perturb(x, sample_rate=self.sample_rate),
            lambda x, lengths: time_dropout(
                x, sample_rate=self.sample_rate, lengths=lengths
            ),
        ]

    @torch.no_grad()
    def __call__(
        self,
        waveforms: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Applies two distinct augmentations to the input batch.

        Args:
            waveforms (torch.Tensor): The input waveforms. Shape [batch, time].
            lengths (torch.Tensor | None, optional): The lengths of each waveform. Defaults to None.

        Raises:
            AssertionError: If waveforms are not 2D shaped [batch, time].
            AssertionError: If lengths is not None and has an invalid shape.
            AssertionError: If lengths is not None and does not share the same device as waveforms.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The augmented waveforms and lengths (if provided).
        """
        if waveforms.ndim != 2:
            raise AssertionError("expected waveforms shaped [batch, time]")

        if waveforms.numel() == 0:
            return waveforms if lengths is None else (waveforms, lengths)

        if lengths is not None:
            if lengths.ndim != 1 or lengths.numel() != waveforms.size(0):
                raise AssertionError("expected lengths shaped [batch]")
            if lengths.device != waveforms.device:
                raise AssertionError("lengths tensor must share device with waveforms")

        perm = torch.randperm(len(self._base_ops), device=waveforms.device)
        take = min(2, len(self._base_ops))
        indices = perm[:take].tolist()
        for idx in indices:
            op = self._base_ops[idx]
            waveforms = op(waveforms, lengths)
        return waveforms if lengths is None else (waveforms, lengths)


__all__ = ["Wav2Aug"]
