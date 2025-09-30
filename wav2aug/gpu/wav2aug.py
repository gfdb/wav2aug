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


_INTEGER_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)


class Wav2Aug:
    """Apply two random GPU augmentations to a batch of waveforms."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        self._base_ops: List[Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]] = [
            lambda x, lengths: add_noise(x, self.sample_rate),
            lambda x, lengths: add_babble_noise(x),
            lambda x, lengths: chunk_swap(x),
            lambda x, lengths: freq_drop(x),
            lambda x, lengths: invert_polarity(x),
            lambda x, lengths: rand_amp_clip(x),
            lambda x, lengths: rand_amp_scale(x),
            lambda x, lengths: speed_perturb(x, lengths=lengths),
            lambda x, lengths: time_dropout(x, sample_rate=self.sample_rate),
        ]

    @torch.no_grad()
    def __call__(
        self,
        waveforms: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply two distinct augmentations to the input batch."""
        if waveforms.ndim != 2:
            raise AssertionError("expected waveforms shaped [batch, time]")
        if waveforms.device.type != "cuda":
            raise AssertionError("Wav2Aug expects CUDA tensors")
        if waveforms.numel() == 0:
            return waveforms if lengths is None else (waveforms, lengths)

        if lengths is not None:
            if lengths.ndim != 1 or lengths.numel() != waveforms.size(0):
                raise AssertionError("expected lengths shaped [batch]")
            if lengths.device != waveforms.device:
                raise AssertionError("lengths tensor must share device with waveforms")
            if not (torch.is_floating_point(lengths) or lengths.dtype in _INTEGER_DTYPES):
                raise AssertionError("lengths tensor must use a float or integer dtype")

        indices = torch.randperm(len(self._base_ops), device=waveforms.device)[:2].tolist()
        for idx in indices:
            op = self._base_ops[idx]
            waveforms = op(waveforms, lengths)
        return waveforms if lengths is None else (waveforms, lengths)


__all__ = ["Wav2Aug"]
