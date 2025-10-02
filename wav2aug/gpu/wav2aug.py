from __future__ import annotations

from typing import Callable, List
import os

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

_CHECK_ENV = "WAV2AUG_GPU_CHECK_FINITE"


class Wav2Aug:
    """Apply two random GPU augmentations to a batch of waveforms.

    Set environment variable WAV2AUG_GPU_CHECK_FINITE=1 to enable a runtime
    finite check after each op; if non-finite values are produced the op's
    effect is reverted and a warning is printed. This is intended for
    debugging training-time NaNs and has a small performance cost.
    """

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

        check_finite = bool(int(os.getenv(_CHECK_ENV, "0")))
        indices = torch.randperm(len(self._base_ops), device=waveforms.device)[:2].tolist()
        for idx in indices:
            op = self._base_ops[idx]
            before = waveforms.clone() if check_finite else None
            waveforms = op(waveforms, lengths)
            if check_finite and not torch.isfinite(waveforms).all():
                if before is not None:
                    waveforms.copy_(before)
                print(f"[wav2aug] reverted non-finite output from op index {idx}", flush=True)
        return waveforms if lengths is None else (waveforms, lengths)


__all__ = ["Wav2Aug"]
