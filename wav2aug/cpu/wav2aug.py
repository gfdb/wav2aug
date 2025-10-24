import random
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from wav2aug.cpu import (add_babble_noise, add_noise, chunk_swap, freq_drop,
                         invert_polarity, rand_amp_clip, rand_amp_scale,
                         speed_perturb, time_drop)


class Wav2Aug:
    """Applies two random augmentation to input waveform per call.

    Args:
        sample_rate (int): Sample rate in Hz (e.g., 16000, 44100).
        batch_size (int): Number of waveforms summed for babble noise.
    """

    def __init__(
        self, 
        sample_rate: int,
        batch_size: int
    ):
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        
        self._active_buffer: Optional[torch.Tensor] = None
        self._ready_buffer: Optional[torch.Tensor] = None
        
        self._base_ops: List[Callable] = [
            lambda x: add_noise(x, self.sample_rate),
            lambda x: chunk_swap(x),
            lambda x: freq_drop(x),
            lambda x: invert_polarity(x),
            lambda x: rand_amp_clip(x),
            lambda x: rand_amp_scale(x),
            lambda x: speed_perturb(x),
            lambda x: time_drop(x, self.sample_rate),
        ]

    @torch.no_grad()
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply two random augmentations to the waveform.
        
        Args:
            waveform: Input audio tensor [T] or [C, T], must be on CPU
            
        Returns:
            Augmented waveform with same shape as input
        """

        x = waveform
        assert x.ndim in (1, 2), "expected [T] or [C, T]"

        if x.numel() == 0:
            return x

        x = x.view(1, -1) if x.ndim == 1 else x

        self._update_buffers(x)

        if len(self._base_ops) < 2:
            # in practice this should never happen
            raise RuntimeError(
                "not enough augmentation operations available to apply"
            )

        ops = torch.randperm(len(self._base_ops))[:2]
        op1 = self._base_ops[ops[0]]
        op2 = self._base_ops[ops[1]]
        return op2(op1(x))

    @torch.no_grad()
    def _update_buffers(self, waveform: torch.Tensor) -> None:
        """Update the dual buffer system with a new waveform.
        
        The dual-buffer system works as follows:
        1. Accumulate waveforms in the active buffer until ``batch_size`` is reached
        2. When full, the active buffer becomes the ready buffer (available for babble noise)
        3. A new active buffer is initialized with the next waveform
        
        Buffers automatically grow to accommodate the longest waveform seen in the dataset.
        Shorter waveforms are zero-padded to match the current buffer length to maintain
        consistent tensor shapes for accumulation.
        
        Args:
            waveform: Waveform in [C, T] format.
        """
        if self._active_buffer is None:
            # init buffer with first waveform
            self._active_buffer = waveform.clone()

        if self._active_buffer.shape != waveform.shape:
            # handle length differences by padding to whoever is shorter
            buffer_len = self._active_buffer.shape[-1]
            waveform_len = waveform.shape[-1]

            if buffer_len < waveform_len:
                self._active_buffer = F.pad(self._active_buffer, (0, waveform_len - buffer_len))
            else:
                waveform = F.pad(waveform, (0, buffer_len - waveform_len))
            
            self._active_buffer.add_(waveform)

        if self._active_buffer.shape[-1] >= self.batch_size:
            self._ready_buffer = self._active_buffer
            # add babble noise to ops once ready buffer is available
            if len(self._base_ops) == 8:
                self._base_ops.append(
                    lambda x: add_babble_noise(x, self._ready_buffer.clone())
                )
            self._active_buffer = None  # will be reinitialized with next waveform