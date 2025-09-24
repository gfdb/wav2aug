import random
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from wav2aug.cpu import (add_babble_noise, add_noise, chunk_swap, freq_drop,
                         invert_polarity, rand_amp_clip, rand_amp_scale,
                         speed_perturb, time_drop)

_EPS = 1e-14


def _match_channels(x: torch.Tensor, C: int) -> torch.Tensor:
    """Match tensor to target channel count.
    
    Args:
        x: Input tensor in [C, T] format.
        C: Target channel count.
        
    Returns:
        Tensor with channel dimension adjusted to C channels.
    """
    if x.size(0) == C:
        return x
    if x.size(0) == 1 and C > 1:
        return x.repeat(C, 1)
    return x.mean(dim=0, keepdim=True).repeat(C, 1)

class Wav2Aug:
    """Applies two random augmentation to input waveform per call.

    Args:
        sample_rate (int): Sample rate in Hz (e.g., 16000, 44100).
        buffer_capacity (int): Number of waveforms summed for babble noise.
            Default: 16.

    """

    def __init__(
        self, 
        sample_rate: int,
        *, 
        buffer_capacity: int = 16, 
    ):
        self.sample_rate = int(sample_rate)
        self.buffer_capacity = int(buffer_capacity)
        
        self._active_buffer: Optional[torch.Tensor] = None
        self._ready_buffer: Optional[torch.Tensor] = None
        self._active_count: int = 0
        self._C: Optional[int] = None
        
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

        # set channel count from first waveform
        if self._C is None:
            self._C = int(x.size(0))
        
        # match channels for variable channel datasets before buffer storage
        x_for_buffer = _match_channels(x, self._C)
        self._update_buffers(x_for_buffer)

        if len(self._base_ops) < 2:
            # in practice this should never happen
            return x
            
        op1, op2 = random.sample(self._base_ops, 2)
        y = op1(x)
        return op2(y)

    @torch.no_grad()
    def _update_buffers(self, waveform_ct: torch.Tensor) -> None:
        """Update the dual buffer system with a new waveform.
        
        The dual-buffer system works as follows:
        1. Accumulate waveforms in the active buffer until buffer_capacity is reached
        2. When full, the active buffer becomes the ready buffer (available for babble noise)
        3. A new active buffer is initialized with the next waveform
        
        Buffers automatically grow to accommodate the longest waveform seen in the dataset.
        Shorter waveforms are zero-padded to match the current buffer length to maintain
        consistent tensor shapes for accumulation.
        
        Args:
            waveform_ct: Waveform in [C, T] format with channels already matched via _match_channels.
        """
        if self._active_buffer is None:
            # Initialize buffer with first waveform
            self._active_buffer = waveform_ct.clone()
            self._active_count = 1
        else:
            # Handle length differences by padding to max length
            buffer_len = self._active_buffer.shape[1]
            waveform_len = waveform_ct.shape[1]
            max_len = max(buffer_len, waveform_len)
            
            # Pad buffer if needed
            if buffer_len < max_len:
                self._active_buffer = F.pad(self._active_buffer, (0, max_len - buffer_len))
            
            # Pad incoming waveform if needed
            if waveform_len < max_len:
                waveform_ct = F.pad(waveform_ct, (0, max_len - waveform_len))
            
            self._active_buffer.add_(waveform_ct)
            self._active_count += 1

        if self._active_count >= self.buffer_capacity:
            self._ready_buffer = self._active_buffer
            # add babble noise to ops once ready buffer is available
            if len(self._base_ops) == 8:
                self._base_ops.append(
                    lambda x: add_babble_noise(x, self._ready_buffer.clone())
                )
            self._active_buffer = None  # will be reinitialized with next waveform
            self._active_count = 0