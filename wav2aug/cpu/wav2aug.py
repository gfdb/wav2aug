from typing import Callable, List
from multiprocessing import Value, Lock
import torch
import torch.nn.functional as F

from wav2aug.cpu import (add_babble_noise, add_noise, chunk_swap, freq_drop,
                         invert_polarity, rand_amp_clip, rand_amp_scale,
                         speed_perturb, time_drop)


class Wav2Aug:
    """Applies two random augmentations to each waveform, each batch will only see two 
    types of augmentations. Thread safe.

    Args:
        sample_rate (int): Sample rate in Hz (e.g., 16000, 44100).
        batch_size (int): Number of waveforms per batch. When this many
            waveforms have been processed, new augmentations are sampled.

    """
    def __init__(self, sample_rate: int, batch_size: int):
        self.sample_rate = int(sample_rate)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        
        # shared state for multi-worker coordination 
        # ensures consistent augmentations across processes
        self._batch_count = Value('i', 0)  
        self._op1_idx = Value('i', -1)
        self._op2_idx = Value('i', -1)
        
        # babble noise buffers, shared across processes
        self._current_buffer = None
        self._ready_buffer = None
        self._channels = Value('i', -1)
        
        # for shared state updates
        self._lock = Lock()

        # base operations (babble noise added later after first batch)
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
        self._base_ops_count = len(self._base_ops)

    def _update_babble_buffers(self, x: torch.Tensor) -> None:
        """Update babble noise buffers with current waveform.
        
        Args:
            x: Waveform in [C, T] format

        Note: This method assumes the caller already holds self._lock.
        """
        # set channels from first waveform
        if self._channels.value == -1:
            self._channels.value = x.size(0)
        
        # match channels for consistency
        target_channels = self._channels.value
        if x.size(0) == target_channels:
            x_matched = x
        elif x.size(0) == 1 and target_channels > 1:
            # mono -> multi-channel: duplicate across channels
            x_matched = x.repeat(target_channels, 1)
        else:
            # multi-channel -> mono or different channel count, average then repeat
            x_matched = x.mean(dim=0, keepdim=True).repeat(target_channels, 1)
        
        if self._current_buffer is None:
            # initialize current buffer with first waveform
            self._current_buffer = x_matched.clone()
        else:
            # add to current buffer, extending to longest duration
            current_len = self._current_buffer.size(1)
            new_len = x_matched.size(1)
            
            if new_len > current_len:
                # new waveform is longer, extend current buffer
                self._current_buffer = F.pad(self._current_buffer, (0, new_len - current_len))
                self._current_buffer.add_(x_matched)
            elif current_len > new_len:
                # current buffer is longer, pad new waveform and add
                x_padded = F.pad(x_matched, (0, current_len - new_len))
                self._current_buffer.add_(x_padded)
            else:
                # same length, direct addition
                self._current_buffer.add_(x_matched)
    
    def _complete_batch(self) -> None:
        """Complete current batch by moving buffers and enabling babble noise if needed.
        
        Note: This method assumes the caller already holds self._lock.
        """
        if self._current_buffer is not None:
            # move current buffer to ready buffer for next batch
            self._ready_buffer = self._current_buffer.clone()
            self._current_buffer = None
            
            # enable babble noise if not already available
            if len(self._base_ops) == self._base_ops_count:
                ready_buffer_ref = self._ready_buffer
                self._base_ops.append(
                    lambda x, buf=ready_buffer_ref: add_babble_noise(x, buf.clone())
                )

    def _sample_new_ops(self) -> None:
        """Sample new operations and update shared state.
        
        Note: This method assumes the caller already holds self._lock.
        """
        num_ops = len(self._base_ops)

        if num_ops < 2:
            raise RuntimeError("The number of operations in the pool must be at least 2," \
            "_base_ops was likely modified.")
        
        perm = torch.randperm(num_ops)
        
        self._op1_idx.value = int(perm[0])
        self._op2_idx.value = int(perm[1])

    @torch.no_grad()
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply two augmentations to the waveform.
        
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

        # update batch count and handle babble noise buffers
        with self._lock:
            count = self._batch_count.value
            
            # update buffers for babble noise
            self._update_babble_buffers(x)
            
            if count == 0:
                self._sample_new_ops()
                
            self._batch_count.value = (count + 1) % self.batch_size
            
            if self._batch_count.value == 0:
                self._complete_batch()
            
            if self._op1_idx.value <= 0 or self._op2_idx.value <= 0:
                raise RuntimeError("Augmentation operations have not been initialized.")

            op1 = self._base_ops[self._op1_idx.value]
            op2 = self._base_ops[self._op2_idx.value]

        y = op1(x)
        result = op2(y)
        
        return result.squeeze(0) if waveform.ndim == 1 else result