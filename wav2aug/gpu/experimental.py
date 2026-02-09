from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch

from .amplitude_clipping import rand_amp_clip
from .amplitude_scaling import rand_amp_scale
from .chunk_swapping import chunk_swap
from .frequency_dropout import freq_drop
from .noise_addition import NoiseLoader, add_babble_noise, add_noise
from .polarity_inversion import invert_polarity
from .speed_perturbation import speed_perturb
from .time_dropout import time_dropout

# Type alias for an augmentation op: (waveforms, lengths) -> waveforms
AugOp = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]


class Wav2AugGrouped:
    """Applies two augmentations per batch using grouped sampling.

    Augmentations are organised into collision groups so that two
    augmentations from the same group are never selected together.
    Two distinct groups are drawn (without replacement) with
    size-weighted probabilities, and one augmentation is sampled
    uniformly from each selected group.  The application order of
    the two chosen augmentations is randomised.

    Groups (by failure / collision mode):
        G_noise = {add_noise, babble_noise}
        G_time  = {time_drop, chunk_swap}
        G_amp   = {amp_clip, amp_scale, polarity_inversion}
        G_sp    = {speed_perturb}
        G_fd    = {freq_drop}

    Group weights are proportional to group size (w_g = |G_g| / 9)
    so that singletons are not over-selected relative to larger groups.
    """

    def __init__(
        self,
        sample_rate: int,
        noise_dir: str | None = None,
        noise_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialise Wav2AugGrouped.

        Args:
            sample_rate: Audio sample rate in Hz.
            noise_dir: Directory containing noise files.  When *None* the
                default cached noise pack is used (auto-downloaded if needed).
            noise_dtype: Data type for storing preloaded noise in memory.
                Defaults to float32.  Use float16 for memory efficiency.
        """
        self.sample_rate = int(sample_rate)
        self.noise_dtype = noise_dtype

        if noise_dir is None:
            from wav2aug.data.fetch import ensure_pack

            noise_dir = ensure_pack("pointsource_noises")
        self._noise_loader = NoiseLoader(
            noise_dir, sample_rate, storage_dtype=noise_dtype
        )

        self._groups: Dict[str, List[Tuple[str, AugOp]]] = {
            "noise": [
                (
                    "add_noise",
                    lambda x, lengths: add_noise(
                        x, self._noise_loader, snr_low=0.0, snr_high=10.0
                    ),
                ),
                ("babble_noise", lambda x, lengths: add_babble_noise(x)),
            ],
            "time": [
                (
                    "time_drop",
                    lambda x, lengths: time_dropout(
                        x, sample_rate=self.sample_rate, lengths=lengths
                    ),
                ),
                ("chunk_swap", lambda x, lengths: chunk_swap(x)),
            ],
            "amp": [
                ("amp_clip", lambda x, lengths: rand_amp_clip(x)),
                ("amp_scale", lambda x, lengths: rand_amp_scale(x)),
                ("polarity_inversion", lambda x, lengths: invert_polarity(x)),
            ],
            "sp": [
                (
                    "speed_perturb",
                    lambda x, lengths: speed_perturb(x, sample_rate=self.sample_rate),
                ),
            ],
            "fd": [
                ("freq_drop", lambda x, lengths: freq_drop(x)),
            ],
        }

        self._group_names: List[str] = list(self._groups.keys())
        total_augs = sum(len(g) for g in self._groups.values())
        self._group_weights = torch.tensor(
            [len(self._groups[g]) / total_augs for g in self._group_names],
            dtype=torch.float64,
        )

    def _sample_two_groups(self) -> Tuple[str, str]:
        """Draw two distinct groups without replacement using size weights."""
        indices = torch.multinomial(self._group_weights, 2, replacement=False)
        return self._group_names[indices[0].item()], self._group_names[indices[1].item()]

    def _sample_op_from_group(self, group_name: str) -> AugOp:
        """Sample one augmentation uniformly from *group_name*."""
        ops = self._groups[group_name]
        idx = torch.randint(len(ops), (1,)).item()
        return ops[idx][1]

    @torch.no_grad()
    def __call__(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply two grouped-sampled augmentations to the input batch.

        Args:
            waveforms: Input waveforms shaped ``[batch, time]``.
            lengths: Optional lengths of each waveform, shaped ``[batch]``.

        Returns:
            Augmented waveforms (and lengths, if provided).

        Raises:
            AssertionError: On invalid input shapes or device mismatches.
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

        g1, g2 = self._sample_two_groups()

        op1 = self._sample_op_from_group(g1)
        op2 = self._sample_op_from_group(g2)

        waveforms = op1(waveforms, lengths)
        waveforms = op2(waveforms, lengths)

        return waveforms if lengths is None else (waveforms, lengths)


__all__ = ["Wav2AugGrouped"]
