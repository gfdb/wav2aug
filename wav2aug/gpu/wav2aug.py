from __future__ import annotations

from typing import Callable, List

import torch

from .amplitude_clipping import rand_amp_clip
from .amplitude_scaling import rand_amp_scale
from .chunk_swapping import chunk_swap
from .frequency_dropout import freq_drop
from .noise_addition import NoiseLoader, add_babble_noise, add_noise
from .polarity_inversion import invert_polarity
from .speed_perturbation import speed_perturb
from .time_dropout import time_dropout


class Wav2Aug:
    """Applies two random augmentations to a batch of waveforms when called."""

    def __init__(
        self,
        sample_rate: int,
        noise_dir: str | None = None,
        top_k: int = 9,
        noise_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize Wav2Aug.

        Args:
            sample_rate: Audio sample rate in Hz.
            noise_dir: Directory containing noise files. If None, will use the
                default cached noise pack (auto-downloaded if needed).
            top_k: Number of top augmentations to use, ordered by effectiveness.
                Default is 9 (all augmentations). Common values: 3, 6, or 9.
                Order (best to worst): Noise Addition, Freq Drop, Time Drop,
                Speed Perturb, Amp Clip, Chunk Swap, Babble Noise, Amp Scale,
                Polarity Inversion.
            noise_dtype: Data type for storing preloaded noise in memory.
                Defaults to float32. Use float16 for memory efficiency.
        """
        self.sample_rate = int(sample_rate)
        self.noise_dtype = noise_dtype

        # Initialize noise loader
        if noise_dir is None:
            from wav2aug.data.fetch import ensure_pack

            noise_dir = ensure_pack("pointsource_noises")
        self._noise_loader = NoiseLoader(
            noise_dir, sample_rate, storage_dtype=noise_dtype
        )

        # All ops ordered by effectiveness (best first)
        all_ops: List[Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]] = [
            # top 3
            lambda x, lengths: add_noise(
                x, self._noise_loader, snr_low=0.0, snr_high=10.0
            ),
            lambda x, lengths: freq_drop(x),
            lambda x, lengths: time_dropout(
                x, sample_rate=self.sample_rate, lengths=lengths
            ),
            # top 6
            lambda x, lengths: speed_perturb(x, sample_rate=self.sample_rate),
            lambda x, lengths: rand_amp_clip(x),
            lambda x, lengths: chunk_swap(x),
            # all 9
            lambda x, lengths: add_babble_noise(x),
            lambda x, lengths: rand_amp_scale(x),
            lambda x, lengths: invert_polarity(x),
        ]

        if top_k < 1 or top_k > len(all_ops):
            raise ValueError(f"top_k must be between 1 and {len(all_ops)}, got {top_k}")

        self._base_ops = all_ops[:top_k]

    @torch.no_grad()
    def __call__(
        self,
        waveforms: torch.Tensor,
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

        take = min(2, len(self._base_ops))
        indices = torch.randint(
            low=0,
            high=len(self._base_ops),
            size=(take,),
            device=waveforms.device,
        ).tolist()
        for idx in indices:
            op = self._base_ops[idx]
            waveforms = op(waveforms, lengths)
        return waveforms if lengths is None else (waveforms, lengths)


__all__ = ["Wav2Aug"]
