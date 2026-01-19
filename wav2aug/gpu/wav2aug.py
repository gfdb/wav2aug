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
        noise_preload: bool = True,
        top_k: int | None = None,
    ) -> None:
        """Initialize Wav2Aug.

        Args:
            sample_rate: Audio sample rate in Hz.
            noise_dir: Directory containing noise files. If None, will use the
                default cached noise pack (auto-downloaded if needed).
            noise_preload: If True (default), preload all noise files into CPU RAM
                at initialization for fast sampling. If False, load files on-demand.
            top_k: Number of top augmentations to use, ordered by effectiveness.
                If None, all 9 augmentations are used. Common values: 3, 6, or 9.
                Order (best to worst): Noise Addition, Freq Drop, Time Drop,
                Speed Perturb, Amp Clip, Chunk Swap, Babble Noise, Amp Scale,
                Polarity Inversion.
        """
        self.sample_rate = int(sample_rate)

        # Initialize noise loader
        if noise_dir is None:
            from wav2aug.data.fetch import ensure_pack

            noise_dir = ensure_pack("pointsource_noises")
        self._noise_loader = NoiseLoader(noise_dir, sample_rate, preload=noise_preload)

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

        # select top-k ops
        if top_k is None:
            top_k = len(all_ops)
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


class Wav2AugViews:
    """Creates multiple views of a batch: one unaugmented original plus augmented copies.

    Each augmented copy receives an independent augmentation pass via Wav2Aug.
    All copies are padded to the same time dimension and concatenated along the batch axis.
    """

    def __init__(
        self,
        augmenter: Wav2Aug,
        views: int = 4,
    ) -> None:
        """Initialize Wav2AugViews.

        Args:
            augmenter: A Wav2Aug instance used to augment each copy.
            views: Total number of views to produce (including the original).
                Must be >= 2. For example, views=2 means 1 original + 1 augmented.
        """
        if views < 2:
            raise ValueError("views must be >= 2")
        self._augmenter = augmenter
        self._views = views

    @torch.no_grad()
    def __call__(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Create multiple views of the input batch.

        Args:
            waveforms: Input waveforms of shape [batch, time].
            lengths: Optional relative lengths of shape [batch], values in (0, 1].

        Returns:
            Waveforms of shape [batch * views, max_time] and optionally updated lengths.
            The first `batch` samples are the unaugmented originals.
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

        original = waveforms
        orig_time = original.shape[1]

        # Collect all views: (waveform, time_before_padding)
        view_data: list[tuple[torch.Tensor, int]] = [(original, orig_time)]

        for _ in range(self._views - 1):
            copy = original.clone()
            augmented = self._augmenter(copy, lengths)
            # Handle case where augmenter returns tuple
            if isinstance(augmented, tuple):
                augmented = augmented[0]
            view_data.append((augmented, augmented.shape[1]))

        # Find max time across all views
        max_time = max(t for _, t in view_data)

        # Pad each view to max_time and compute adjusted lengths
        padded_views: list[torch.Tensor] = []
        adjusted_lengths: list[torch.Tensor] = []

        for wav, time_before_pad in view_data:
            if wav.shape[1] < max_time:
                pad_amount = max_time - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, pad_amount))
            padded_views.append(wav)

            if lengths is not None:
                # Adjust relative lengths: original ratio scaled by time change
                scale = time_before_pad / max_time
                adjusted_lengths.append(lengths * scale)

        # Concatenate along batch dimension
        out_waveforms = torch.cat(padded_views, dim=0)

        if lengths is not None:
            out_lengths = torch.cat(adjusted_lengths, dim=0)
            return out_waveforms, out_lengths

        return out_waveforms

    def replicate_labels(
        self,
        labels: torch.Tensor,
        deep: bool = True,
    ) -> torch.Tensor:
        """Replicate labels to match the multi-view batch.

        The output will have labels repeated `views` times along the batch
        dimension to match the output of __call__.

        Args:
            labels: Labels tensor of shape [batch, ...].
            deep: If True (default), creates a contiguous copy of the repeated
                labels. If False, returns an expanded view that shares storage
                with the original tensor (saves memory but modifications affect
                all copies).

        Returns:
            Labels tensor of shape [batch * views, ...].
        """
        if deep:
            return labels.repeat(self._views, *([1] * (labels.ndim - 1)))
        else:
            expanded = labels.unsqueeze(0).expand(
                self._views, *labels.shape
            )
            return expanded.reshape(-1, *labels.shape[1:])


__all__ = ["Wav2Aug", "Wav2AugViews"]
