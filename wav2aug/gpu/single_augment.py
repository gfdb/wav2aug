"""Single augmentation selector for A/B testing with SpeechBrain."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from wav2aug.gpu import (add_babble_noise, add_noise, chunk_swap, freq_drop,
                         invert_polarity, rand_amp_clip, rand_amp_scale,
                         speed_perturb, time_dropout)

# Mapping of augmentation names to their wav2aug implementations
# Names match the speechbrain equivalents for easy comparison
AUGMENTATION_REGISTRY: Dict[str, str] = {
    "add_noise": "add_noise",
    "speed_perturb": "speed_perturb",
    "drop_freq": "freq_drop",
    "drop_chunk": "time_dropout",
    "clipping": "rand_amp_clip",
    "rand_amp": "rand_amp_scale",
    "sign_flip": "invert_polarity",
    "chunk_swap": "chunk_swap",
    "babble_noise": "add_babble_noise",
    # Also allow wav2aug native names
    "freq_drop": "freq_drop",
    "time_dropout": "time_dropout",
    "rand_amp_clip": "rand_amp_clip",
    "rand_amp_scale": "rand_amp_scale",
    "invert_polarity": "invert_polarity",
    "add_babble_noise": "add_babble_noise",
}


class SingleAugment:
    """Applies a single specified augmentation for isolated performance testing.
    
    This class is designed for A/B testing individual augmentations against
    their SpeechBrain equivalents to identify performance discrepancies.
    
    Usage with SpeechBrain hparams (usa.yaml):
    ```yaml
    # Select augmentation via CLI: --augmentation_name=drop_chunk
    augmentation_name: drop_chunk
    
    wav_augment: !new:wav2aug.gpu.SingleAugment
        sample_rate: !ref <sample_rate>
        augmentation_name: !ref <augmentation_name>
    ```
    
    Then run with: python train.py hparams/usa.yaml --augmentation_name=drop_freq
    
    Available augmentation names (SpeechBrain names -> wav2aug):
        - add_noise      -> add_noise
        - speed_perturb  -> speed_perturb  
        - drop_freq      -> freq_drop
        - drop_chunk     -> time_dropout
        - clipping       -> rand_amp_clip
        - rand_amp       -> rand_amp_scale
        - sign_flip      -> invert_polarity
        - chunk_swap     -> chunk_swap
        - babble_noise   -> add_babble_noise
        
    You can also use wav2aug native names directly.
    """

    def __init__(
        self,
        sample_rate: int,
        augmentation_name: str,
        # Optional parameters to match speechbrain config values
        # Noise addition params
        snr_low: float = 0.0,
        snr_high: float = 10.0,
        # Speed perturbation params
        speed_changes: tuple = (0.9, 1.0, 1.1),
        # Frequency dropout params
        drop_freq_low: float = 0.0,
        drop_freq_high: float = 1.0,
        drop_freq_count_low: int = 1,
        drop_freq_count_high: int = 8,
        drop_freq_width: float = 0.1,
        # Time dropout (drop_chunk) params
        drop_chunk_count_low: int = 1,
        drop_chunk_count_high: int = 8,
        drop_chunk_length_low: int = 0,
        drop_chunk_length_high: int = 4000,
        # Clipping params
        clip_low: float = 0.0,
        clip_high: float = 0.75,
        # Random amplitude params
        amp_low: float = 0.05,
        amp_high: float = 0.5,
        # Polarity inversion params
        sign_flip_prob: float = 0.6,
        # Babble noise params
        babble_snr_low: float = 0.0,
        babble_snr_high: float = 20.0,
    ) -> None:
        """Initialize the single augmentation selector.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            augmentation_name: Name of the augmentation to apply.
                Use 'none' or 'identity' to disable augmentation.
            **kwargs: Additional parameters passed to specific augmentations.
        """
        self.sample_rate = int(sample_rate)
        self.augmentation_name = augmentation_name.lower().strip()
        
        # Store all parameters for the augmentations
        self._params = {
            "snr_low": snr_low,
            "snr_high": snr_high,
            "speed_changes": speed_changes,
            "drop_freq_low": drop_freq_low,
            "drop_freq_high": drop_freq_high,
            "drop_freq_count_low": drop_freq_count_low,
            "drop_freq_count_high": drop_freq_count_high,
            "drop_freq_width": drop_freq_width,
            "drop_chunk_count_low": drop_chunk_count_low,
            "drop_chunk_count_high": drop_chunk_count_high,
            "drop_chunk_length_low": drop_chunk_length_low,
            "drop_chunk_length_high": drop_chunk_length_high,
            "clip_low": clip_low,
            "clip_high": clip_high,
            "amp_low": amp_low,
            "amp_high": amp_high,
            "sign_flip_prob": sign_flip_prob,
            "babble_snr_low": babble_snr_low,
            "babble_snr_high": babble_snr_high,
        }
        
        # Resolve the augmentation name
        if self.augmentation_name in ("none", "identity", ""):
            self._op: Optional[Callable] = None
        elif self.augmentation_name in AUGMENTATION_REGISTRY:
            canonical_name = AUGMENTATION_REGISTRY[self.augmentation_name]
            self._op = self._build_operation(canonical_name)
        else:
            available = sorted(set(AUGMENTATION_REGISTRY.keys()))
            raise ValueError(
                f"Unknown augmentation: '{augmentation_name}'. "
                f"Available: {available}"
            )

    def _build_operation(
        self, name: str
    ) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """Build the augmentation operation with captured parameters."""
        p = self._params
        sr = self.sample_rate
        
        if name == "add_noise":
            return lambda x, lengths: add_noise(
                x, sr, snr_low=p["snr_low"], snr_high=p["snr_high"]
            )
        elif name == "speed_perturb":
            return lambda x, lengths: speed_perturb(
                x, sr, speed_changes=tuple(p["speed_changes"])
            )
        elif name == "freq_drop":
            return lambda x, lengths: freq_drop(
                x,
                bound_low=p["drop_freq_low"],
                bound_high=p["drop_freq_high"],
                band_count_low=p["drop_freq_count_low"],
                band_count_high=p["drop_freq_count_high"],
                band_width=p["drop_freq_width"],
            )
        elif name == "time_dropout":
            return lambda x, lengths: time_dropout(
                x,
                sr,
                lengths=lengths,
                chunk_count_low=p["drop_chunk_count_low"],
                chunk_count_high=p["drop_chunk_count_high"],
                chunk_size_low=p["drop_chunk_length_low"],
                chunk_size_high=p["drop_chunk_length_high"],
            )
        elif name == "rand_amp_clip":
            return lambda x, lengths: rand_amp_clip(
                x, clip_low=p["clip_low"], clip_high=p["clip_high"]
            )
        elif name == "rand_amp_scale":
            return lambda x, lengths: rand_amp_scale(
                x, amp_low=p["amp_low"], amp_high=p["amp_high"]
            )
        elif name == "invert_polarity":
            return lambda x, lengths: invert_polarity(x, prob=p["sign_flip_prob"])
        elif name == "chunk_swap":
            return lambda x, lengths: chunk_swap(x)
        elif name == "add_babble_noise":
            return lambda x, lengths: add_babble_noise(
                x, snr_low=p["babble_snr_low"], snr_high=p["babble_snr_high"]
            )
        else:
            raise ValueError(f"No implementation for: {name}")

    @torch.no_grad()
    def __call__(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the selected augmentation.
        
        Args:
            waveforms: Input waveforms. Shape [batch, time].
            lengths: Relative valid lengths of each waveform (0-1). Shape [batch].
            
        Returns:
            Tuple of (augmented_waveforms, lengths) to match SpeechBrain interface.
        """
        if waveforms.ndim != 2:
            raise AssertionError("expected waveforms shaped [batch, time]")
            
        if lengths is None:
            lengths = torch.ones(waveforms.size(0), device=waveforms.device)
            
        if self._op is None:
            # Identity - no augmentation
            return waveforms, lengths
            
        augmented = self._op(waveforms, lengths)
        return augmented, lengths

    def replicate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """No-op label replication for compatibility with SpeechBrain.
        
        Since SingleAugment doesn't multiply batches, labels are returned as-is.
        """
        return labels

    @classmethod
    def list_augmentations(cls) -> list[str]:
        """Return list of available augmentation names."""
        return sorted(set(AUGMENTATION_REGISTRY.keys()))


__all__ = ["SingleAugment", "AUGMENTATION_REGISTRY"]
