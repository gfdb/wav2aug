from __future__ import annotations

from functools import lru_cache

import torch
import torchaudio


@lru_cache(maxsize=32)
def _get_resampler(orig_freq: int, new_freq: int, device: str) -> torchaudio.transforms.Resample:
    """Get a cached Resample transform. The kernel is computed once and reused."""
    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resampler.to(device)


@torch.no_grad()
def speed_perturb(
    waveforms: torch.Tensor,
    sample_rate: int,
    *,
    speeds: tuple[int, ...] = (90, 100, 110),
) -> torch.Tensor:
    """Apply a single random speed factor to every waveform in the batch.

    Speed values are percentages (e.g., 90 = 90% = slower, 110 = 110% = faster).
    Using integer percentages ensures good GCD with common sample rates,
    which makes the resampling filter efficient.

    Args:
        waveforms (torch.Tensor): The input waveforms. Shape [batch, time].
        sample_rate (int): The sample rate of the audio.
        speeds (tuple[int, ...], optional): Speed percentages to choose from.
            Defaults to (90, 100, 110).

    Returns:
        torch.Tensor: The perturbed waveforms.
    """

    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    batch, total_time = waveforms.shape
    if batch == 0 or total_time < 2:
        return waveforms

    speed_idx = torch.randint(len(speeds), (1,)).item()
    speed_pct = speeds[speed_idx]

    if speed_pct == 100:
        return waveforms

    # Use integer percentage for good GCD with sample_rate
    # e.g., 16000 * 90 // 100 = 14400, GCD(16000, 14400) = 1600
    new_freq = sample_rate * speed_pct // 100
    device_str = str(waveforms.device)
    resampler = _get_resampler(sample_rate, new_freq, device_str)
    
    return resampler(waveforms)


__all__ = ["speed_perturb"]
