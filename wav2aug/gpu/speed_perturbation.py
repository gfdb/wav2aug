from __future__ import annotations

import torch
import torchaudio


@torch.no_grad()
def speed_perturb(
    waveforms: torch.Tensor,
    sample_rate: int,
    *,
    speed_changes: tuple[float, ...] = (0.9, 1.0, 1.1),
) -> torch.Tensor:
    """Apply a single random speed factor to every waveform in the batch.

    Args:
        waveforms (torch.Tensor): The input waveforms.
        sample_rate (int): The sample rate of the audio.
        speed_changes (tuple[float, ...], optional): The speed changes to apply. Defaults to (0.9, 1.0, 1.1).

    Raises:
        AssertionError: If the input waveforms are not 2D or if the batch size is 0.

    Returns:
        torch.Tensor: The perturbed waveforms.
    """

    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    batch, total_time = waveforms.shape
    if batch == 0 or total_time < 2:
        return waveforms

    device = waveforms.device

    speed_idx = torch.randint(len(speed_changes), (1,), device=device)
    speed = speed_changes[speed_idx]

    if speed == 1.0:
        return waveforms

    resampled = torchaudio.functional.resample(
        waveforms,
        orig_freq = sample_rate,
        new_freq = sample_rate * speed,
    ).to(waveforms.device)

    return resampled


__all__ = ["speed_perturb"]
