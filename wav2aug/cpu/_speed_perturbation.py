from typing import Final, List

import torch
import torchaudio


@torch.no_grad()
def speed_perturb(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    speed_changes: tuple[float, ...] = (0.9, 1.0, 1.1),
) -> torch.Tensor:
    """Apply speed perturbation by resampling audio.

    Randomly selects speed factor from {0.9, 1.0, 1.1} and adjusts
    waveform duration accordingly using linear interpolation.

    Args:
        waveform: Audio tensor in [T] or [C, T] format. Must be on CPU.

    Returns:
        Resampled waveform with adjusted duration.
        
    Raises:
        AssertionError: If waveform is not on CPU or has wrong dimensions.
    """
    assert waveform.device.type == "cpu"
    assert waveform.ndim in (1, 2), "expect [T] or [C, T]"
    assert len(speed_changes) > 0, "speed_changes must be non-empty"
    
    dim = waveform.ndim

    speed = speed_changes[torch.randint(len(speed_changes), (1,)).item()]

    if speed == 1.0:
        return waveform
    
    if dim == 1:
        waveform = waveform.unsqueeze(0)

    y = torchaudio.functional.resample(
        waveform = waveform,
        orig_freq = sample_rate,
        new_freq = int(sample_rate * speed)
    )

    return y.squeeze(0) if dim == 1 else y
