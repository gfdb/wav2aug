from typing import Final, List

import torch
import torch.nn.functional as F

# these are percents
_SPEED_CHANGES: Final[List[int]] = [90, 100, 110]

@torch.no_grad()
def speed_perturb(waveform: torch.Tensor) -> torch.Tensor:
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

    speed = float(_SPEED_CHANGES[int(torch.randint(0, len(_SPEED_CHANGES), ()))]) / 100.0

    if speed == 1.0:
        return waveform

    if waveform.ndim == 1:
        x = waveform.view(1, 1, -1)
        T = x.shape[-1]
        if T < 2:
            return waveform
        new_T = max(1, int(round(T / speed)))

        y = F.interpolate(x, size=new_T, mode="linear", align_corners=True)
        return y.view(new_T)

    C, T = waveform.shape
    if T < 2:
        return waveform
    
    new_T = max(1, int(round(T / speed)))

    x = waveform.unsqueeze(0)
    y = F.interpolate(x, size=new_T, mode="linear", align_corners=True)
    return y.squeeze(0)
