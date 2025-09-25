import torch


@torch.no_grad()
def rand_amp_clip(
    waveform: torch.Tensor,
    clip_low: float = 0.0,
    clip_high: float = 0.75,
    eps: float = 1e-12
) -> torch.Tensor:
    """Apply random amplitude clipping with peak normalization.

    Normalizes by per channel peak, clips to random threshold, then rescales
    to original peak. Uses single clip value to preserve channel balance.

    Args:
        waveform: Audio tensor in [T] or [C, T] format. Must be on CPU.
        clip_low: Minimum clip threshold as fraction of peak. Default: 0.0.
        clip_high: Maximum clip threshold as fraction of peak. Default: 0.75.
        eps: Small constant to prevent division by zero. Default: 1e-12.

    Returns:
        The input tensor modified in-place.
        
    Raises:
        AssertionError: If waveform is not on CPU or has wrong dimensions.
    """
    assert waveform.device.type == "cpu", "CPU only"
    assert waveform.ndim in (1, 2), "expected [T] or [C, T]"
    if waveform.numel() == 0:
        return waveform

    if waveform.ndim == 1:
        peak = waveform.abs().max().clamp_min(1.0)
    else:
        peak = waveform.abs().amax(dim=1, keepdim=True).clamp_min(1.0)

    waveform.div_(peak)

    clip = torch.rand((), device=waveform.device) * (clip_high - clip_low) + clip_low
    clip = clip.clamp_min(eps).to(waveform.dtype)

    waveform.clamp_(-clip, clip)
    waveform.mul_(peak / clip)

    return waveform
