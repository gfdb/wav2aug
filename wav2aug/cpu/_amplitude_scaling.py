import torch


@torch.no_grad()
def rand_amp_scale(
    waveform: torch.Tensor,
    amp_low: float = 0.05,
    amp_high: float = 0.5
) -> torch.Tensor:
    """Apply random amplitude scaling with peak normalization.

    Normalizes to unit peak per channel, then scales by random factor
    sampled uniformly between amp_low and amp_high.

    Args:
        waveform: Audio tensor in [T] or [C, T] format. Must be on CPU.
        amp_low: Minimum amplitude scale factor. Default: 0.05.
        amp_high: Maximum amplitude scale factor. Default: 0.5.

    Returns:
        The input tensor modified in-place.
        
    Raises:
        AssertionError: If waveform is not on CPU or has wrong dimensions.
    """
    assert waveform.device.type == "cpu"
    assert waveform.ndim in (1, 2)
    if waveform.numel() == 0:
        return waveform

    if waveform.ndim == 1:
        denom = waveform.abs().max().clamp_min(1.0)
    else:
        denom = waveform.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    waveform.div_(denom)

    amp = torch.rand((), device=waveform.device) * (amp_high - amp_low) + amp_low
    waveform.mul_(amp.to(waveform.dtype))
    return waveform
