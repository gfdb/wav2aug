from typing import Final

import torch

# tuned bounds @16 kHz (scale linearly with sample_rate)
_CHUNK_SIZE_LOW: Final[int] = 0
_CHUNK_SIZE_HIGH: Final[int] = 4000
_CHUNK_COUNT_LOW: Final[int] = 1
_CHUNK_COUNT_HIGH: Final[int] = 8
_BASE_SAMPLE_RATE: Final[int] = 16_000

@torch.no_grad()
def time_drop(waveform: torch.Tensor, sample_rate: int = 16_000) -> torch.Tensor:
    """Zero out random time blocks for temporal dropout augmentation.

    Randomly selects 1-8 time segments and sets them to zero. Segment sizes
    are scaled proportionally with sample rate. Overlapping segments are allowed.

    Args:
        waveform: Audio tensor in [T] or [C, T] format. Must be on CPU.
        sample_rate: Audio sample rate in Hz for duration scaling.

    Returns:
        The input tensor modified in-place.
        
    Raises:
        AssertionError: If waveform is not on CPU or has wrong dimensions.
    """
    assert waveform.device.type == "cpu", "CPU only"
    assert waveform.ndim in (1, 2), "expected [T] or [C, T]"

    if waveform.ndim == 1:
        total_time = int(waveform.numel())
        view = waveform.view(total_time, 1)
    else:
        C, T = int(waveform.shape[0]), int(waveform.shape[1])
        total_time = T
        view = waveform.transpose(0, 1)

    if total_time == 0:
        return waveform

    chunk_count = int(torch.randint(_CHUNK_COUNT_LOW, _CHUNK_COUNT_HIGH + 1, ()))

    if sample_rate != _BASE_SAMPLE_RATE:
        scale = float(sample_rate) / float(_BASE_SAMPLE_RATE)
        min_len = max(1, int(_CHUNK_SIZE_LOW * scale))
        max_len = max(min_len, int(_CHUNK_SIZE_HIGH * scale))
    else:
        min_len = _CHUNK_SIZE_LOW
        max_len = _CHUNK_SIZE_HIGH

    # Clamp max_len to not exceed total available time
    max_len = min(max_len, total_time)
    min_len = min(min_len, max_len)

    lengths = torch.randint(min_len, max_len + 1, (chunk_count,), dtype=torch.long)

    start_max = (total_time - lengths).clamp_min(0)

    starts = torch.empty(chunk_count, dtype=torch.long)
    for i in range(chunk_count):
        hi = int(start_max[i].item()) + 1
        starts[i] = torch.randint(0, hi, ())

    for i in range(chunk_count):
        L = int(lengths[i])
        if L <= 0:
            continue
        s = int(starts[i])
        view.narrow(0, s, L).zero_()

    return waveform
