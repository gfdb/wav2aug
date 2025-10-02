from __future__ import annotations

import torch

_CHUNK_SIZE_LOW: int = 0
_CHUNK_SIZE_HIGH: int = 4000
_CHUNK_COUNT_LOW: int = 1
_CHUNK_COUNT_HIGH: int = 8
_BASE_SAMPLE_RATE: int = 16_000


def _scaled_bounds(sample_rate: int) -> tuple[int, int]:
    if sample_rate != _BASE_SAMPLE_RATE:
        scale = float(sample_rate) / float(_BASE_SAMPLE_RATE)
        min_len = max(1, int(round(_CHUNK_SIZE_LOW * scale)))
        max_len = max(min_len, int(round(_CHUNK_SIZE_HIGH * scale)))
    else:
        min_len = _CHUNK_SIZE_LOW
        max_len = _CHUNK_SIZE_HIGH
    return min_len, max_len


@torch.no_grad()
def time_dropout(
    waveforms: torch.Tensor,
    sample_rate: int = _BASE_SAMPLE_RATE,
    *,
    chunk_count_low: int = _CHUNK_COUNT_LOW,
    chunk_count_high: int = _CHUNK_COUNT_HIGH,
    chunk_size_low: int = _CHUNK_SIZE_LOW,
    chunk_size_high: int = _CHUNK_SIZE_HIGH,
) -> torch.Tensor:
    """Apply time dropout with per-sample independent random zeroed segments.

    Each waveform draws its own number of chunks (uniform integer in
    [chunk_count_low, chunk_count_high]), random lengths in the (scaled)
    range [chunk_size_low, chunk_size_high], and random start positions.
    Segments are zeroed in-place.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms with shape [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("time_dropout expects CUDA tensors")

    batch, total_time = waveforms.shape
    if batch == 0 or total_time == 0:
        return waveforms

    if chunk_count_low < 0:
        raise ValueError("chunk_count_low must be non-negative")
    if chunk_count_high < chunk_count_low:
        raise ValueError("chunk_count_high must be >= chunk_count_low")

    min_len, max_len = _scaled_bounds(sample_rate)
    if sample_rate != _BASE_SAMPLE_RATE:
        min_len = max(min_len, int(round(chunk_size_low * (sample_rate / _BASE_SAMPLE_RATE))))
        max_len = max(min_len, int(round(chunk_size_high * (sample_rate / _BASE_SAMPLE_RATE))))
    else:
        min_len = max(min_len, chunk_size_low)
        max_len = max(min_len, chunk_size_high)

    max_len = min(max_len, total_time)
    min_len = min(min_len, max_len)
    if chunk_count_high == 0 or max_len == 0:
        return waveforms

    device = waveforms.device

    for b in range(batch):
        chunk_count = int(
            torch.randint(
                chunk_count_low,
                chunk_count_high + 1,
                (),
                device=device,
            ).item()
        )
        if chunk_count == 0:
            continue
        lengths = torch.randint(
            min_len,
            max_len + 1,
            (chunk_count,),
            device=device,
        )
        lengths = torch.clamp(lengths, max=total_time)
        rand = torch.rand((chunk_count,), device=device)
        start_max = (total_time - lengths).clamp_min(0)
        starts = torch.floor(rand * (start_max + 1).to(rand.dtype)).to(torch.long)
        for i in range(chunk_count):
            s = int(starts[i].item())
            e = int((starts[i] + lengths[i]).item())
            waveforms[b, s:e] = 0.0
    return waveforms


__all__ = ["time_dropout"]
