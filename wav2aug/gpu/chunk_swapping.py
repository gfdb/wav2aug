from __future__ import annotations

from typing import Final

import torch

_NUM_CHUNKS: Final[int] = 4
_CHUNK_SIZE_FRAC: Final[float] = 0.01


@torch.no_grad()
def chunk_swap(
    waveforms: torch.Tensor,
) -> torch.Tensor:
    """Swap non-overlapping chunks for each waveform in the batch.

    The GPU implementation mirrors the CPU behaviour by selecting four
    non-overlapping segments of length ``ceil(0.01 * time)`` and permuting
    them independently per waveform.

    Args:
        waveforms: Tensor of shape [batch, time] on CUDA device.

    Returns:
        The input ``waveforms`` tensor, modified in-place.

    Raises:
        ValueError: If the waveform is shorter than the total chunk span.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("chunk_swap expects CUDA tensors")

    batch, total_time = waveforms.shape
    if batch == 0 or total_time == 0:
        return waveforms

    chunk_size = max(1, int(total_time * _CHUNK_SIZE_FRAC))
    if _NUM_CHUNKS * chunk_size > total_time:
        raise ValueError("Not enough time steps to apply chunk swap.")

    device = waveforms.device

    src = waveforms.clone()
    arange_chunk = torch.arange(chunk_size, device=device)

    for b in range(batch):
        slack = total_time - _NUM_CHUNKS * chunk_size
        if slack == 0:
            offsets = torch.zeros(_NUM_CHUNKS, device=device, dtype=torch.long)
        else:
            scores = torch.rand((slack + _NUM_CHUNKS,), device=device)
            topk = torch.topk(scores, _NUM_CHUNKS, largest=False).indices
            offsets = torch.sort(topk).values
            offsets = offsets - torch.arange(_NUM_CHUNKS, device=device)
        starts = offsets + torch.arange(_NUM_CHUNKS, device=device) * chunk_size
        perm_scores = torch.rand((_NUM_CHUNKS,), device=device)
        perm = torch.argsort(perm_scores)
        if torch.equal(perm, torch.arange(_NUM_CHUNKS, device=device)):
            continue
        for dest_chunk in range(_NUM_CHUNKS):
            dest_start = starts[dest_chunk]
            src_chunk = perm[dest_chunk]
            src_start = starts[src_chunk]
            dest_idx = dest_start + arange_chunk
            src_idx = src_start + arange_chunk
            waveforms[b, dest_idx] = src[b, src_idx]

    return waveforms


__all__ = ["chunk_swap"]
