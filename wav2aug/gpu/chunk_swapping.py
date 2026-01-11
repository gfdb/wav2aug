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

    The implementation selects four non-overlapping segments of length
    ``ceil(0.01 * time)`` and permutes them independently per waveform.

    This version uses fully vectorized gather/scatter operations.

    Args:
        waveforms: Tensor of shape [batch, time].

    Returns:
        A new tensor with chunks swapped.

    Raises:
        ValueError: If the waveform is shorter than the total chunk span.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    batch, total_time = waveforms.shape
    if batch == 0 or total_time == 0:
        return waveforms

    chunk_size = max(1, int(total_time * _CHUNK_SIZE_FRAC))
    if _NUM_CHUNKS * chunk_size > total_time:
        raise ValueError("Not enough time steps to apply chunk swap.")

    device = waveforms.device

    # Precompute ranges
    arange_chunk = torch.arange(chunk_size, device=device)
    arange_n = torch.arange(_NUM_CHUNKS, device=device)

    slack = total_time - _NUM_CHUNKS * chunk_size

    # Sample chunk positions: [batch, _NUM_CHUNKS]
    if slack == 0:
        starts = (arange_n * chunk_size).unsqueeze(0).expand(batch, -1)
    else:
        scores = torch.rand((batch, slack + _NUM_CHUNKS), device=device)
        topk = torch.topk(scores, _NUM_CHUNKS, dim=1, largest=False).indices
        offsets = torch.sort(topk, dim=1).values - arange_n
        starts = offsets + arange_n * chunk_size

    # Sample permutations: [batch, _NUM_CHUNKS]
    perms = torch.argsort(torch.rand((batch, _NUM_CHUNKS), device=device), dim=1)

    # Source starts after permutation
    src_starts = starts.gather(1, perms)

    # Build gather indices: start with identity [batch, total_time]
    indices = (
        torch.arange(total_time, device=device)
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )

    # Compute all destination and source positions: [batch, _NUM_CHUNKS * chunk_size]
    dest_indices = (starts.unsqueeze(2) + arange_chunk).reshape(batch, -1)
    src_indices = (src_starts.unsqueeze(2) + arange_chunk).reshape(batch, -1)

    # Single scatter to update index mapping
    indices.scatter_(1, dest_indices, src_indices)

    # Apply gather to get swapped waveforms
    return waveforms.gather(1, indices)


__all__ = ["chunk_swap"]
