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

    This version uses vectorized gather operations to avoid per-sample
    Python loops while producing identical output to the original.

    Args:
        waveforms: Tensor of shape [batch, time].

    Returns:
        The input ``waveforms`` tensor, modified in-place.

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

    # Clone source for reading (since we modify in-place)
    src = waveforms.clone()

    # Precompute chunk index offsets
    arange_chunk = torch.arange(chunk_size, device=device)
    arange_n = torch.arange(_NUM_CHUNKS, device=device)

    # Compute slack for position sampling
    slack = total_time - _NUM_CHUNKS * chunk_size

    # Sample positions for all samples at once: [batch, slack + _NUM_CHUNKS]
    if slack == 0:
        # Chunks tile perfectly - same offsets for all samples
        offsets = torch.zeros(batch, _NUM_CHUNKS, device=device, dtype=torch.long)
    else:
        scores = torch.rand((batch, slack + _NUM_CHUNKS), device=device)
        # Get top-k indices per sample
        topk = torch.topk(scores, _NUM_CHUNKS, dim=1, largest=False).indices
        offsets = torch.sort(topk, dim=1).values  # [batch, _NUM_CHUNKS]
        offsets = offsets - arange_n.unsqueeze(0)  # [batch, _NUM_CHUNKS]

    # Compute chunk start positions: [batch, _NUM_CHUNKS]
    starts = offsets + arange_n.unsqueeze(0) * chunk_size

    # Sample permutations per sample: [batch, _NUM_CHUNKS]
    perm_scores = torch.rand((batch, _NUM_CHUNKS), device=device)
    perms = torch.argsort(perm_scores, dim=1)

    # Check which samples have identity permutation
    identity = arange_n.unsqueeze(0).expand(batch, _NUM_CHUNKS)
    is_identity = (perms == identity).all(dim=1)  # [batch]

    # For samples with identity permutation, nothing changes
    # For others, we need to apply the swap

    # Build gather indices for each sample: [batch, total_time]
    # Start with identity mapping
    indices = torch.arange(total_time, device=device).unsqueeze(0).expand(batch, -1).clone()

    # For each chunk position, update indices to point to source chunk
    # dest_chunk gets data from src_chunk = perms[:, dest_chunk]
    for dest_chunk in range(_NUM_CHUNKS):
        # Source chunk index per sample: [batch]
        src_chunk = perms[:, dest_chunk]

        # Destination start per sample: [batch]
        dest_start = starts[:, dest_chunk]

        # Source start per sample: [batch] - gather from starts using src_chunk
        src_start = starts.gather(1, src_chunk.unsqueeze(1)).squeeze(1)

        # For each position in the chunk, set indices[b, dest_start + i] = src_start + i
        # We need to handle this per sample
        for i in range(chunk_size):
            # Use scatter to update indices
            dest_pos = dest_start + i  # [batch]
            src_pos = src_start + i  # [batch]
            indices.scatter_(1, dest_pos.unsqueeze(1), src_pos.unsqueeze(1))

    # Apply gather to get swapped waveforms
    result = src.gather(1, indices)

    # For identity permutation samples, restore original
    # Actually, for identity perms, indices already point to same positions
    # But we can skip them for efficiency by masking
    # However, the gather already handles this correctly

    # Copy result back in-place
    waveforms.copy_(result)

    return waveforms


__all__ = ["chunk_swap"]
