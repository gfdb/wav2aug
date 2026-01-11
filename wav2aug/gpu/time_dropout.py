from __future__ import annotations

import torch


def _scaled_bounds(
    sample_rate: int,
    *,
    chunk_size_low: int,
    chunk_size_high: int,
    base_sample_rate: int,
) -> tuple[int, int]:
    """Return chunk length bounds scaled to the provided sample rate."""

    if sample_rate != base_sample_rate:
        scale = float(sample_rate) / float(base_sample_rate)
        min_len = max(1, int(round(chunk_size_low * scale)))
        max_len = max(min_len, int(round(chunk_size_high * scale)))
    else:
        min_len = chunk_size_low
        max_len = chunk_size_high
    return min_len, max_len


@torch.no_grad()
def time_dropout(
    waveforms: torch.Tensor,
    sample_rate: int = 16_000,
    *,
    lengths: torch.Tensor | None = None,
    chunk_count_low: int = 1,
    chunk_count_high: int = 8,
    chunk_size_low: int = 0,
    chunk_size_high: int = 4000,
    base_sample_rate: int = 16_000,
) -> torch.Tensor:
    """Apply time dropout using vectorized mask operations.

    This implementation uses vectorized mask operations while maintaining
    per-sample chunk counts to match speechbrain's DropChunk semantics.

    Each sample draws its own number of chunks (uniform in [chunk_count_low,
    chunk_count_high]), but we use the maximum chunk count across the batch
    to enable vectorization. Samples with fewer chunks simply have some
    chunks with zero length.

    Args:
        waveforms: Input waveforms. Shape [batch, time].
        sample_rate: The sample rate of the audio. Defaults to 16_000.
        lengths: Relative valid lengths of each waveform (0-1). Shape [batch].
            If None, assumes all samples have full length.
        chunk_count_low: Minimum number of chunks to drop. Defaults to 1.
        chunk_count_high: Maximum number of chunks to drop. Defaults to 8.
        chunk_size_low: Minimum size of each chunk in samples. Defaults to 0.
        chunk_size_high: Maximum size of each chunk in samples. Defaults to 4000.
        base_sample_rate: Reference sample rate for scaling chunk lengths.

    Returns:
        Waveforms with time dropout applied (in-place modification).

    Raises:
        AssertionError: If waveforms are not 2D.
        ValueError: If chunk parameters are invalid.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms with shape [batch, time]")

    batch, total_time = waveforms.shape
    if batch == 0 or total_time == 0:
        return waveforms

    if chunk_count_low < 0:
        raise ValueError("chunk_count_low must be non-negative")
    if chunk_count_high < chunk_count_low:
        raise ValueError("chunk_count_high must be >= chunk_count_low")
    if chunk_size_low < 0:
        raise ValueError("chunk_size_low must be non-negative")
    if chunk_size_high < chunk_size_low:
        raise ValueError("chunk_size_high must be >= chunk_size_low")

    device = waveforms.device

    if lengths is None:
        valid_lengths = torch.full(
            (batch,), total_time, device=device, dtype=torch.long
        )
    else:
        valid_lengths = (lengths * total_time).to(torch.long).clamp_(0, total_time)

    min_len, max_len = _scaled_bounds(
        sample_rate,
        chunk_size_low=chunk_size_low,
        chunk_size_high=chunk_size_high,
        base_sample_rate=base_sample_rate,
    )

    max_len = min(max_len, total_time)
    min_len = min(min_len, max_len)
    if chunk_count_high == 0 or max_len == 0:
        return waveforms

    # Sample per-sample chunk counts (like speechbrain's DropChunk)
    # Shape: [batch]
    drop_times = torch.randint(
        chunk_count_low, chunk_count_high + 1, (batch,), device=device
    )

    # Use max chunk count for vectorization; samples with fewer chunks
    # will have some chunks zeroed out via masking
    max_chunks = drop_times.max().item()
    if max_chunks == 0:
        return waveforms

    # Sample chunk lengths: [batch, max_chunks]
    chunk_lengths = torch.randint(
        min_len, max_len + 1, (batch, max_chunks), device=device
    )

    # Mask out chunks beyond each sample's drop_times
    # chunk_idx: [1, max_chunks], drop_times: [batch, 1]
    chunk_idx = torch.arange(max_chunks, device=device).unsqueeze(0)
    active_mask = chunk_idx < drop_times.unsqueeze(1)  # [batch, max_chunks]

    # Zero out inactive chunks
    chunk_lengths = chunk_lengths * active_mask

    # Clamp to valid lengths per sample
    chunk_lengths = torch.minimum(chunk_lengths, valid_lengths.unsqueeze(1))

    # Compute valid start range per sample/chunk: start_max = valid_len - chunk_len
    start_max = (valid_lengths.unsqueeze(1) - chunk_lengths).clamp_min(0)

    # Sample start positions: uniform in [0, start_max]
    rand = torch.rand((batch, max_chunks), device=device)
    starts = (rand * (start_max + 1).float()).long()

    # Build a time index: [1, total_time]
    time_idx = torch.arange(total_time, device=device).unsqueeze(0)  # [1, T]

    # For each chunk, create a mask where time_idx is in [start, start+length)
    # starts: [B, max_chunks] -> [B, max_chunks, 1]
    # chunk_lengths: [B, max_chunks] -> [B, max_chunks, 1]
    starts_exp = starts.unsqueeze(2)  # [B, max_chunks, 1]
    ends_exp = (starts + chunk_lengths).unsqueeze(2)  # [B, max_chunks, 1]
    time_idx_exp = time_idx.unsqueeze(1)  # [1, 1, T]

    # Mask: True where we should zero out
    # [B, max_chunks, T] -> any across chunks -> [B, T]
    chunk_mask = (time_idx_exp >= starts_exp) & (time_idx_exp < ends_exp)
    drop_mask = chunk_mask.any(dim=1)  # [B, T]

    # Zero out masked positions
    waveforms.masked_fill_(drop_mask, 0.0)

    return waveforms


__all__ = ["time_dropout"]
