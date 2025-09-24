from __future__ import annotations

from typing import Final

import torch

_NUM_CHUNKS: Final[int] = 4
_CHUNK_SIZE_FRAC: Final[float] = 0.01

@torch.no_grad()
def _sample_unique_sorted_floyd(M: int, N: int) -> torch.Tensor:
    """Sample N unique integers from [0, M) in sorted order using Floyd's algorithm.
    
    Args:
        M: Upper bound (exclusive) for sampling range.
        N: Number of unique integers to sample.
        
    Returns:
        Sorted tensor of N unique integers.
    """
    S: set[int] = set()
    for j in range(M - N, M):
        r = int(torch.randint(0, j + 1, ()).item())
        if r in S:
            S.add(j)
        else:
            S.add(r)
    return torch.tensor(sorted(S), dtype=torch.long)

@torch.no_grad()
def chunk_swap(
    waveform: torch.Tensor,
) -> torch.Tensor:
    """Swap 4 non-overlapping chunks at random positions.

    Selects 4 chunks of fixed size (1% of waveform length) and swaps their
    positions randomly. Uses efficient in-place cycle decomposition.

    Args:
        waveform: Audio tensor in [T] or [C, T] format. Must be on CPU.

    Returns:
        The input tensor modified in-place.
        
    Raises:
        AssertionError: If waveform is not on CPU or has wrong dimensions.
        ValueError: If waveform is too short for 4 chunks.
    """
    assert waveform.device.type == "cpu", "CPU only"
    assert waveform.ndim in (1, 2), "expected [T] or [C, T]"

    perm = torch.randperm(4)
    if torch.all(perm == torch.arange(4)):
        return waveform

    mono = (waveform.ndim == 1)
    if mono:
        T = int(waveform.numel())
        C = 1
        view = waveform.view(1, T)
    else:
        C, T = int(waveform.shape[0]), int(waveform.shape[1])
        view = waveform

    chunk_size = max(1, int(T * _CHUNK_SIZE_FRAC))
    if _NUM_CHUNKS * chunk_size > T:
        raise ValueError("Not enough time steps to apply chunk swap.")

    time_minus_buffer = T - _NUM_CHUNKS * chunk_size
    # uniform non-overlapping starts via stars-and-bars (no O(T) randperm)
    unique_samples = _sample_unique_sorted_floyd(time_minus_buffer + _NUM_CHUNKS, _NUM_CHUNKS)

    ARANGE_NUM_CHUNKS = torch.arange(_NUM_CHUNKS, dtype=torch.long)
    non_decreasing_seq = unique_samples - ARANGE_NUM_CHUNKS

    starts = non_decreasing_seq + ARANGE_NUM_CHUNKS * chunk_size

    need_shape = (C, chunk_size)
    scratch = torch.empty(need_shape, dtype=view.dtype)

    # in place cycle decomposition with O(chunk_size*C) scratch
    visited = torch.zeros(_NUM_CHUNKS, dtype=torch.bool)
    for i in range(_NUM_CHUNKS):
        if visited[i] or int(perm[i].item()) == i:
            visited[i] = True
            continue
        j = i
        s = int(starts[j].item())
        scratch.copy_(view[:, s:s + chunk_size])
        while True:
            nj = int(perm[j].item())
            s_next = int(starts[nj].item())
            if nj == i:
                view[:, s:s + chunk_size] = scratch
                visited[j] = True
                break
            view[:, s:s + chunk_size] = view[:, s_next:s_next + chunk_size]
            visited[j] = True
            j, s = nj, s_next

    return waveform