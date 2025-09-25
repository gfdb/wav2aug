import pytest
import torch

import wav2aug.cpu._chunk_swapping as mod


def test_identity_noop(monkeypatch):
    monkeypatch.setattr(torch, "randperm", lambda n: torch.arange(n))
    monkeypatch.setattr(mod, "_sample_unique_sorted_floyd", lambda M, N: torch.arange(N))

    x = torch.randint(0, 100, (2, 200))
    x_clone = x.clone()
    out = mod.chunk_swap(x)

    assert out.data_ptr() == x.data_ptr()
    assert torch.equal(out, x_clone)


def test_correct_mapping_multichannel(monkeypatch):
    # Choose small, explicit starts via stars-and-bars:
    # Let y = [1, 4, 7, 8]; then z = y + arange(4) = [1, 5, 9, 11]
    y = torch.tensor([1, 4, 7, 8], dtype=torch.long)
    z = y + torch.arange(4, dtype=torch.long)

    monkeypatch.setattr(mod, "_sample_unique_sorted_floyd", lambda M, N: z.clone())
    monkeypatch.setattr(torch, "randperm", lambda n: torch.tensor([1, 2, 0, 3], dtype=torch.long))

    C, T = 2, 200
    x = torch.stack([torch.arange(T), 1000 + torch.arange(T)], dim=0)
    x_clone = x.clone()

    k = max(1, int(T * mod._CHUNK_SIZE_FRAC))
    starts = (y + torch.arange(4, dtype=torch.long) * k).tolist()
    perm = [1, 2, 0, 3]

    out = mod.chunk_swap(x)

    assert out.shape == x_clone.shape

    for i in range(mod._NUM_CHUNKS):
        dst = starts[i]
        src = starts[perm[i]]
        assert torch.equal(out[:, dst:dst + k], x_clone[:, src:src + k])

    assert not torch.equal(out, x_clone)


def test_mono_supported(monkeypatch):
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    z = y + torch.arange(4, dtype=torch.long)
    monkeypatch.setattr(mod, "_sample_unique_sorted_floyd", lambda M, N: z.clone())
    monkeypatch.setattr(torch, "randperm", lambda n: torch.tensor([2, 0, 3, 1], dtype=torch.long))

    T = 400
    x = torch.arange(T)
    x_clone = x.clone()

    k = max(1, int(T * mod._CHUNK_SIZE_FRAC))
    starts = (y + torch.arange(4, dtype=torch.long) * k).tolist()
    perm = [2, 0, 3, 1]

    out = mod.chunk_swap(x)

    for i in range(mod._NUM_CHUNKS):
        dst = starts[i]
        src = starts[perm[i]]
        assert torch.equal(out[dst:dst + k], x_clone[src:src + k])


def test_raises_when_too_short():
    """Test that very short waveforms raise ValueError."""
    x = torch.arange(3)
    with pytest.raises(ValueError):
        mod.chunk_swap(x)
