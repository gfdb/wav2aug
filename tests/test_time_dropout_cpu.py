import pytest
import torch

import wav2aug.cpu._time_dropout as mod


def _stub_randint_factory(scalars, vectors):
    """Create a deterministic stub for torch.randint.

    Args:
        scalars: Sequence of ints to return for scalar randint calls (size == ()).
        vectors: Sequence of sequences to return for vector randint calls (size == (n,)).

    Returns:
        Callable that mimics torch.randint and pops from the provided sequences.
    """
    scalars = list(scalars)
    vectors = [torch.as_tensor(v, dtype=torch.long) for v in vectors]

    def _stub(low, high=None, size=(), **kwargs):
        if size == () or size is None:
            assert scalars, "exhausted scalar randint stubs"
            return torch.tensor(int(scalars.pop(0)), dtype=torch.long)
        else:
            assert vectors, "exhausted vector randint stubs"
            return vectors.pop(0)

    return _stub


def test_empty_waveform_noop():
    """Empty input should be returned unchanged."""
    x = torch.empty(0)
    out = mod.time_drop(x)
    assert out.data_ptr() == x.data_ptr()
    assert out.numel() == 0


def test_mono_controlled_two_masks(monkeypatch):
    """Mono case: with chunk_count=2, lengths=[3,3], starts=[1,7], only those spans are zeroed."""
    T = 20
    x = torch.arange(T, dtype=torch.float32)
    x0 = x.clone()

    stub = _stub_randint_factory(
        scalars=[2, 1, 7],
        vectors=[[3, 3]]
    )
    monkeypatch.setattr(mod.torch, "randint", stub)

    out = mod.time_drop(x)

    assert torch.all(out[1:4] == 0)
    assert torch.all(out[7:10] == 0)
    assert torch.equal(out[:1], x0[:1])
    assert torch.equal(out[4:7], x0[4:7])
    assert torch.equal(out[10:], x0[10:])


def test_multichannel_zeroes_all_channels(monkeypatch):
    """Multichannel case: masking a span zeroes every channel in that time range."""
    C, T = 2, 12
    x = torch.arange(C*T, dtype=torch.float32).view(C, T)
    stub = _stub_randint_factory(
        scalars=[1, 5],
        vectors=[[4]]
    )
    monkeypatch.setattr(mod.torch, "randint", stub)

    out = mod.time_drop(x)
    assert torch.all(out[5:9, :] == 0)
    assert torch.equal(out[:5, :], x[:5, :])
    assert torch.equal(out[9:, :], x[9:, :])


def test_overlaps_allowed(monkeypatch):
    """Overlapping blocks: union of intervals is zero; non-overlapped parts remain."""
    T = 15
    x = torch.ones(T)
    stub = _stub_randint_factory(
        scalars=[2, 2, 4],
        vectors=[[4, 5]]
    )
    monkeypatch.setattr(mod.torch, "randint", stub)

    out = mod.time_drop(x)
    assert torch.all(out[2:9] == 0)
    assert torch.all(out[:2] == 1)
    assert torch.all(out[9:] == 1)


def test_samplerate_scaling(monkeypatch):
    """Sample-rate scaling: at 32 kHz, a 5000-sample block is allowed and zeroed from start."""
    C, T = 1, 6000
    x = torch.ones((C, T))
    stub = _stub_randint_factory(
        scalars=[1, 0],
        vectors=[[5000]]
    )
    monkeypatch.setattr(mod.torch, "randint", stub)

    out = mod.time_drop(x, sample_rate=32_000)
    assert torch.all(out[:5000, 0] == 0)
    assert torch.all(out[5000:, 0] == 1)


def test_zero_length_chunks_have_no_effect(monkeypatch):
    """Zero-length chunks are no-ops; waveform remains identical."""
    T = 50
    x = torch.randn(T)
    x0 = x.clone()
    stub = _stub_randint_factory(
        scalars=[3, 0, 0, 0],
        vectors=[[0, 0, 0]]
    )
    monkeypatch.setattr(mod.torch, "randint", stub)

    out = mod.time_drop(x)
    assert torch.equal(out, x0)
