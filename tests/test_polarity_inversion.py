import pytest
import torch

from wav2aug.cpu import invert_polarity


def test_returns_original_when_prob_triggers(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor(0.20))
    x = torch.randn(8, 2)
    out = invert_polarity(x)
    assert out is x
    assert torch.equal(out, x)

def test_returns_negated_otherwise(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor(0.80))
    x = torch.tensor([[1.0, -2.0], [3.0, 0.0]])
    out = invert_polarity(x)
    assert out is not x
    assert torch.equal(out, -x)
    assert out.dtype == x.dtype
    assert out.device == x.device

def test_shape_is_preserved(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor(0.80))
    x = torch.randn(5, 3, 2)
    out = invert_polarity(x)
    assert out.shape == x.shape

def test_boundary_at_point_four(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor(0.40))
    x = torch.tensor([1.0, -1.0, 2.0])
    out = invert_polarity(x)
    assert torch.equal(out, -x)
