import pytest
import torch

from wav2aug.gpu import (
    Wav2Aug,
    add_babble_noise,
    add_noise,
    chunk_swap,
    freq_drop,
    invert_polarity,
    rand_amp_clip,
    rand_amp_scale,
    speed_perturb,
    time_dropout,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _waveforms(
    batch: int = 3, time: int = 256, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, time, device=DEVICE, dtype=dtype)


def test_rand_amp_clip_inplace_preserves_shape():
    waveforms = _waveforms()
    ptr = waveforms.data_ptr()
    out = rand_amp_clip(waveforms)
    assert out.data_ptr() == ptr
    assert out.shape == (waveforms.size(0), waveforms.size(1))
    assert torch.isfinite(out).all()


def test_rand_amp_scale_inplace_preserves_shape():
    waveforms = _waveforms()
    ptr = waveforms.data_ptr()
    out = rand_amp_scale(waveforms)
    assert out.data_ptr() == ptr
    assert out.shape == (waveforms.size(0), waveforms.size(1))
    assert torch.isfinite(out).all()


def test_chunk_swap_outputs_permutation():
    batch, time = 2, 400
    base = torch.arange(batch * time, device=DEVICE, dtype=torch.float32).view(
        batch, time
    )
    reference = base.clone()
    out = chunk_swap(base)
    assert out.data_ptr() == base.data_ptr()
    assert torch.allclose(
        torch.sort(out, dim=1).values, torch.sort(reference, dim=1).values
    )


def test_freq_drop_no_nan_and_inplace():
    waveforms = _waveforms()
    ptr = waveforms.data_ptr()
    out = freq_drop(waveforms)
    assert out.data_ptr() == ptr
    assert torch.isnan(out).logical_not().all()


def test_add_noise_with_stub(monkeypatch):
    def _stub_noise_like(ref, sample_rate, noise_dir):
        return torch.zeros_like(ref)

    monkeypatch.setattr(
        "wav2aug.gpu.noise_addition._sample_noise_like", _stub_noise_like
    )
    waveforms = torch.ones(2, 128, device=DEVICE, dtype=torch.float32)
    ptr = waveforms.data_ptr()
    out = add_noise(
        waveforms,
        sample_rate=16_000,
        snr_low=0.0,
        snr_high=0.0,
        download=False,
        noise_dir="ignored",
    )
    assert out.data_ptr() == ptr
    assert torch.isfinite(out).all()


def test_add_babble_noise_identity_for_singleton_batch():
    waveforms = torch.full((1, 64), 2.0, device=DEVICE, dtype=torch.float32)
    ptr = waveforms.data_ptr()
    out = add_babble_noise(waveforms, snr_low=0.0, snr_high=0.0)
    assert out.data_ptr() == ptr
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_invert_polarity_flips_when_prob_one():
    waveforms = _waveforms(batch=2, time=32)
    original = waveforms.clone()
    out = invert_polarity(waveforms, prob=1.0)
    assert torch.allclose(out, -original)


def test_speed_perturb_adjusts_length():
    waveforms = torch.linspace(
        0, 1, steps=200, device=DEVICE, dtype=torch.float32
    ).repeat(2, 1)
    out = speed_perturb(waveforms, 16000, speed_changes=(0.5,))
    expected_len = int(round(200 * 1 / 0.5))
    assert out.shape == (2, expected_len)
    assert out.device.type == "cuda"


def test_time_dropout_zeroes_segments():
    waveforms = torch.ones(2, 64, device=DEVICE, dtype=torch.float32)
    lengths = torch.ones(2, device=DEVICE, dtype=torch.float32)
    ptr = waveforms.data_ptr()
    out = time_dropout(
        waveforms,
        lengths,
        chunk_count_low=1,
        chunk_count_high=1,
        chunk_size_low=2,
        chunk_size_high=2,
    )
    assert out.data_ptr() == ptr
    zeros_per_row = (out == 0).sum(dim=1)
    assert torch.all(zeros_per_row >= 2)


def test_wav2aug_runs_with_stubbed_noise(monkeypatch):
    def _noop_add_noise(waveforms, sample_rate, **kwargs):
        return waveforms

    monkeypatch.setattr("wav2aug.gpu.wav2aug.add_noise", _noop_add_noise)

    aug = Wav2Aug(sample_rate=16_000)
    waveforms = _waveforms(batch=3, time=256)
    lengths = torch.ones(3, device=DEVICE, dtype=torch.float32)
    out_wave, out_lengths = aug(waveforms, lengths=lengths)
    assert out_wave.shape[0] == waveforms.shape[0]
    assert out_wave.device.type == "cuda"
    assert out_lengths.data_ptr() == lengths.data_ptr()
