from __future__ import annotations

import torch

from wav2aug.utils._aug_utils import _sample_noise_like

_EPS = 1e-14


@torch.no_grad()
def _mix_noise(
    waveforms: torch.Tensor,
    noise: torch.Tensor,
    *,
    snr_low: float,
    snr_high: float,
) -> torch.Tensor:
    """Mix noise into the waveforms at a specified SNR.

    Args:
        waveforms (torch.Tensor): The input waveforms. Shape [batch, time].
        noise (torch.Tensor): The noise waveforms to mix in. Shape [batch, time].
        snr_low (float): The minimum SNR (Signal-to-Noise Ratio) in dB.
        snr_high (float): The maximum SNR (Signal-to-Noise Ratio) in dB.

    Raises:
        AssertionError: If waveforms and noise are not 2D shaped [batch, time].
        AssertionError: If waveforms and noise do not have identical shapes.
        AssertionError: If waveforms and noise are not on the CUDA device.

    Returns:
        torch.Tensor: The waveforms with mixed noise.
    """
    if waveforms.ndim != 2 or noise.ndim != 2:
        raise AssertionError("expected waveforms and noise shaped [batch, time]")
    if waveforms.shape != noise.shape:
        raise AssertionError("waveforms and noise must have identical shapes")
    if waveforms.device.type != "cuda" or noise.device.type != "cuda":
        raise AssertionError("noise mixing expects CUDA tensors")

    if waveforms.numel() == 0:
        return waveforms

    device = waveforms.device
    dtype = waveforms.dtype

    snr = torch.rand((waveforms.size(0), 1), device=device, dtype=dtype)
    snr = snr * (snr_high - snr_low) + snr_low
    pow10 = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), snr / 20.0)
    factor = 1.0 / (pow10 + 1.0)

    signal_rms = waveforms.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(_EPS)
    noise_rms = noise.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(_EPS)

    waveforms.mul_(1.0 - factor)
    waveforms.add_(noise * (factor * signal_rms / noise_rms))
    return waveforms


@torch.no_grad()
def add_noise(
    waveforms: torch.Tensor,
    sample_rate: int,
    *,
    snr_low: float = 0.0,
    snr_high: float = 10.0,
    noise_dir: str | None = None,
    download: bool = True,
    pack: str = "pointsource_noises",
) -> torch.Tensor:
    """Add point-source noise to each waveform in the batch.

    Args:
        waveforms (torch.Tensor): The input waveforms. Shape [batch, time].
        sample_rate (int): The sample rate of the audio.
        snr_low (float, optional): The minimum SNR (Signal-to-Noise Ratio) in dB. Defaults to 0.0.
        snr_high (float, optional): The maximum SNR (Signal-to-Noise Ratio) in dB. Defaults to 10.0.
        noise_dir (str | None, optional): The directory containing noise files. Defaults to None.
        download (bool, optional): Whether to download noise files if not found. Defaults to True.
        pack (str, optional): The name of the noise pack to use. Defaults to "pointsource_noises".

    Raises:
        AssertionError: If waveforms are not 2D shaped [batch, time].
        AssertionError: If waveforms are not on the CUDA device.

    Returns:
        torch.Tensor: The waveforms with point-source noise added.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("add_noise expects CUDA tensors")

    if waveforms.numel() == 0:
        return waveforms

    if noise_dir is None and download:
        from wav2aug.data.fetch import ensure_pack

        noise_dir = ensure_pack(pack)

    batch, total_time = waveforms.shape
    device = waveforms.device
    dtype = waveforms.dtype

    # sample independent noise per waveform
    noises = []
    for _ in range(batch):
        ref = torch.empty(1, total_time, dtype=dtype)
        sample = _sample_noise_like(ref, sample_rate, noise_dir)
        noise_sample = sample.to(device=device, dtype=dtype).view(-1)
        noises.append(noise_sample)
    noise = torch.stack(noises, dim=0)

    return _mix_noise(
        waveforms,
        noise,
        snr_low=snr_low,
        snr_high=snr_high,
    )


@torch.no_grad()
def add_babble_noise(
    waveforms: torch.Tensor,
    *,
    snr_low: float = 0.0,
    snr_high: float = 20.0,
) -> torch.Tensor:
    """Add babble noise derived from the batch sum.

    Args:
        waveforms (torch.Tensor): The input waveforms. Shape [batch, time].
        snr_low (float, optional): The minimum SNR (Signal-to-Noise Ratio) in dB. Defaults to 0.0.
        snr_high (float, optional): The maximum SNR (Signal-to-Noise Ratio) in dB. Defaults to 20.0.

    Raises:
        AssertionError: If waveforms are not 2D shaped [batch, time].
        AssertionError: If waveforms are not on the CUDA device.

    Returns:
        torch.Tensor: The waveforms with babble noise added.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("add_babble_noise expects CUDA tensors")
    if waveforms.numel() == 0:
        return waveforms

    batch = waveforms.size(0)
    if batch == 1:
        noise = waveforms.clone()
    else:
        total = torch.sum(waveforms, dim=0, keepdim=True)
        noise = (total - waveforms) / (batch - 1)
    return _mix_noise(waveforms, noise, snr_low=snr_low, snr_high=snr_high)


__all__ = ["add_noise", "add_babble_noise"]
