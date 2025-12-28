from __future__ import annotations

import os
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from wav2aug.utils._aug_utils import _sample_noise_like

_EPS = 1e-14
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}


def _list_audio_files(root: str) -> list[str]:
    """List all audio files recursively in directory."""
    out = []
    for d, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in _AUDIO_EXTS:
                out.append(os.path.join(d, fn))
    return sorted(out)


class _NoiseDataset(IterableDataset):
    """Iterable dataset that yields random noise samples indefinitely."""

    def __init__(self, noise_dir: str, sample_rate: int, max_length: int = 16000 * 10):
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.files = _list_audio_files(noise_dir)
        if not self.files:
            raise ValueError(f"No audio files found in {noise_dir}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            idx = torch.randint(0, len(self.files), (1,)).item()
            try:
                from torchcodec.decoders import AudioDecoder
                dec = AudioDecoder(self.files[idx], sample_rate=self.sample_rate)
                samp = dec.get_all_samples()
                audio = samp.data.contiguous().mean(dim=0)  # mono, shape [time]
                
                # Truncate if too long
                if audio.shape[0] > self.max_length:
                    start = torch.randint(0, audio.shape[0] - self.max_length, (1,)).item()
                    audio = audio[start:start + self.max_length]
                
                yield audio
            except Exception:
                # Skip bad files
                continue


class NoiseLoader:
    """Efficient noise loader with background prefetching via DataLoader.
    
    Usage:
        noise_loader = NoiseLoader(noise_dir, sample_rate=16000, num_workers=2)
        
        # In training loop:
        noisy = add_noise(waveforms, noise_loader, snr_low=0, snr_high=10)
    """

    def __init__(
        self,
        noise_dir: str,
        sample_rate: int,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Initialize the noise loader.
        
        Args:
            noise_dir: Directory containing noise audio files.
            sample_rate: Target sample rate for noise.
            num_workers: Number of background workers for loading. 0 = main process.
            prefetch_factor: Number of batches to prefetch per worker.
        """
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.files = _list_audio_files(noise_dir)
        if not self.files:
            raise ValueError(f"No audio files found in {noise_dir}")
        
        if num_workers > 0:
            dataset = _NoiseDataset(noise_dir, sample_rate)
            self._loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=True,
            )
            self._iter: Iterator[torch.Tensor] | None = None
        else:
            self._loader = None
            self._iter = None

    def _load_one(self) -> torch.Tensor:
        """Load a single noise sample directly (no DataLoader)."""
        from torchcodec.decoders import AudioDecoder
        
        idx = torch.randint(0, len(self.files), (1,)).item()
        dec = AudioDecoder(self.files[idx], sample_rate=self.sample_rate)
        samp = dec.get_all_samples()
        audio = samp.data.contiguous().mean(dim=0)  # mono, shape [time]
        return audio

    def get_batch(self, batch_size: int, length: int) -> torch.Tensor:
        """Get a batch of noise samples.
        
        Args:
            batch_size: Number of noise samples needed.
            length: Required length of each sample in frames.
            
        Returns:
            Tensor of shape [batch_size, length].
        """
        noises = []
        
        if self._loader is not None:
            # Use DataLoader with workers
            if self._iter is None:
                self._iter = iter(self._loader)
            
            for _ in range(batch_size):
                noise = next(self._iter).squeeze(0)
                noise = self._pad_or_crop(noise, length)
                noises.append(noise)
        else:
            # Direct loading (num_workers=0)
            for _ in range(batch_size):
                noise = self._load_one()
                noise = self._pad_or_crop(noise, length)
                noises.append(noise)
        
        return torch.stack(noises, dim=0)

    def _pad_or_crop(self, noise: torch.Tensor, length: int) -> torch.Tensor:
        """Pad or crop noise to target length."""
        if noise.shape[0] < length:
            noise = F.pad(noise, (0, length - noise.shape[0]))
        elif noise.shape[0] > length:
            start = torch.randint(0, noise.shape[0] - length + 1, (1,)).item()
            noise = noise[start:start + length]
        return noise


@torch.no_grad()
def _mix_noise(
    waveforms: torch.Tensor,
    noise: torch.Tensor,
    *,
    snr_low: float,
    snr_high: float,
) -> torch.Tensor:
    """Mix noise into the waveforms at a specified SNR.

    Uses SpeechBrain's mixing formula which properly preserves signal
    energy based on the target SNR. The formula computes amplitude factors
    such that the resulting mixture has the desired signal-to-noise ratio.

    Args:
        waveforms (torch.Tensor): The input waveforms. Shape [batch, time].
        noise (torch.Tensor): The noise waveforms to mix in. Shape [batch, time].
        snr_low (float): The minimum SNR (Signal-to-Noise Ratio) in dB.
        snr_high (float): The maximum SNR (Signal-to-Noise Ratio) in dB.

    Raises:
        AssertionError: If waveforms and noise are not 2D shaped [batch, time].
        AssertionError: If waveforms and noise do not have identical shapes.

    Returns:
        torch.Tensor: The waveforms with mixed noise.
    """
    if waveforms.ndim != 2 or noise.ndim != 2:
        raise AssertionError("expected waveforms and noise shaped [batch, time]")
    if waveforms.shape != noise.shape:
        raise AssertionError("waveforms and noise must have identical shapes")

    if waveforms.numel() == 0:
        return waveforms

    device = waveforms.device
    dtype = waveforms.dtype

    # Sample random SNR for each item in batch
    snr = torch.rand((waveforms.size(0), 1), device=device, dtype=dtype)
    snr = snr * (snr_high - snr_low) + snr_low

    # Convert SNR from dB to amplitude ratio: 10^(SNR/20)
    # Then compute noise amplitude factor: 1 / (amplitude_ratio + 1)
    snr_amplitude = torch.pow(
        torch.tensor(10.0, device=device, dtype=dtype), snr / 20.0
    )
    noise_amplitude_factor = 1.0 / (snr_amplitude + 1.0)

    # Compute RMS amplitudes
    signal_rms = waveforms.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(_EPS)
    noise_rms = noise.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(_EPS)

    # Scale the clean signal by (1 - noise_amplitude_factor)
    waveforms.mul_(1.0 - noise_amplitude_factor)

    # Compute target noise amplitude and scale noise accordingly
    noise_scale = (noise_amplitude_factor * signal_rms) / noise_rms
    waveforms.add_(noise * noise_scale)

    return waveforms


@torch.no_grad()
def add_noise(
    waveforms: torch.Tensor,
    sample_rate_or_loader: int | NoiseLoader,
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
        sample_rate_or_loader: Either the sample rate (int) for legacy behavior,
            or a NoiseLoader instance for efficient background loading.
        snr_low (float, optional): The minimum SNR in dB. Defaults to 0.0.
        snr_high (float, optional): The maximum SNR in dB. Defaults to 10.0.
        noise_dir (str | None, optional): Directory containing noise files.
            Only used when sample_rate_or_loader is an int. Defaults to None.
        download (bool, optional): Whether to download noise files if not found.
            Only used when sample_rate_or_loader is an int. Defaults to True.
        pack (str, optional): The name of the noise pack to use.
            Only used when sample_rate_or_loader is an int. Defaults to "pointsource_noises".

    Returns:
        torch.Tensor: The waveforms with point-source noise added.
    
    Example:
        # Fast path with NoiseLoader (recommended):
        loader = NoiseLoader("/path/to/noise", sample_rate=16000, num_workers=4)
        noisy = add_noise(waveforms, loader, snr_low=0, snr_high=10)
        
        # Legacy path (slower, loads from disk each call):
        noisy = add_noise(waveforms, 16000, snr_low=0, snr_high=10, noise_dir="/path/to/noise")
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    if waveforms.numel() == 0:
        return waveforms

    batch, total_time = waveforms.shape
    device = waveforms.device
    dtype = waveforms.dtype

    if isinstance(sample_rate_or_loader, NoiseLoader):
        # Fast path: use the NoiseLoader
        noise = sample_rate_or_loader.get_batch(batch, total_time)
        noise = noise.to(device=device, dtype=dtype)
    else:
        # Legacy path: load noise synchronously
        sample_rate = sample_rate_or_loader
        
        if noise_dir is None and download:
            from wav2aug.data.fetch import ensure_pack
            noise_dir = ensure_pack(pack)

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

    Returns:
        torch.Tensor: The waveforms with babble noise added.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    if waveforms.numel() == 0:
        return waveforms

    batch = waveforms.size(0)
    if batch == 1:
        noise = waveforms.clone()
    else:
        total = torch.sum(waveforms, dim=0, keepdim=True)
        noise = (total - waveforms) / (batch - 1)
    return _mix_noise(waveforms, noise, snr_low=snr_low, snr_high=snr_high)


__all__ = ["add_noise", "add_babble_noise", "NoiseLoader"]
