from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

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


class NoiseLoader:
    """Noise loader that preloads all noise files into CPU RAM.

    Loads all noise files into CPU RAM at initialization for
    zero-I/O sampling during training.

    Usage:
        # Default:
        noise_loader = NoiseLoader(noise_dir, sample_rate=16000)

        # Custom storage dtype (e.g., for lower memory):
        noise_loader = NoiseLoader(noise_dir, sample_rate=16000, storage_dtype=torch.float16)

        # In training loop:
        noisy = add_noise(waveforms, noise_loader, snr_low=0, snr_high=10)
    """

    def __init__(
        self,
        noise_dir: str,
        sample_rate: int,
        storage_dtype: torch.dtype = torch.float32,
    ):
        """Initialize the noise loader.

        Args:
            noise_dir: Directory containing noise audio files.
            sample_rate: Target sample rate for noise.
            storage_dtype: Data type for storing preloaded audio in memory.
                Defaults to float32. Use float16 for lower memory usage.
        """
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.storage_dtype = storage_dtype
        self.files = _list_audio_files(noise_dir)
        if not self.files:
            raise ValueError(f"No audio files found in {noise_dir}")

        # Preloaded noise bank (1D tensor of all concatenated noise)
        self._noise_bank: torch.Tensor = self._preload_all()

    def _preload_all(self) -> torch.Tensor:
        """Load all noise files into memory."""
        from torchcodec.decoders import AudioDecoder

        chunks: list[torch.Tensor] = []

        for f in tqdm(self.files, desc="Loading noise pack", unit="file"):
            try:
                dec = AudioDecoder(f, sample_rate=self.sample_rate)
                samp = dec.get_all_samples()
                audio = samp.data.contiguous().mean(dim=0)  # mono, shape [time]
                # Convert each chunk immediately to save peak RAM
                chunks.append(audio.to(self.storage_dtype))
            except Exception:
                # Skip bad files
                continue

        if not chunks:
            raise ValueError(
                f"No valid audio files could be loaded from {self.noise_dir}"
            )

        return torch.cat(chunks, dim=0)

    def get_batch(self, batch_size: int, length: int) -> torch.Tensor:
        """Get a batch of noise samples.

        Args:
            batch_size: Number of noise samples needed.
            length: Required length of each sample in frames.

        Returns:
            Tensor of shape [batch_size, length] on CPU.
        """
        bank_len = self._noise_bank.shape[0]

        if bank_len <= length:
            # Noise bank shorter than requested - pad it
            noise = self._noise_bank.unsqueeze(0).expand(batch_size, -1)
            noise = F.pad(noise, (0, length - bank_len))
            return noise

        # Generate random start indices for each sample
        max_start = bank_len - length
        starts = torch.randint(0, max_start + 1, (batch_size,))

        # Vectorized slicing: create index tensor [batch_size, length]
        # where each row is [start, start+1, ..., start+length-1]
        offsets = torch.arange(length)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [batch_size, length]
        return self._noise_bank[indices]


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
    loader: NoiseLoader,
    *,
    snr_low: float = 0.0,
    snr_high: float = 10.0,
) -> torch.Tensor:
    """Add point-source noise to each waveform in the batch.

    Args:
        waveforms (torch.Tensor): The input waveforms. Shape [batch, time].
        loader: A NoiseLoader instance for efficient noise sampling.
        snr_low (float, optional): The minimum SNR in dB. Defaults to 0.0.
        snr_high (float, optional): The maximum SNR in dB. Defaults to 10.0.

    Returns:
        torch.Tensor: The waveforms with point-source noise added.

    Example:
        loader = NoiseLoader("/path/to/noise", sample_rate=16000)
        noisy = add_noise(waveforms, loader, snr_low=0, snr_high=10)
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")

    if waveforms.numel() == 0:
        return waveforms

    batch, total_time = waveforms.shape
    device = waveforms.device
    dtype = waveforms.dtype

    noise = loader.get_batch(batch, total_time)
    noise = noise.to(device=device, dtype=dtype)

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

    # Use batch sum as noise for all samples (matches SpeechBrain)
    noise = torch.sum(waveforms, dim=0, keepdim=True)
    noise = noise.expand_as(waveforms)

    return _mix_noise(waveforms, noise, snr_low=snr_low, snr_high=snr_high)


__all__ = ["add_noise", "add_babble_noise", "NoiseLoader"]
