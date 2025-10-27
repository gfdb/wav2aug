
import os
from typing import Optional

import torch
import torch.nn.functional as F

_EPS = 1e-14
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}


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
def apply_snr_and_mix(
    view: torch.Tensor,
    noise: torch.Tensor,
    snr_low: float,
    snr_high: float,
) -> torch.Tensor:
    """Apply SNR scaling and mix noise into waveform.

    Computes signal and noise RMS, samples random SNR, and mixes noise
    at appropriate level. Modifies both input tensors in-place.

    Args:
        view: Clean waveform in [C, T] format. Modified in-place.
        noise: Noise tensor in [C, T] format. Modified in-place.
        snr_low: Minimum SNR in dB.
        snr_high: Maximum SNR in dB.

    Returns:
        The modified view tensor (same object as input).
    """
    """Apply SNR scaling and mix noise into the waveform.

    Args:
        view (torch.Tensor): The input waveform. Shape [C, T].
        noise (torch.Tensor): The noise to mix. Shape [C, T].
        snr_low (float): Minimum SNR (dB).
        snr_high (float): Maximum SNR (dB).

    Returns:
        torch.Tensor: The waveform with noise mixed in.
    """
    r_x = view.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(_EPS)

    SNR = torch.rand(()) * (snr_high - snr_low) + snr_low
    factor = 1.0 / (torch.pow(torch.tensor(10.0, dtype=view.dtype), SNR / 20.0) + 1.0)
    view.mul_(1.0 - factor)

    r_n = noise.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(_EPS)

    noise.mul_(factor * r_x / r_n)
    view.add_(noise)

    return view

def _sample_noise_like(x: torch.Tensor, sr: int, noise_dir: Optional[str]) -> torch.Tensor:
    """Sample noise matching waveform shape.

    Loads random audio file from noise_dir or generates random noise if no
    directory provided. Resamples and crops/pads to match input dimensions.

    Args:
        x: Reference tensor in [C, T] format for shape matching.
        sr: Target sample rate for resampling noise files.
        noise_dir: Directory containing noise files. If None, generates random noise.

    Returns:
        Noise tensor with same shape as x.
    """
    C, T = x.shape
    if not noise_dir:
        return torch.randn(C, T, dtype=x.dtype)

    files = _list_audio_files(noise_dir)
    if not files:
        return torch.randn(C, T, dtype=x.dtype)

    idx = int(torch.randint(0, len(files), ()))
    from torchcodec.decoders import AudioDecoder
    dec = AudioDecoder(files[idx], sample_rate=int(sr))
    samp = dec.get_all_samples()
    n = samp.data.contiguous().to(dtype=x.dtype)

    if n.size(0) == 1 and C > 1:
        n = n.repeat(C, 1)
    elif n.size(0) != C:
        n = n.mean(dim=0, keepdim=True).repeat(C, 1)

    nT = n.size(1)
    if nT > T:
        off = int(torch.randint(0, nT - T + 1, ()))
        n = n[:, off:off+T]
    elif nT < T:
        n = F.pad(n, (0, T - nT))
    return n

def _list_audio_files(root: str) -> list[str]:
    """List all audio files recursively in directory.
    
    Args:
        root: Root directory path to search.
        
    Returns:
        Sorted list of audio file paths.
    """
    out = []
    for d, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in _AUDIO_EXTS:
                out.append(os.path.join(d, fn))
    return sorted(out)