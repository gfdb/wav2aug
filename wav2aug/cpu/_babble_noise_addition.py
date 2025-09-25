import torch
import torch.nn.functional as F

from wav2aug.cpu._aug_utils import apply_snr_and_mix


@torch.no_grad()
def add_babble_noise(
    waveform: torch.Tensor,
    babble_waveform: torch.Tensor,
    snr_low: float = 0.0,
    snr_high: float = 20.0,
) -> torch.Tensor:
    """Add babble noise using given babble-like waveform.

    Automatically matches channels and adjusts length to `waveform`.

    Args:
        waveform: Input audio in [T] or [C, T] format. Must be on CPU.
        babble_waveform: Babble-like noise waveform in [T] or [C, T] format. Must be on CPU.
        snr_low: Minimum SNR in dB. Default: 0.0.
        snr_high: Maximum SNR in dB. Default: 20.0.

    Returns:
        The waveform tensor with babble noise mixed in.
        
    Raises:
        AssertionError: If tensors are not on CPU or have wrong dimensions.
    """
    assert waveform.device.type == "cpu", "Input waveform must be on CPU."
    assert babble_waveform.device.type == "cpu", "Batch sum must be on CPU."
    assert waveform.ndim in (1, 2), "Expected waveform shape [T] or [C, T]."
    assert babble_waveform.ndim in (1, 2), "Expected babble_waveform shape [T] or [C, T]."

    if waveform.numel() == 0:
        return waveform

    if waveform.ndim == 1:
        view = waveform.view(1, -1)
        batch_view = babble_waveform.view(1, -1)
        squeeze = True
    else:
        view = waveform
        batch_view = babble_waveform
        squeeze = False

    C, T = view.shape

    if batch_view.shape[0] != C:
        batch_view = batch_view.mean(dim=0, keepdim=True).repeat(C, 1)

    if batch_view.shape[1] > T:
        batch_view = batch_view[:, :T]
    elif batch_view.shape[1] < T:
        batch_view = F.pad(batch_view, (0, T - batch_view.shape[1]))

    apply_snr_and_mix(view, batch_view, snr_low, snr_high)

    return view.view(-1) if squeeze else view