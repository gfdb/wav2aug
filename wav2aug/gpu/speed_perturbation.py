from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

_SPEED_CHANGES: Final[tuple[float, ...]] = (0.9, 1.0, 1.1)


_INTEGER_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@torch.no_grad()
def speed_perturb(
    waveforms: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply batch-wide speed perturbation while preserving batch shape.

    The entire batch shares a speed factor sampled from ``{0.9, 1.0, 1.1}``.
    Resampling is performed with linear interpolation.
    Output waveforms retain the original length ``T`` by random cropping when
    stretched and zero-padding when compressed. If ``lengths`` is provided,
    it must be shaped ``[batch]`` and reside on the same CUDA device. Integer
    and floating-point dtypes are accepted; floats are rounded to the nearest
    sample before processing. The tensor is updated in-place to reflect the
    amount of non-padded audio after the perturbation.
    """
    if waveforms.ndim != 2:
        raise AssertionError("expected waveforms shaped [batch, time]")
    if waveforms.device.type != "cuda":
        raise AssertionError("speed_perturb expects CUDA tensors")

    batch, total_time = waveforms.shape
    if batch == 0 or total_time < 2:
        return waveforms

    device = waveforms.device
    dtype = waveforms.dtype

    options = torch.tensor(_SPEED_CHANGES, device=device, dtype=dtype)
    speed_indices = torch.randint(0, len(_SPEED_CHANGES), (batch,), device=device)
    speeds = options.index_select(0, speed_indices)

    if lengths is None:
        effective_lengths = torch.full((batch,), total_time, device=device, dtype=torch.long)
        lengths_target: torch.Tensor | None = None
        lengths_dtype = torch.long
    else:
        if lengths.ndim != 1 or lengths.numel() != batch:
            raise AssertionError("expected lengths shaped [batch]")
        if lengths.device != device:
            raise AssertionError("lengths tensor must be on same CUDA device as waveforms")
        if torch.is_floating_point(lengths):
            rounded = torch.round(lengths)
            effective_lengths = rounded.to(dtype=torch.long)
        elif lengths.dtype in _INTEGER_DTYPES:
            effective_lengths = lengths.clone() if lengths.dtype == torch.long else lengths.to(dtype=torch.long)
        else:
            raise AssertionError("lengths tensor must use a float or integer dtype")
        lengths_target = lengths
        lengths_dtype = lengths.dtype
        effective_lengths = torch.clamp(effective_lengths, min=0, max=total_time)

    effective_lengths = torch.clamp(effective_lengths, min=0, max=total_time)

    base = waveforms.clone()

    positive_mask = effective_lengths > 0
    new_lengths = torch.zeros_like(effective_lengths)
    if positive_mask.any():
        lengths_float = effective_lengths.to(dtype=dtype)
        scaled = lengths_float[positive_mask] / speeds[positive_mask]
        scaled = torch.clamp(scaled, min=1.0)
        new_lengths[positive_mask] = torch.round(scaled).to(torch.long)

    resampled_map: dict[int, torch.Tensor] = {}
    for b in range(batch):
        in_len = int(effective_lengths[b].item())
        out_len = int(new_lengths[b].item())
        if in_len <= 0:
            continue
        subset = base[b:b+1, :in_len]
        if out_len == in_len:
            resampled = subset[0]
        elif in_len == 1:
            resampled = subset[:, :1].expand(-1, out_len).contiguous()[0]
        else:
            resampled = F.interpolate(
                subset.unsqueeze(1),
                size=out_len,
                mode="linear",
                align_corners=True,
            ).squeeze(1)[0]
        resampled_map[b] = resampled

    updated_lengths = effective_lengths.clone()

    crop_fracs = torch.rand((batch,), device=device)

    for batch_idx in range(batch):
        current_len = int(effective_lengths[batch_idx].item())
        target_len = int(new_lengths[batch_idx].item())

        if current_len <= 0:
            waveforms[batch_idx].zero_()
            updated_lengths[batch_idx] = 0
            continue

        resampled = resampled_map.get(batch_idx)
        if resampled is None:
            waveforms[batch_idx].zero_()
            waveforms[batch_idx, : min(current_len, total_time)] = base[
                batch_idx, : min(current_len, total_time)
            ]
            updated_lengths[batch_idx] = min(current_len, total_time)
            continue

        if target_len >= total_time:
            start_max = max(0, target_len - total_time)
            if start_max > 0:
                start = int(torch.floor(crop_fracs[batch_idx] * (start_max + 1)).item())
                start = min(start, start_max)
            else:
                start = 0

            segment = resampled[start : start + total_time]
            waveforms[batch_idx].zero_()
            waveforms[batch_idx, : segment.size(0)] = segment
            updated_lengths[batch_idx] = total_time
        else:
            waveforms[batch_idx].zero_()
            if target_len > 0:
                waveforms[batch_idx, :target_len] = resampled[:target_len]
            updated_lengths[batch_idx] = target_len

    if lengths_target is not None:
        if lengths_dtype == torch.long:
            lengths_target.copy_(updated_lengths)
        else:
            lengths_target.copy_(updated_lengths.to(dtype=lengths_dtype))

    return waveforms


__all__ = ["speed_perturb"]
