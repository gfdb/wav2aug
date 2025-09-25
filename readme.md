# Wav2Aug: Toward Universal Time-Domain Speech Augmentation 

A minimalistic PyTorch-based audio augmentation library for speech and audio processing. The goal of this library is to provide a general purpose speech augmentation policy that can be used on any task and perform well without having to tune augmentation hyperparameters. Just install, and start augmenting. Applies two random augmentations per call.

![Diagram](https://raw.githubusercontent.com/gfdb/wav2aug/main/wav2aug.png)

## Features

- **Minimal dependencies**: We only rely on PyTorch and torchcodec.
- **9 core augmentations**: amplitude scaling/clipping, noise addition, frequency dropout, polarity inversion, chunk swapping, speed perturbation, time dropout, and babble noise.
- **In-place operations**: All cpu augmentations are done in place.

## Installation

### pip
```bash
pip install wav2aug
```

### uv
```bash
uv add wav2aug
```

## Quick Start

```python
import torch
from wav2aug import Wav2Aug

# Initialize augmenter
aug = Wav2Aug(sample_rate=16000)

# Process audio (supports [T] mono or [C, T] multi-channel)
waveform = torch.randn(8000)  # 0.5s at 16kHz
augmented = aug(waveform)
```

## Augmentation Types

- **Amplitude Scaling/Clipping**: Random gain and peak limiting
- **Noise Addition**: Environmental noise with SNR control
- **Frequency Dropout**: Spectral masking with random notch filters
- **Polarity Inversion**: Random phase flip
- **Chunk Swapping**: Temporal segment reordering
- **Speed Perturbation**: Time-scale modification
- **Time Dropout**: Random silence insertion
- **Babble Noise**: Multi-speaker background (auto-enabled with sufficient buffer)

## Development Installation

```bash
git clone https://github.com/gfdb/wav2aug
cd wav2aug
uv python pin 3.10 # or greater
uv sync
uv sync --extra test # for test deps
```

## Tests

```bash
uv run pytest tests/
```
