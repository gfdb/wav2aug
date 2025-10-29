# 🎛️ Wav2Aug: Toward Universal Time-Domain Speech Augmentation

A minimalistic PyTorch-based audio augmentation library for speech and audio augmentation. The goal of this library is to provide a general purpose speech augmentation policy that can be used on any task and perform well without having to tune augmentation hyperparameters. Just install, and start augmenting. Applies two random augmentations per call.

![Diagram](https://raw.githubusercontent.com/gfdb/wav2aug/main/wav2aug.png)

## ⚙️ Features

* **Minimal dependencies**: we only rely on PyTorch, torchcodec, and torchaudio.
* **9 core augmentations**: amplitude scaling/clipping, noise addition, frequency dropout, polarity inversion, chunk swapping, speed perturbation, time dropout, and babble noise.
* **Simplicity**: just install and start augmenting!

## 📦 Installation

### pip

```bash
pip install wav2aug
```

### uv

```bash
uv add wav2aug
```

## 🚀 Quick Start

```python
import torch
from wav2aug.gpu import Wav2Aug

# Initialize the augmenter once
augmenter = Wav2Aug(sample_rate=16000)

# in the forward pass
wavs = torch.randn(3, 50000)
lens = torch.tensor([1])

aug_wavs, aug_lens = augmenter(wavs, lens)
```

That's it!

## 🧪 Augmentation Types

* 🔊 **Amplitude Scaling/Clipping**: Random gain and peak limiting
* 🌫️ **Noise Addition**: Environmental noise with SNR control
* 📶 **Frequency Dropout**: Spectral masking with random notch filters
* 🔄 **Polarity Inversion**: Random phase flip
* 🧩 **Chunk Swapping**: Temporal segment reordering
* ⏱️ **Speed Perturbation**: Time-scale modification
* 🕳️ **Time Dropout**: Random silence insertion
* 👥 **Babble Noise**: Multi-speaker background (auto-enabled with sufficient buffer)

## 🛠️ Development Installation

### uv

```bash
git clone https://github.com/gfdb/wav2aug
cd wav2aug

# create venv and pin Python
uv venv
source .venv/bin/activate
uv python pin 3.10  # or 3.11/3.12

# runtime only
uv sync

# extras
uv sync --extra dev
uv sync --extra test
```

### pip

```bash
git clone https://github.com/gfdb/wav2aug
cd wav2aug

# create venv
python -m venv .venv
source .venv/bin/activate

# runtime only
python -m pip install .

# editable + extras for development
python -m pip install -e '.[dev,test]'
```

## ✅ Tests

### uv

```bash
uv run pytest -q tests/
```

### pip

```bash
pytest -q tests/
```

## 🤝 Contributing

* Issues and PRs are welcome and encouraged!

* Bug reports: please open an issue with a minimal repro (env, torch/torchaudio/torchcodec versions, code snippet, expected vs. actual, traceback).

* Feature requests: please open an issue with use-case and proposed feature.

* PRs: keep them focused. Add tests when behavior changes. Don't forget to run formatters and tests before submitting!