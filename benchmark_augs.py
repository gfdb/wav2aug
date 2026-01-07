#!/usr/bin/env python
"""Benchmark all wav2aug augmentations."""

import time

import torch
from tqdm import tqdm

# Benchmark settings
BATCH_SIZE = 32
SAMPLE_RATE = 16000
DURATION_SEC = 1.0
NUM_ITERATIONS = 100
WARMUP = 50

# Create realistic waveform (simulated speech-like signal)
length = int(SAMPLE_RATE * DURATION_SEC)
waveforms = torch.randn(BATCH_SIZE, length) * 0.3  # Typical speech amplitude

print(f"Benchmark settings:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sample rate: {SAMPLE_RATE}")
print(f"  Duration: {DURATION_SEC}s ({length} samples)")
print(f"  Iterations: {NUM_ITERATIONS} (warmup: {WARMUP})")
print()

from wav2aug.data.fetch import ensure_pack
# Import augmentations
from wav2aug.gpu import (NoiseLoader, add_babble_noise, add_noise, chunk_swap,
                         freq_drop, invert_polarity, rand_amp_clip,
                         rand_amp_scale, speed_perturb, time_dropout)

# Setup noise loader
print("Loading noise pack...")
noise_dir = ensure_pack("pointsource_noises")
noise_loader = NoiseLoader(noise_dir, sample_rate=SAMPLE_RATE, preload=True)
print(f"Noise loaded: {noise_loader.preloaded_duration_seconds:.0f}s of audio\n")

# Define augmentations to benchmark
augmentations = {
    "add_noise": lambda x: add_noise(x, noise_loader, snr_low=0, snr_high=10),
    "add_babble_noise": lambda x: add_babble_noise(x, snr_low=0, snr_high=20),
    "chunk_swap": lambda x: chunk_swap(x),
    "freq_drop": lambda x: freq_drop(x),
    "invert_polarity": lambda x: invert_polarity(x),
    "rand_amp_clip": lambda x: rand_amp_clip(x),
    "rand_amp_scale": lambda x: rand_amp_scale(x),
    "speed_perturb": lambda x: speed_perturb(x, sample_rate=SAMPLE_RATE),
    "time_dropout": lambda x: time_dropout(x, sample_rate=SAMPLE_RATE),
}

results = {}

for name, aug_fn in augmentations.items():
    # Fresh waveforms for each augmentation
    x = waveforms.clone()
    
    # Warmup
    for _ in range(WARMUP):
        _ = aug_fn(x.clone())
    
    # Benchmark
    start = time.perf_counter()
    for _ in tqdm(range(NUM_ITERATIONS)):
        _ = aug_fn(x.clone())
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / NUM_ITERATIONS) * 1000
    results[name] = avg_ms
    print(f"{name:20s}: {avg_ms:8.3f} ms/batch")

print()
print("=" * 45)
print("Summary (sorted by speed):")
print("=" * 45)
for name, ms in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:20s}: {ms:8.3f} ms/batch")
