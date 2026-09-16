from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from .dsp import CorruptionSpec, corrupt, to_mono_float32
from .reproducibility import sample_seed


@dataclass(frozen=True)
class ESC50Item:
    filename: str
    fold: int
    target: int
    category: str


class ESC50Corpus:
    def __init__(self, root: Path, metadata_file: str, audio_dir: str, source_sample_rate: int) -> None:
        self.root = root
        self.audio_root = root / audio_dir
        self.source_sample_rate = source_sample_rate
        frame = pd.read_csv(root / metadata_file).sort_values("filename").reset_index(drop=True)
        self.items = [
            ESC50Item(str(row.filename), int(row.fold), int(row.target), str(row.category))
            for row in frame.itertuples(index=False)
        ]

    @property
    def categories(self) -> list[str]:
        mapping = {item.target: item.category for item in self.items}
        return [mapping[index] for index in sorted(mapping)]

    def select(self, folds: Sequence[int] | None = None) -> list[ESC50Item]:
        if folds is None:
            return list(self.items)
        selected = set(int(fold) for fold in folds)
        return [item for item in self.items if item.fold in selected]

    def load(self, item: ESC50Item) -> np.ndarray:
        audio, sample_rate = sf.read(self.audio_root / item.filename, dtype="float32", always_2d=False)
        if sample_rate != self.source_sample_rate:
            raise ValueError(f"Unexpected sample rate for {item.filename}: {sample_rate}")
        return to_mono_float32(audio)


def resample_and_fit(audio: np.ndarray, source_rate: int, target_rate: int, seconds: float, repeat: bool) -> torch.Tensor:
    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if source_rate != target_rate:
        waveform = audio_functional.resample(waveform, source_rate, target_rate)
    target_samples = int(round(target_rate * seconds))
    if waveform.numel() < target_samples:
        if repeat and waveform.numel() > 0:
            repeats = (target_samples + waveform.numel() - 1) // waveform.numel()
            waveform = waveform.repeat(repeats)
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))
    return waveform[:target_samples].contiguous()


def corrupted_batches(
    corpus: ESC50Corpus,
    items: Sequence[ESC50Item],
    spec: CorruptionSpec,
    global_seed: int,
    target_rate: int,
    target_seconds: float,
    repeat: bool,
    batch_size: int,
) -> Iterator[tuple[list[ESC50Item], torch.Tensor]]:
    batch_items: list[ESC50Item] = []
    batch_audio: list[torch.Tensor] = []
    for item in items:
        clean = corpus.load(item)
        seed = sample_seed(item.filename, spec.order, spec.snr_db, global_seed)
        shifted, _, _ = corrupt(clean, corpus.source_sample_rate, spec, seed)
        batch_items.append(item)
        batch_audio.append(resample_and_fit(shifted, corpus.source_sample_rate, target_rate, target_seconds, repeat))
        if len(batch_items) == batch_size:
            yield batch_items, torch.stack(batch_audio)
            batch_items, batch_audio = [], []
    if batch_items:
        yield batch_items, torch.stack(batch_audio)


def clean_batches(
    corpus: ESC50Corpus,
    items: Sequence[ESC50Item],
    target_rate: int,
    target_seconds: float,
    repeat: bool,
    batch_size: int,
) -> Iterator[tuple[list[ESC50Item], torch.Tensor]]:
    batch_items: list[ESC50Item] = []
    batch_audio: list[torch.Tensor] = []
    for item in items:
        batch_items.append(item)
        batch_audio.append(resample_and_fit(corpus.load(item), corpus.source_sample_rate, target_rate, target_seconds, repeat))
        if len(batch_items) == batch_size:
            yield batch_items, torch.stack(batch_audio)
            batch_items, batch_audio = [], []
    if batch_items:
        yield batch_items, torch.stack(batch_audio)
