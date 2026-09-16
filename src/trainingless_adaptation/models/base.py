from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class AudioModelAdapter(ABC):
    name: str
    sample_rate: int
    input_seconds: float
    repeat_short_audio: bool = False

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def predict(
        self,
        waveforms: torch.Tensor,
        proposed: bool,
        cutoff_hz: float,
        ending_block: int,
        mask_inclusive: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def module(self) -> torch.nn.Module:
        raise NotImplementedError

    def load_fold(self, fold: int, checkpoint: Path | None = None) -> None:
        if checkpoint is not None:
            raise NotImplementedError(f"{self.name} does not support fold checkpoints")


def normalized_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint)!r}")
    state: object = checkpoint
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("No state dictionary found in checkpoint")
    return {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}


def load_best_matching_state(module: torch.nn.Module, checkpoint: object) -> tuple[list[str], list[str]]:
    raw = normalized_state_dict(checkpoint)
    target = module.state_dict()
    prefixes = ("", "module.", "model.", "net.", "model.net.", "module.net.")
    candidates: list[dict[str, torch.Tensor]] = []
    for prefix in prefixes:
        mapped = {
            key[len(prefix):] if prefix and key.startswith(prefix) else key: value
            for key, value in raw.items()
        }
        compatible = {key: value for key, value in mapped.items() if key in target and target[key].shape == value.shape}
        candidates.append(compatible)
    best = max(candidates, key=len)
    if not best:
        raise RuntimeError("Checkpoint has no tensors compatible with the target model")
    missing, unexpected = module.load_state_dict(best, strict=False)
    return list(missing), list(unexpected)
