from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from ..adaptation import TokenGrid, capture_silent_outputs, replace_high_frequency_tokens, replacement_hooks
from ..dsp import kaldi_mel_centers, map_cutoff_bin, mel_cutoff_bin
from .base import AudioModelAdapter


class BEATsClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, classes: int = 50) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(backbone.cfg.encoder_embed_dim, classes)
        self.grid_shape: tuple[int, int] | None = None

    def encode_fbank(self, fbank: torch.Tensor) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        fbank = fbank.to(device)
        features = self.backbone.patch_embedding(fbank.unsqueeze(1))
        self.grid_shape = (int(features.shape[-2]), int(features.shape[-1]))
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        features = self.backbone.layer_norm(features)
        if self.backbone.post_extract_proj is not None:
            features = self.backbone.post_extract_proj(features)
        encoded, _ = self.backbone.encoder(self.backbone.dropout_input(features), padding_mask=None)
        return encoded

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        fbank = self.backbone.preprocess(waveforms.detach().cpu())
        return self.encode_fbank(fbank)

    def forward_fbank(self, fbank: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_fbank(fbank).mean(dim=1))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(waveforms).mean(dim=1))


class BEATsAdapter(AudioModelAdapter):
    name = "beats"
    sample_rate = 16000
    input_seconds = 5.0

    def __init__(
        self,
        vendor_root: Path,
        pretrained_checkpoint: Path,
        fold_checkpoint: Path | None,
        device: torch.device,
    ) -> None:
        super().__init__(device)
        sys.path.insert(0, str(vendor_root))
        try:
            beats_module = importlib.import_module("BEATs")
        finally:
            sys.path.pop(0)
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
        config = beats_module.BEATsConfig(checkpoint["cfg"])
        config.finetuned_model = False
        backbone = beats_module.BEATs(config)
        backbone.load_state_dict(checkpoint["model"], strict=True)
        self.model = BEATsClassifier(backbone)
        self.blocks = list(self.model.backbone.encoder.layers)
        # §IV-A, Eqs. (5)-(6): match torchaudio Kaldi fbank defaults used by BEATs.
        self.mel_centers = kaldi_mel_centers(128, 512, 16000.0, 20.0, 0.0)
        self._silent_cache: dict[int, list[torch.Tensor]] = {}
        if fold_checkpoint is not None:
            self.load_fold(0, fold_checkpoint)
        else:
            self.model.to(device).eval()

    @property
    def module(self) -> torch.nn.Module:
        return self.model

    def load_fold(self, fold: int, checkpoint: Path | None = None) -> None:
        if checkpoint is None:
            raise ValueError("BEATs fold loading requires a checkpoint")
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self._silent_cache.clear()

    def _grid(self) -> TokenGrid:
        if self.model.grid_shape is None:
            raise RuntimeError("BEATs patch grid is unavailable before encoding")
        time_bins, frequency_bins = self.model.grid_shape
        return TokenGrid(frequency_bins, time_bins, sequence_first=True, time_major=True)

    def _silent(self, samples: int) -> list[torch.Tensor]:
        if samples not in self._silent_cache:
            self._silent_cache[samples] = capture_silent_outputs(
                self.blocks,
                lambda: self.model(torch.zeros(1, samples)),
            )
        return self._silent_cache[samples]

    @torch.inference_mode()
    def predict(
        self,
        waveforms: torch.Tensor,
        proposed: bool,
        cutoff_hz: float,
        ending_block: int,
        mask_inclusive: bool,
    ) -> torch.Tensor:
        waveforms = waveforms.cpu()
        if not proposed:
            return self.model(waveforms)
        silent = self._silent(waveforms.shape[-1])
        mel_bin = mel_cutoff_bin(self.mel_centers, cutoff_hz)

        def replace(_index: int, tokens: torch.Tensor, silent_tokens: torch.Tensor) -> torch.Tensor:
            grid = self._grid()
            cutoff_bin = map_cutoff_bin(mel_bin, 128, grid.frequency_bins)
            return replace_high_frequency_tokens(tokens, silent_tokens, grid, cutoff_bin, mask_inclusive)

        with replacement_hooks(self.blocks, silent, ending_block, replace):
            return self.model(waveforms)
