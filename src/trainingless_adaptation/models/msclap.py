from __future__ import annotations

import importlib
import sys
from pathlib import Path

import librosa
import torch
import torch.nn.functional as functional
import yaml
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from ..adaptation import TokenGrid, capture_silent_outputs, replace_high_frequency_tokens, replacement_hooks
from ..dsp import map_cutoff_bin, mel_cutoff_bin
from .base import AudioModelAdapter, load_best_matching_state


class OfflineTextEncoder(torch.nn.Module):
    def __init__(self, projection_class: type[torch.nn.Module]) -> None:
        super().__init__()
        self.text_model = "gpt2"
        self.base = GPT2Model(GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12))
        self.projection = projection_class(768, 1024)

    def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.base(**tokens)[0]
        sequence_lengths = torch.ne(tokens["input_ids"], 0).sum(-1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), sequence_lengths]
        return self.projection(pooled)


class OfflineCLAP(torch.nn.Module):
    def __init__(self, audio_encoder_class: type[torch.nn.Module], projection_class: type[torch.nn.Module]) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder_class("HTSAT", 768, 1024, 44100, 1024, 320, 64, 50, 8000, 527)
        self.caption_encoder = OfflineTextEncoder(projection_class)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())


class MSCLAPAdapter(AudioModelAdapter):
    name = "msclap"
    sample_rate = 44100
    input_seconds = 7.0
    repeat_short_audio = True

    def __init__(
        self,
        vendor_root: Path,
        checkpoint: Path,
        gpt2_dir: Path,
        categories: list[str],
        prompt: str,
        device: torch.device,
    ) -> None:
        super().__init__(device)
        sys.path.insert(0, str(vendor_root))
        try:
            clap_module = importlib.import_module("msclap.models.clap")
        finally:
            sys.path.pop(0)
        self.model = OfflineCLAP(clap_module.AudioEncoder, clap_module.Projection)
        load_best_matching_state(self.model, torch.load(checkpoint, map_location="cpu"))
        self.model.to(device).eval()
        self.tokenizer = GPT2Tokenizer(
            vocab_file=str(gpt2_dir / "encoder.json"),
            merges_file=str(gpt2_dir / "vocab.bpe"),
        )
        self.tokenizer.add_special_tokens({"pad_token": "!"})
        self.prompt = prompt
        self.categories = categories
        self.text_embeddings = self._text_embeddings()
        self.htsat = self.model.audio_encoder.base.htsat
        self.blocks = list(self.htsat.layers)
        self.mel_centers = librosa.mel_frequencies(n_mels=64, fmin=50.0, fmax=8000.0, htk=False)
        self._silent_cache: dict[int, list[torch.Tensor]] = {}

    @property
    def module(self) -> torch.nn.Module:
        return self.model

    def _text_embeddings(self) -> torch.Tensor:
        prompts = [self.prompt.format(label=category.replace("_", " ")) for category in self.categories]
        encoded = []
        for prompt in prompts:
            tokens = self.tokenizer.encode_plus(
                text=prompt + " <|endoftext|>",
                add_special_tokens=True,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoded.append({key: value.reshape(-1) for key, value in tokens.items() if key in {"input_ids", "attention_mask"}})
        batch = {key: torch.stack([item[key] for item in encoded]).to(self.device) for key in encoded[0]}
        with torch.inference_mode():
            embeddings = self.model.caption_encoder(batch)
        return functional.normalize(embeddings, dim=-1)

    def _grid(self, block_index: int) -> TokenGrid:
        block = self.blocks[block_index]
        height, width = block.input_resolution
        if block.downsample is not None:
            height, width = height // 2, width // 2
        repeats = int(self.htsat.freq_ratio)
        if height % repeats:
            raise ValueError(f"HTS-AT frequency layout {height} is not divisible by {repeats}")
        return TokenGrid(height // repeats, width, frequency_repeats=repeats)

    def _silent(self, samples: int) -> list[torch.Tensor]:
        if samples not in self._silent_cache:
            silent = torch.zeros(1, samples, device=self.device)
            self._silent_cache[samples] = capture_silent_outputs(
                self.blocks,
                lambda: self.model.audio_encoder(silent),
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
        waveforms = waveforms.to(self.device)
        if proposed:
            silent = self._silent(waveforms.shape[-1])
            mel_bin = mel_cutoff_bin(self.mel_centers, cutoff_hz)

            def replace(index: int, tokens: torch.Tensor, silent_tokens: torch.Tensor) -> torch.Tensor:
                grid = self._grid(index)
                cutoff_bin = map_cutoff_bin(mel_bin, 64, grid.frequency_bins)
                return replace_high_frequency_tokens(tokens, silent_tokens, grid, cutoff_bin, mask_inclusive)

            with replacement_hooks(self.blocks, silent, ending_block, replace):
                audio_embeddings = self.model.audio_encoder(waveforms)[0]
        else:
            audio_embeddings = self.model.audio_encoder(waveforms)[0]
        audio_embeddings = functional.normalize(audio_embeddings, dim=-1)
        return self.model.logit_scale.exp() * audio_embeddings @ self.text_embeddings.T
