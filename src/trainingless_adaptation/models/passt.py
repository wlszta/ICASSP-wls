from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import torchaudio

from ..adaptation import TokenGrid, capture_silent_outputs, replace_high_frequency_tokens, replacement_hooks
from ..dsp import kaldi_mel_centers, map_cutoff_bin, mel_cutoff_bin
from .base import AudioModelAdapter, load_best_matching_state


class PaSSTMel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = 32000
        self.n_mels = 128
        self.n_fft = 1024
        self.win_length = 800
        self.hop_length = 320
        self.fmin = 0.0
        self.fmax = 15000.0
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False), persistent=False)
        self.register_buffer("preemphasis", torch.tensor([[[-0.97, 1.0]]]), persistent=False)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        emphasized = functional.conv1d(waveforms.unsqueeze(1), self.preemphasis).squeeze(1)
        spectrum = torch.stft(
            emphasized,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=True,
        ).abs().square()
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sample_rate,
            self.fmin,
            self.fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = functional.pad(mel_basis, (0, 1)).to(spectrum.device, spectrum.dtype)
        mel = torch.matmul(mel_basis, spectrum)
        return (torch.log(mel + 1e-5) + 4.5) / 5.0


def _install_ba3l_stub() -> None:
    if "ba3l.ingredients.ingredient" in sys.modules:
        return

    class Ingredient:
        def __init__(self, _name: str) -> None:
            pass

        def command(self, function=None):
            return function if function is not None else lambda value: value

        def add_config(self, **_kwargs) -> None:
            return None

    ba3l = types.ModuleType("ba3l")
    ingredients = types.ModuleType("ba3l.ingredients")
    ingredient = types.ModuleType("ba3l.ingredients.ingredient")
    ingredient.Ingredient = Ingredient
    sys.modules["ba3l"] = ba3l
    sys.modules["ba3l.ingredients"] = ingredients
    sys.modules["ba3l.ingredients.ingredient"] = ingredient


class PaSSTAdapter(AudioModelAdapter):
    name = "passt"
    sample_rate = 32000
    input_seconds = 5.0

    def __init__(self, vendor_root: Path, checkpoint: Path, device: torch.device) -> None:
        super().__init__(device)
        _install_ba3l_stub()
        sys.path.insert(0, str(vendor_root))
        try:
            passt = importlib.import_module("models.passt")
        finally:
            sys.path.pop(0)
        self.model = passt.get_model(
            arch="passt_s_swa_p16_128_ap476",
            pretrained=False,
            n_classes=50,
            in_channels=1,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=998,
            u_patchout=0,
            s_patchout_t=0,
            s_patchout_f=0,
        )
        load_best_matching_state(self.model, torch.load(checkpoint, map_location="cpu"))
        self.model.to(device).eval()
        self.mel = PaSSTMel().to(device).eval()
        self.blocks = list(self.model.blocks)
        # §IV-A, Eqs. (5)-(6): match PaSST's Kaldi-compatible mel centers.
        self.mel_centers = kaldi_mel_centers(128, 1024, 32000.0, 0.0, 15000.0)
        self._silent_cache: dict[tuple[int, int], list[torch.Tensor]] = {}

    @property
    def module(self) -> torch.nn.Module:
        return self.model

    def load_fold(self, fold: int, checkpoint: Path | None = None) -> None:
        if checkpoint is None:
            raise ValueError("PaSST fold loading requires a checkpoint")
        load_best_matching_state(self.model, torch.load(checkpoint, map_location="cpu"))
        self.model.to(self.device).eval()
        self._silent_cache.clear()

    def _grid(self, mel: torch.Tensor) -> TokenGrid:
        convolution = self.model.patch_embed.proj
        frequency = (mel.shape[-2] + 2 * convolution.padding[0] - convolution.dilation[0] * (convolution.kernel_size[0] - 1) - 1) // convolution.stride[0] + 1
        time = (mel.shape[-1] + 2 * convolution.padding[1] - convolution.dilation[1] * (convolution.kernel_size[1] - 1) - 1) // convolution.stride[1] + 1
        return TokenGrid(int(frequency), int(time), token_offset=int(self.model.num_tokens))

    def _silent(self, samples: int) -> list[torch.Tensor]:
        key = (samples, self.device.index or 0)
        if key not in self._silent_cache:
            silent_mel = self.mel(torch.zeros(1, samples, device=self.device))
            self._silent_cache[key] = capture_silent_outputs(
                self.blocks,
                lambda: self.model(silent_mel.unsqueeze(1)),
            )
        return self._silent_cache[key]

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
        mel = self.mel(waveforms)
        if not proposed:
            return self.model(mel.unsqueeze(1))[0]
        grid = self._grid(mel)
        mel_bin = mel_cutoff_bin(self.mel_centers, cutoff_hz)
        cutoff_bin = map_cutoff_bin(mel_bin, 128, grid.frequency_bins)
        silent = self._silent(waveforms.shape[-1])

        def replace(_index: int, tokens: torch.Tensor, silent_tokens: torch.Tensor) -> torch.Tensor:
            return replace_high_frequency_tokens(tokens, silent_tokens, grid, cutoff_bin, mask_inclusive)

        with replacement_hooks(self.blocks, silent, ending_block, replace):
            return self.model(mel.unsqueeze(1))[0]
