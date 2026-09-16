from __future__ import annotations

import json
import argparse

import torch

from trainingless_adaptation.adaptation import TokenGrid, capture_silent_outputs
from trainingless_adaptation.config import load_config
from trainingless_adaptation.experiments import build_adapter, corpus_from_config
from trainingless_adaptation.reproducibility import set_deterministic


def roundtrip(tokens: torch.Tensor, grid: TokenGrid) -> bool:
    batch = tokens.transpose(0, 1) if grid.sequence_first else tokens
    end = grid.token_offset + grid.patch_tokens
    patches = batch[:, grid.token_offset:end]
    if grid.time_major:
        shaped = patches.reshape(
            batch.shape[0], grid.time_bins, grid.frequency_repeats, grid.frequency_bins, batch.shape[-1]
        )
    else:
        shaped = patches.reshape(
            batch.shape[0], grid.frequency_repeats, grid.frequency_bins, grid.time_bins, batch.shape[-1]
        )
    return bool(torch.equal(patches, shaped.reshape_as(patches)))


def direct_silent(adapter, samples: int) -> tuple[list[torch.Tensor], list[TokenGrid]]:
    if adapter.name == "passt":
        mel = adapter.mel(torch.zeros(1, samples, device=adapter.device))
        outputs = capture_silent_outputs(adapter.blocks, lambda: adapter.model(mel.unsqueeze(1)))
        grids = [adapter._grid(mel)] * len(outputs)
    elif adapter.name == "msclap":
        silent = torch.zeros(1, samples, device=adapter.device)
        outputs = capture_silent_outputs(adapter.blocks, lambda: adapter.model.audio_encoder(silent))
        grids = [adapter._grid(index) for index in range(len(outputs))]
    else:
        outputs = capture_silent_outputs(adapter.blocks, lambda: adapter.model(torch.zeros(1, samples)))
        grids = [adapter._grid()] * len(outputs)
    return outputs, grids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strict.yaml")
    config = load_config(parser.parse_args().config)
    set_deterministic(config.seed)
    device = torch.device("cuda")
    corpus = corpus_from_config(config)
    results = []
    passed = True
    for model_name in ("passt", "msclap", "beats"):
        adapter = build_adapter(config, model_name, device, corpus, fold=1)
        samples = int(adapter.sample_rate * adapter.input_seconds)
        cached = adapter._silent(samples)
        direct, grids = direct_silent(adapter, samples)
        max_error = max(float((left - right).abs().max().item()) for left, right in zip(cached, direct))
        grid_checks = [roundtrip(tokens, grid) for tokens, grid in zip(direct, grids)]
        shape_checks = []
        for tokens, grid in zip(direct, grids):
            sequence = tokens.shape[0] if grid.sequence_first else tokens.shape[1]
            shape_checks.append(grid.token_offset + grid.patch_tokens <= sequence)
        model_passed = max_error == 0.0 and all(grid_checks) and all(shape_checks)
        passed = passed and model_passed
        results.append(
            {
                "model": model_name,
                "blocks": len(direct),
                "silent_cache_max_absolute_error": max_error,
                "token_tf_roundtrip": all(grid_checks),
                "grid_within_sequence": all(shape_checks),
                "passed": model_passed,
            }
        )
        del adapter
        torch.cuda.empty_cache()

    destination = config.manifest_dir / "adaptation-integration.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not passed:
        raise SystemExit("Adaptation integration acceptance check failed")


if __name__ == "__main__":
    main()
