from __future__ import annotations

import json
import argparse

import torch

from trainingless_adaptation.config import load_config
from trainingless_adaptation.data import clean_batches
from trainingless_adaptation.experiments import build_adapter, corpus_from_config
from trainingless_adaptation.reproducibility import set_deterministic, state_dict_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strict.yaml")
    config = load_config(parser.parse_args().config)
    set_deterministic(config.seed)
    device = torch.device("cuda")
    corpus = corpus_from_config(config)
    results = []
    for name in ("passt", "msclap", "beats"):
        adapter = build_adapter(config, name, device, corpus, 1)
        item = corpus.items[0]
        _, waveforms = next(
            clean_batches(
                corpus,
                [item],
                adapter.sample_rate,
                adapter.input_seconds,
                adapter.repeat_short_audio,
                1,
            )
        )
        before = state_dict_sha256(adapter.module)
        conventional = adapter.predict(waveforms, False, 1000.0, int(config.model(name)["default_ending_block"]), True)
        proposed = adapter.predict(waveforms, True, 1000.0, int(config.model(name)["default_ending_block"]), True)
        after = state_dict_sha256(adapter.module)
        if conventional.shape != (1, 50) or proposed.shape != (1, 50):
            raise RuntimeError(f"Unexpected {name} output shapes: {conventional.shape}, {proposed.shape}")
        if not torch.isfinite(conventional).all() or not torch.isfinite(proposed).all():
            raise RuntimeError(f"Non-finite logits from {name}")
        if before != after:
            raise RuntimeError(f"Parameters changed during {name} smoke inference")
        results.append(
            {
                "model": name,
                "conventional_top1": int(conventional.argmax(dim=-1).item()),
                "proposed_top1": int(proposed.argmax(dim=-1).item()),
                "mean_absolute_logit_delta": float((conventional - proposed).abs().mean().cpu()),
                "parameter_sha256_before": before,
                "parameter_sha256_after": after,
                "parameters_unchanged": before == after,
                "passed": before == after,
            }
        )
        del adapter
        torch.cuda.empty_cache()
    destination = config.manifest_dir / "model-smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
