from __future__ import annotations

import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .config import ExperimentConfig
from .constants import PAPER_TABLE_I, PASST_SUFFIX
from .data import ESC50Corpus, ESC50Item, clean_batches, corrupted_batches
from .dsp import CorruptionSpec
from .models import BEATsAdapter, MSCLAPAdapter, PaSSTAdapter
from .reproducibility import set_deterministic


def corpus_from_config(config: ExperimentConfig) -> ESC50Corpus:
    data = config.section("data")
    return ESC50Corpus(
        config.resolve(data["esc50_dir"]),
        data["metadata_file"],
        data["audio_dir"],
        int(data["source_sample_rate"]),
    )


def passt_checkpoint(config: ExperimentConfig, fold: int) -> Path:
    suffix = PASST_SUFFIX[fold]
    relative = config.values["weights"]["passt_pattern"].format(fold=fold, suffix=suffix)
    return config.resolve(config.values["weights"]["root"]) / relative


def beats_fold_checkpoint(config: ExperimentConfig, fold: int) -> Path:
    training = config.values.get("beats_training", {})
    checkpoint_root = training.get("checkpoint_root", "checkpoints/beats")
    return config.resolve(checkpoint_root) / f"fold{fold}.pt"


def build_adapter(config: ExperimentConfig, name: str, device: torch.device, corpus: ESC50Corpus, fold: int = 1):
    vendor_root = config.resolve(config.values["vendor"]["root"])
    weight_root = config.resolve(config.values["weights"]["root"])
    if name == "passt":
        return PaSSTAdapter(vendor_root / "PaSST", passt_checkpoint(config, fold), device)
    if name == "msclap":
        model_config = config.model("msclap")
        return MSCLAPAdapter(
            vendor_root / "MSCLAP",
            weight_root / config.values["weights"]["msclap"],
            config.resolve(config.values["weights"]["gpt2_dir"]),
            corpus.categories,
            model_config["prompt"],
            device,
        )
    if name == "beats":
        return BEATsAdapter(
            vendor_root / "unilm" / "beats",
            weight_root / config.values["weights"]["beats"],
            beats_fold_checkpoint(config, fold),
            device,
        )
    raise ValueError(f"Unknown model: {name}")


def condition_path(root: Path, model: str, fold: int, snr_db: int, order: int, ending_block: int) -> Path:
    return root / "predictions" / model / f"fold{fold}" / f"snr{snr_db}_order{order}_end{ending_block}.csv"


def evaluate_condition(
    config: ExperimentConfig,
    corpus: ESC50Corpus,
    adapter,
    items: list[ESC50Item],
    fold: int,
    output_path: Path,
    cutoff_hz: float,
    order: int,
    snr_db: int,
    noise_reference: str,
    mask_inclusive: bool,
    ending_block: int,
    global_seed: int,
    filter_mode: str = "causal",
    methods: tuple[str, ...] = ("conventional", "proposed"),
) -> pd.DataFrame:
    model_config = config.model(adapter.name)
    rows: list[dict[str, object]] = []
    batches = corrupted_batches(
        corpus,
        items,
        CorruptionSpec(cutoff_hz, order, snr_db, noise_reference, filter_mode),
        global_seed,
        adapter.sample_rate,
        adapter.input_seconds,
        adapter.repeat_short_audio,
        int(model_config["batch_size"]),
    )
    for batch_items, waveforms in tqdm(batches, desc=f"{adapter.name} f{fold} {snr_db}/{order}", leave=False):
        predictions = {
            method: adapter.predict(
                waveforms,
                method == "proposed",
                cutoff_hz,
                ending_block,
                mask_inclusive,
            ).argmax(dim=-1).cpu().tolist()
            for method in methods
        }
        for item_index, item in enumerate(batch_items):
            for method in methods:
                prediction = predictions[method][item_index]
                rows.append(
                    {
                        "filename": item.filename,
                        "fold": item.fold,
                        "target": item.target,
                        "prediction": prediction,
                        "correct": int(prediction == item.target),
                        "model": adapter.name,
                        "method": method,
                        "snr_db": snr_db,
                        "order": order,
                        "cutoff_hz": cutoff_hz,
                        "noise_reference": noise_reference,
                        "mask_inclusive": mask_inclusive,
                        "filter_mode": filter_mode,
                        "ending_block": ending_block,
                        "seed": global_seed,
                    }
                )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def summarize_table(predictions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        predictions.groupby(["model", "method", "snr_db", "order"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )
    summary["accuracy"] *= 100.0
    summary["paper_accuracy"] = [
        PAPER_TABLE_I[(row.model, row.method, int(row.snr_db), int(row.order))]
        for row in summary.itertuples(index=False)
    ]
    summary["absolute_error_pp"] = (summary["accuracy"] - summary["paper_accuracy"]).abs()
    return summary.sort_values(["model", "method", "snr_db", "order"]).reset_index(drop=True)


def run_table1(
    config: ExperimentConfig,
    track: str,
    device_name: str,
    resume: bool,
    cutoff_hz: float | None = None,
    noise_reference: str | None = None,
    mask_inclusive: bool | None = None,
    seed: int | None = None,
    output_root: Path | None = None,
    conventional_source_root: Path | None = None,
) -> pd.DataFrame:
    set_deterministic(config.seed if seed is None else seed)
    device = torch.device(device_name)
    corpus = corpus_from_config(config)
    corruption = config.section("corruption")
    cutoff_hz = float(corruption["cutoff_hz"] if cutoff_hz is None else cutoff_hz)
    noise_reference = str(corruption["noise_reference"] if noise_reference is None else noise_reference)
    mask_inclusive = bool(corruption["mask_inclusive"] if mask_inclusive is None else mask_inclusive)
    filter_mode = str(corruption.get("filter_mode", "causal"))
    global_seed = config.seed if seed is None else seed
    root = output_root or config.output_dir(track) / "table1"
    root.mkdir(parents=True, exist_ok=True)
    config.dump_run(
        root / "resolved_config.yaml",
        command="run-table1",
        track=track,
        device=device_name,
        resume=resume,
        cutoff_hz=cutoff_hz,
        noise_reference=noise_reference,
        mask_inclusive=mask_inclusive,
        filter_mode=filter_mode,
        seed=global_seed,
        conventional_source_root=str(conventional_source_root) if conventional_source_root is not None else None,
    )
    frames: list[pd.DataFrame] = []
    for model_name in ("passt", "msclap", "beats"):
        folds = [0] if model_name == "msclap" else [1, 2, 3, 4, 5]
        for fold in folds:
            adapter = build_adapter(config, model_name, device, corpus, max(fold, 1))
            items = corpus.items if fold == 0 else corpus.select([fold])
            ending_block = int(config.model(model_name)["default_ending_block"])
            for snr_db, order in itertools.product(corruption["snrs_db"], corruption["orders"]):
                path = condition_path(root, model_name, fold, int(snr_db), int(order), ending_block)
                if resume and path.exists():
                    frame = pd.read_csv(path)
                else:
                    conventional_path = (
                        condition_path(
                            conventional_source_root,
                            model_name,
                            fold,
                            int(snr_db),
                            int(order),
                            ending_block,
                        )
                        if conventional_source_root is not None
                        else None
                    )
                    if conventional_path is not None and conventional_path.exists():
                        proposed = evaluate_condition(
                            config,
                            corpus,
                            adapter,
                            items,
                            fold,
                            path,
                            cutoff_hz,
                            int(order),
                            int(snr_db),
                            noise_reference,
                            mask_inclusive,
                            ending_block,
                            global_seed,
                            filter_mode,
                            ("proposed",),
                        )
                        conventional = pd.read_csv(conventional_path)
                        conventional = conventional[conventional.method == "conventional"].copy()
                        conventional["cutoff_hz"] = cutoff_hz
                        conventional["noise_reference"] = noise_reference
                        conventional["mask_inclusive"] = mask_inclusive
                        conventional["filter_mode"] = filter_mode
                        conventional["seed"] = global_seed
                        frame = pd.concat([conventional, proposed], ignore_index=True)
                        frame.to_csv(path, index=False)
                    else:
                        frame = evaluate_condition(
                            config,
                            corpus,
                            adapter,
                            items,
                            fold,
                            path,
                            cutoff_hz,
                            int(order),
                            int(snr_db),
                            noise_reference,
                            mask_inclusive,
                            ending_block,
                            global_seed,
                            filter_mode,
                        )
                frames.append(frame)
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(root / "predictions.csv", index=False)
    summary = summarize_table(predictions)
    summary.to_csv(root / "table1.csv", index=False)
    (root / "table1.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    return summary


def run_figure7(config: ExperimentConfig, track: str, device_name: str, resume: bool) -> pd.DataFrame:
    set_deterministic(config.seed)
    device = torch.device(device_name)
    corpus = corpus_from_config(config)
    corruption = config.section("corruption")
    if track == "aligned":
        selection_path = config.output_dir("aligned") / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
        corruption["cutoff_hz"] = selection["cutoff_hz"]
        corruption["noise_reference"] = selection["noise_reference"]
        corruption["mask_inclusive"] = selection["mask_inclusive"]
    root = config.output_dir(track) / "figure7"
    config.dump_run(
        root / "resolved_config.yaml",
        command="run-figure7",
        track=track,
        device=device_name,
        resume=resume,
        cutoff_hz=float(corruption["cutoff_hz"]),
        noise_reference=str(corruption["noise_reference"]),
        mask_inclusive=bool(corruption["mask_inclusive"]),
        filter_mode=str(corruption.get("filter_mode", "causal")),
        seed=config.seed,
    )
    frames: list[pd.DataFrame] = []
    for model_name in ("passt", "msclap", "beats"):
        folds = [0] if model_name == "msclap" else [1, 2, 3, 4, 5]
        for fold in folds:
            adapter = build_adapter(config, model_name, device, corpus, max(fold, 1))
            items = corpus.items if fold == 0 else corpus.select([fold])
            for ending_block, snr_db, order in itertools.product(
                config.model(model_name)["figure7_blocks"], corruption["snrs_db"], corruption["orders"]
            ):
                path = condition_path(root, model_name, fold, int(snr_db), int(order), int(ending_block))
                if resume and path.exists():
                    frame = pd.read_csv(path)
                else:
                    both = evaluate_condition(
                        config,
                        corpus,
                        adapter,
                        items,
                        fold,
                        path,
                        float(corruption["cutoff_hz"]),
                        int(order),
                        int(snr_db),
                        str(corruption["noise_reference"]),
                        bool(corruption["mask_inclusive"]),
                        int(ending_block),
                        config.seed,
                        str(corruption.get("filter_mode", "causal")),
                        ("proposed",),
                    )
                    frame = both.copy()
                    frame.to_csv(path, index=False)
                frames.append(frame[frame.method == "proposed"])
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    predictions = pd.concat(frames, ignore_index=True)
    summary = (
        predictions.groupby(["model", "snr_db", "ending_block"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )
    summary["accuracy"] *= 100.0
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "figure7.csv", index=False)
    plot_figure7(summary, root)
    return summary


def plot_figure7(summary: pd.DataFrame, root: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, model_name in zip(axes, ("passt", "msclap", "beats")):
        model_frame = summary[summary.model == model_name]
        for snr_db in sorted(model_frame.snr_db.unique(), reverse=True):
            curve = model_frame[model_frame.snr_db == snr_db].sort_values("ending_block")
            axis.plot(curve.ending_block, curve.accuracy, marker="o", label=f"SNR {snr_db} dB")
        axis.set_title(model_name.upper())
        axis.set_xlabel("Ending block")
        axis.set_ylabel("Accuracy (%)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(root / "figure7.pdf")
    figure.savefig(root / "figure7.png", dpi=240)
    plt.close(figure)


def load_beats_fbank_cache(config: ExperimentConfig, corpus: ESC50Corpus, device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    cache_path = config.project_root / "assets" / "cache" / "beats-clean-fbank.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        return cache["fbanks"], {filename: index for index, filename in enumerate(cache["filenames"])}
    adapter = build_adapter_without_fold(config, device)
    filenames: list[str] = []
    fbanks: list[torch.Tensor] = []
    batches = clean_batches(corpus, corpus.items, 16000, 5.0, False, 16)
    for batch_items, waveforms in tqdm(batches, desc="cache BEATs fbank"):
        fbanks.append(adapter.model.backbone.preprocess(waveforms).cpu())
        filenames.extend(item.filename for item in batch_items)
    cache = {"filenames": filenames, "fbanks": torch.cat(fbanks, dim=0)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache["fbanks"], {filename: index for index, filename in enumerate(filenames)}


def fbank_batch(items: list[ESC50Item], fbanks: torch.Tensor, indices: dict[str, int]) -> torch.Tensor:
    return fbanks[torch.tensor([indices[item.filename] for item in items], dtype=torch.long)]


def train_epoch(adapter: BEATsAdapter, items: list[ESC50Item], fbanks: torch.Tensor, indices: dict[str, int], optimizer, scheduler, batch_size: int, epoch_seed: int, gradient_clip: float) -> float:
    adapter.model.train()
    generator = np.random.default_rng(epoch_seed)
    shuffled = [items[index] for index in generator.permutation(len(items))]
    losses = []
    for offset in range(0, len(shuffled), batch_size):
        batch_items = shuffled[offset : offset + batch_size]
        labels = torch.tensor([item.target for item in batch_items], device=adapter.device)
        optimizer.zero_grad(set_to_none=True)
        logits = adapter.model.forward_fbank(fbank_batch(batch_items, fbanks, indices))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.inference_mode()
def clean_accuracy(adapter: BEATsAdapter, items: list[ESC50Item], fbanks: torch.Tensor, indices: dict[str, int], batch_size: int) -> float:
    adapter.model.eval()
    targets, predictions = [], []
    for offset in range(0, len(items), batch_size):
        batch_items = items[offset : offset + batch_size]
        logits = adapter.model.forward_fbank(fbank_batch(batch_items, fbanks, indices))
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(item.target for item in batch_items)
    return float(accuracy_score(targets, predictions) * 100.0)


def scheduler_for(optimizer, epochs: int, steps_per_epoch: int, warmup_fraction: float):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, schedule)


def train_beats(config: ExperimentConfig, device_name: str, resume: bool) -> None:
    set_deterministic(config.seed)
    device = torch.device(device_name)
    corpus = corpus_from_config(config)
    fbanks, fbank_indices = load_beats_fbank_cache(config, corpus, device)
    training = config.section("beats_training")
    history_root = config.resolve(training.get("output_root", "outputs/beats_training"))
    history_root.mkdir(parents=True, exist_ok=True)
    config.dump_run(
        history_root / "resolved_config.yaml",
        command="train-beats",
        device=device_name,
        resume=resume,
        seed=config.seed,
    )
    for test_fold in range(1, 6):
        checkpoint_path = beats_fold_checkpoint(config, test_fold)
        if resume and checkpoint_path.exists():
            continue
        validation_fold = test_fold % 5 + 1
        train_folds = [fold for fold in range(1, 6) if fold not in {test_fold, validation_fold}]
        train_items = corpus.select(train_folds)
        validation_items = corpus.select([validation_fold])
        selection_path = history_root / f"fold{test_fold}_selection.json"
        if resume and selection_path.exists():
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            print(f"BEATs fold={test_fold} reused_selection={selected}", flush=True)
        else:
            candidates = []
            for learning_rate, weight_decay in itertools.product(training["learning_rates"], training["weight_decays"]):
                print(
                    f"BEATs fold={test_fold} candidate lr={learning_rate} weight_decay={weight_decay}",
                    flush=True,
                )
                set_deterministic(config.seed)
                adapter = build_adapter_without_fold(config, device)
                optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
                steps = math.ceil(len(train_items) / int(training["batch_size"]))
                scheduler = scheduler_for(optimizer, int(training["max_epochs"]), steps, float(training["warmup_fraction"]))
                best_accuracy, best_epoch, stale = -1.0, 0, 0
                epoch_rows = []
                for epoch in range(1, int(training["max_epochs"]) + 1):
                    loss = train_epoch(
                        adapter,
                        train_items,
                        fbanks,
                        fbank_indices,
                        optimizer,
                        scheduler,
                        int(training["batch_size"]),
                        config.seed + epoch,
                        float(training["gradient_clip"]),
                    )
                    accuracy = clean_accuracy(
                        adapter,
                        validation_items,
                        fbanks,
                        fbank_indices,
                        int(training["batch_size"]),
                    )
                    epoch_rows.append({"epoch": epoch, "loss": loss, "validation_accuracy": accuracy})
                    print(
                        f"BEATs fold={test_fold} lr={learning_rate} wd={weight_decay} "
                        f"epoch={epoch} loss={loss:.5f} val_acc={accuracy:.2f}",
                        flush=True,
                    )
                    if accuracy > best_accuracy:
                        best_accuracy, best_epoch, stale = accuracy, epoch, 0
                    else:
                        stale += 1
                    if stale >= int(training["patience"]):
                        break
                candidate_id = f"fold{test_fold}_lr{learning_rate}_wd{weight_decay}"
                pd.DataFrame(epoch_rows).to_csv(history_root / f"{candidate_id}.csv", index=False)
                candidates.append(
                    {
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "best_accuracy": best_accuracy,
                        "best_epoch": best_epoch,
                    }
                )
                del adapter
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            selected = sorted(candidates, key=lambda item: (-item["best_accuracy"], item["learning_rate"], item["weight_decay"]))[0]
            selection_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print(f"BEATs fold={test_fold} selected={selected}", flush=True)
        all_training_items = corpus.select([fold for fold in range(1, 6) if fold != test_fold])
        set_deterministic(config.seed)
        adapter = build_adapter_without_fold(config, device)
        optimizer = torch.optim.AdamW(
            adapter.model.parameters(),
            lr=selected["learning_rate"],
            weight_decay=selected["weight_decay"],
        )
        steps = math.ceil(len(all_training_items) / int(training["batch_size"]))
        scheduler = scheduler_for(optimizer, int(selected["best_epoch"]), steps, float(training["warmup_fraction"]))
        for epoch in range(1, int(selected["best_epoch"]) + 1):
            loss = train_epoch(
                adapter,
                all_training_items,
                fbanks,
                fbank_indices,
                optimizer,
                scheduler,
                int(training["batch_size"]),
                config.seed + epoch,
                float(training["gradient_clip"]),
            )
            print(
                f"BEATs fold={test_fold} refit_epoch={epoch}/{selected['best_epoch']} loss={loss:.5f}",
                flush=True,
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": adapter.model.state_dict(), "selection": selected, "test_fold": test_fold}, checkpoint_path)
        selection_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print(f"BEATs fold={test_fold} checkpoint={checkpoint_path}", flush=True)


def build_adapter_without_fold(config: ExperimentConfig, device: torch.device) -> BEATsAdapter:
    vendor_root = config.resolve(config.values["vendor"]["root"])
    weight_root = config.resolve(config.values["weights"]["root"])
    return BEATsAdapter(
        vendor_root / "unilm" / "beats",
        weight_root / config.values["weights"]["beats"],
        None,
        device,
    )


def run_sensitivity(config: ExperimentConfig, device_name: str, resume: bool) -> dict[str, object]:
    config.dump_run(
        config.output_dir("aligned") / "sensitivity_resolved_config.yaml",
        command="run-sensitivity",
        track="aligned",
        device=device_name,
        resume=resume,
        seed=config.seed,
    )
    strict_metrics = config.output_dir("strict") / "report" / "metrics.json"
    if strict_metrics.exists():
        report = json.loads(strict_metrics.read_text(encoding="utf-8"))
        if report["checks"]["numerical_passed"]:
            return {"status": "strict_passed", "message": "Sensitivity scan skipped"}
    sensitivity = config.section("sensitivity")
    candidates = []
    candidate_root = config.output_dir("aligned") / "candidates"
    strict_table_root = config.output_dir("strict") / "table1"
    strict_corruption = config.section("corruption")
    conventional_roots: dict[tuple[float, str], Path] = {}
    for cutoff_hz, noise_reference, mask_inclusive in itertools.product(
        sensitivity["cutoffs_hz"], sensitivity["noise_references"], sensitivity["mask_inclusive"]
    ):
        identifier = f"cut{cutoff_hz}_{noise_reference}_{'inclusive' if mask_inclusive else 'exclusive'}"
        root = candidate_root / identifier
        matches_strict = (
            float(cutoff_hz) == float(strict_corruption["cutoff_hz"])
            and str(noise_reference) == str(strict_corruption["noise_reference"])
            and bool(mask_inclusive) == bool(strict_corruption["mask_inclusive"])
        )
        if matches_strict and (strict_table_root / "table1.csv").exists():
            if not (root / "table1.csv").exists():
                if root.exists():
                    shutil.rmtree(root)
                shutil.copytree(strict_table_root, root)
            config.dump_run(
                root / "resolved_config.yaml",
                command="run-table1",
                track="aligned",
                device=device_name,
                resume=resume,
                cutoff_hz=float(cutoff_hz),
                noise_reference=str(noise_reference),
                mask_inclusive=bool(mask_inclusive),
                seed=config.seed,
                reused_from=str(strict_table_root),
            )
            table = pd.read_csv(root / "table1.csv")
        else:
            conventional_source_root = conventional_roots.get((float(cutoff_hz), str(noise_reference)))
            table = run_table1(
                config,
                "aligned",
                device_name,
                resume,
                float(cutoff_hz),
                str(noise_reference),
                bool(mask_inclusive),
                config.seed,
                root,
                conventional_source_root,
            )
        conventional_roots[(float(cutoff_hz), str(noise_reference))] = root
        candidates.append(
            {
                "identifier": identifier,
                "cutoff_hz": cutoff_hz,
                "noise_reference": noise_reference,
                "mask_inclusive": mask_inclusive,
                "mae_pp": float(table.absolute_error_pp.mean()),
                "headline_error_pp": headline_error(table),
            }
        )
    preference = {1000: 0, 500: 1, 2000: 2}
    selected = sorted(
        candidates,
        key=lambda item: (
            item["mae_pp"],
            item["headline_error_pp"],
            preference[item["cutoff_hz"]],
            item["noise_reference"] != "filtered",
            not item["mask_inclusive"],
        ),
    )[0]
    aligned_root = config.output_dir("aligned") / "table1"
    if aligned_root.exists():
        shutil.rmtree(aligned_root)
    shutil.copytree(candidate_root / selected["identifier"], aligned_root)
    selection_path = config.output_dir("aligned") / "selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({"selected": selected, "candidates": candidates}, indent=2), encoding="utf-8")
    robustness = []
    for robustness_seed in sensitivity["robustness_seeds"]:
        robustness_root = config.output_dir("aligned") / "robustness" / f"seed{robustness_seed}"
        table = run_table1(
            config,
            "aligned",
            device_name,
            resume,
            float(selected["cutoff_hz"]),
            str(selected["noise_reference"]),
            bool(selected["mask_inclusive"]),
            int(robustness_seed),
            robustness_root,
        )
        robustness.append(
            {
                "seed": robustness_seed,
                "mae_pp": float(table.absolute_error_pp.mean()),
                "headline_error_pp": headline_error(table),
            }
        )
    (config.output_dir("aligned") / "robustness.json").write_text(
        json.dumps(robustness, indent=2),
        encoding="utf-8",
    )
    return selected


def headline_error(table: pd.DataFrame) -> float:
    selected = table[
        (table.model == "passt") & (table.snr_db == -15) & (table.order == 4)
    ].set_index("method")
    gain = float(selected.loc["proposed", "accuracy"] - selected.loc["conventional", "accuracy"])
    return abs(gain - 20.40)


def validate_clean(config: ExperimentConfig, device_name: str, resume: bool) -> pd.DataFrame:
    set_deterministic(config.seed)
    device = torch.device(device_name)
    corpus = corpus_from_config(config)
    output_path = config.output_dir("strict") / "clean" / "clean.csv"
    if resume and output_path.exists():
        return pd.read_csv(output_path)
    config.dump_run(
        output_path.parent / "resolved_config.yaml",
        command="validate-clean",
        track="strict",
        device=device_name,
        resume=resume,
        seed=config.seed,
    )
    rows = []
    for model_name in ("passt", "msclap", "beats"):
        folds = [0] if model_name == "msclap" else [1, 2, 3, 4, 5]
        for fold in folds:
            adapter = build_adapter(config, model_name, device, corpus, max(fold, 1))
            items = corpus.items if fold == 0 else corpus.select([fold])
            targets, predictions = [], []
            batches = clean_batches(
                corpus,
                items,
                adapter.sample_rate,
                adapter.input_seconds,
                adapter.repeat_short_audio,
                int(config.model(model_name)["batch_size"]),
            )
            for batch_items, waveforms in tqdm(batches, desc=f"clean {model_name} f{fold}", leave=False):
                logits = adapter.predict(
                    waveforms,
                    False,
                    float(config.values["corruption"]["cutoff_hz"]),
                    int(config.model(model_name)["default_ending_block"]),
                    bool(config.values["corruption"]["mask_inclusive"]),
                )
                targets.extend(item.target for item in batch_items)
                predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "samples": len(targets),
                    "accuracy": accuracy_score(targets, predictions) * 100.0,
                }
            )
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
