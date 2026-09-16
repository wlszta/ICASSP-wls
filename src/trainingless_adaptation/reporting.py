from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import ExperimentConfig
from .constants import HEADLINE_GAIN_PP, PAPER_FIGURE7_BEST_BLOCK, PAPER_FIGURE7_RANKS, PAPER_TABLE_I


def paper_gains() -> dict[tuple[str, int, int], float]:
    return {
        (model, snr_db, order): PAPER_TABLE_I[(model, "proposed", snr_db, order)]
        - PAPER_TABLE_I[(model, "conventional", snr_db, order)]
        for model in ("passt", "msclap", "beats")
        for snr_db in (-5, -10, -15)
        for order in (1, 2, 3, 4)
    }


def table_metrics(table: pd.DataFrame) -> dict[str, float]:
    indexed = table.set_index(["model", "method", "snr_db", "order"])
    reproduced_gains = {}
    for model in ("passt", "msclap", "beats"):
        for snr_db in (-5, -10, -15):
            for order in (1, 2, 3, 4):
                proposed = float(indexed.loc[(model, "proposed", snr_db, order), "accuracy"])
                conventional = float(indexed.loc[(model, "conventional", snr_db, order), "accuracy"])
                reproduced_gains[(model, snr_db, order)] = proposed - conventional
    reference_gains = paper_gains()
    keys = sorted(reference_gains)
    sign_agreement = np.mean(
        [np.sign(reproduced_gains[key]) == np.sign(reference_gains[key]) for key in keys]
    )
    gain_correlation = spearmanr(
        [reference_gains[key] for key in keys],
        [reproduced_gains[key] for key in keys],
    ).statistic
    headline_gain = reproduced_gains[("passt", -15, 4)]
    return {
        "table_mae_pp": float(table.absolute_error_pp.mean()),
        "table_max_error_pp": float(table.absolute_error_pp.max()),
        "headline_gain_pp": float(headline_gain),
        "headline_gain_error_pp": abs(float(headline_gain) - HEADLINE_GAIN_PP),
        "gain_sign_agreement": float(sign_agreement),
        "gain_spearman": float(gain_correlation),
    }


def figure_metrics(figure: pd.DataFrame) -> dict[str, object]:
    curve_correlations: dict[str, float] = {}
    for (model, snr_db), reference in PAPER_FIGURE7_RANKS.items():
        curve = figure[(figure.model == model) & (figure.snr_db == snr_db)].sort_values("ending_block")
        if len(curve) != len(reference):
            raise ValueError(f"Incomplete Fig. 7 curve for {model} at {snr_db} dB")
        correlation = spearmanr(reference, curve.accuracy.to_numpy()).statistic
        curve_correlations[f"{model}_{snr_db}"] = float(correlation)

    best_blocks: dict[str, int] = {}
    ending_errors: dict[str, int] = {}
    for model, expected in PAPER_FIGURE7_BEST_BLOCK.items():
        block_means = figure[figure.model == model].groupby("ending_block").accuracy.mean()
        reproduced = int(block_means.idxmax())
        best_blocks[model] = reproduced
        ending_errors[model] = abs(reproduced - expected)
    return {
        "figure_curve_spearman": curve_correlations,
        "figure_spearman_mean": float(np.mean(list(curve_correlations.values()))),
        "figure_best_blocks": best_blocks,
        "figure_ending_block_errors": ending_errors,
        "figure_ending_block_max_error": int(max(ending_errors.values())),
    }


def clean_metrics(config: ExperimentConfig, clean: pd.DataFrame) -> dict[str, object]:
    targets = config.section("clean_acceptance")
    passt = clean[clean.model == "passt"].sort_values("fold")
    passt_expected = np.asarray(targets["passt_expected"], dtype=np.float64)
    passt_errors = np.abs(passt.accuracy.to_numpy(dtype=np.float64) - passt_expected)
    msclap_accuracy = float(clean[clean.model == "msclap"].accuracy.mean())
    beats_accuracy = float(clean[clean.model == "beats"].accuracy.mean())
    return {
        "clean_passt_fold_errors_pp": passt_errors.tolist(),
        "clean_passt_max_error_pp": float(passt_errors.max()),
        "clean_msclap_accuracy": msclap_accuracy,
        "clean_msclap_error_pp": abs(msclap_accuracy - float(targets["msclap_expected"])),
        "clean_beats_accuracy": beats_accuracy,
        "clean_beats_error_pp": abs(beats_accuracy - float(targets["beats_expected"])),
    }


def validation_metrics(config: ExperimentConfig) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric_name, filename in (
        ("dsp_validation", "dsp-validation.json"),
        ("adaptation_integration", "adaptation-integration.json"),
        ("model_smoke", "model-smoke.json"),
        ("device_consistency", "device-consistency.json"),
    ):
        scoped_path = config.manifest_dir / filename
        legacy_path = config.project_root / "manifests" / filename
        paths = [scoped_path] if scoped_path == legacy_path else [scoped_path, legacy_path]
        for path in paths:
            if path.exists():
                records = json.loads(path.read_text(encoding="utf-8"))
                result[f"{metric_name}_passed"] = bool(records) and all(record["passed"] for record in records)
                result[f"{metric_name}_records"] = records
                break
    return result


def acceptance(config: ExperimentConfig, metrics: dict[str, object]) -> dict[str, bool]:
    thresholds = config.section("acceptance")
    clean_thresholds = config.section("clean_acceptance")
    checks = {
        "table_mae": metrics["table_mae_pp"] <= float(thresholds["table_mae_pp"]),
        "headline_gain": metrics["headline_gain_error_pp"] <= float(thresholds["headline_gain_error_pp"]),
        "gain_sign": metrics["gain_sign_agreement"] >= float(thresholds["gain_sign_agreement"]),
        "gain_spearman": metrics["gain_spearman"] >= float(thresholds["gain_spearman"]),
    }
    if "figure_spearman_mean" in metrics:
        checks["figure_spearman"] = metrics["figure_spearman_mean"] >= float(thresholds["figure_spearman"])
        checks["ending_block"] = metrics["figure_ending_block_max_error"] <= int(thresholds["ending_block_error"])
    numerical_keys = list(checks)
    if "clean_passt_max_error_pp" in metrics:
        checks["clean_passt"] = metrics["clean_passt_max_error_pp"] <= float(clean_thresholds["passt_fold_tolerance_pp"])
        checks["clean_msclap"] = metrics["clean_msclap_error_pp"] <= float(clean_thresholds["msclap_tolerance_pp"])
        checks["clean_beats"] = metrics["clean_beats_error_pp"] <= float(clean_thresholds["beats_tolerance_pp"])
    if "adaptation_integration_passed" in metrics:
        checks["adaptation_integration"] = bool(metrics["adaptation_integration_passed"])
    if "dsp_validation_passed" in metrics:
        checks["dsp_validation"] = bool(metrics["dsp_validation_passed"])
    if "model_smoke_passed" in metrics:
        checks["model_smoke"] = bool(metrics["model_smoke_passed"])
    if "device_consistency_passed" in metrics:
        checks["device_consistency"] = bool(metrics["device_consistency_passed"])
    diagnostic_keys = [key for key in checks if key not in numerical_keys]
    checks["numerical_passed"] = all(checks[key] for key in numerical_keys)
    checks["diagnostics_passed"] = all(checks[key] for key in diagnostic_keys)
    checks["passed"] = checks["numerical_passed"] and checks["diagnostics_passed"]
    return checks


def latex_table(table: pd.DataFrame) -> str:
    ordered = table.sort_values(["model", "method", "snr_db", "order"])
    return ordered[["model", "method", "snr_db", "order", "accuracy", "paper_accuracy", "absolute_error_pp"]].to_latex(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    )


def build_report(config: ExperimentConfig, track: str) -> dict[str, object]:
    root = config.output_dir(track)
    table_path = root / "table1" / "table1.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Table I results not found: {table_path}")
    table = pd.read_csv(table_path)
    metrics: dict[str, object] = table_metrics(table)
    clean_path = config.output_dir("strict") / "clean" / "clean.csv"
    clean = pd.read_csv(clean_path) if clean_path.exists() else None
    if clean is not None:
        metrics.update(clean_metrics(config, clean))
    metrics.update(validation_metrics(config))
    figure_path = root / "figure7" / "figure7.csv"
    figure = pd.read_csv(figure_path) if figure_path.exists() else None
    if figure is not None:
        metrics.update(figure_metrics(figure))
    checks = acceptance(config, metrics)
    verdict = (
        "strict_replication_passed"
        if track == "strict" and checks["passed"]
        else "transparent_calibrated_replication_passed"
        if track == "aligned" and checks["passed"]
        else "diagnostic_gates_not_met"
        if checks["numerical_passed"]
        else "numerical_gates_not_met"
    )
    payload: dict[str, object] = {
        "track": track,
        "verdict": verdict,
        "metrics": metrics,
        "checks": checks,
        "clean_available": clean is not None,
        "figure7_available": figure is not None,
        "figure7_reference": "digitized at 180 DPI from the supplied paper PDF",
    }
    selection_path = config.output_dir("aligned") / "selection.json"
    robustness_path = config.output_dir("aligned") / "robustness.json"
    if track == "aligned" and selection_path.exists():
        payload["alignment"] = json.loads(selection_path.read_text(encoding="utf-8"))
    if track == "aligned" and robustness_path.exists():
        payload["robustness"] = json.loads(robustness_path.read_text(encoding="utf-8"))
    report_root = root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    config.dump_run(report_root / "resolved_config.yaml", command="build-report", track=track)
    (report_root / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (report_root / "table1.tex").write_text(latex_table(table), encoding="utf-8")
    (report_root / "report_en.md").write_text(render_markdown(payload, table, clean, figure, chinese=False), encoding="utf-8")
    (report_root / "report_zh.md").write_text(render_markdown(payload, table, clean, figure, chinese=True), encoding="utf-8")
    return payload


def render_markdown(
    payload: dict[str, object],
    table: pd.DataFrame,
    clean: pd.DataFrame | None,
    figure: pd.DataFrame | None,
    chinese: bool,
) -> str:
    metrics = payload["metrics"]
    checks = payload["checks"]
    if chinese:
        heading = "# Trainingless Adaptation \u590d\u73b0\u62a5\u544a"
        summary = "## \u9a8c\u6536\u7ed3\u679c"
        clean_heading = "## \u5e72\u51c0\u57fa\u7ebf"
        controls_heading = "## \u53ef\u91cd\u590d\u6027\u63a7\u5236"
        assumptions_heading = "## \u53e3\u5f84\u4e0e\u9650\u5236"
        passport = [
            "## \u6750\u6599\u62a4\u7167",
            "",
            "- \u7c7b\u578b\uff1a\u53ef\u91cd\u590d\u6027\u5b9e\u9a8c\u62a5\u544a",
            f"- \u9a8c\u8bc1\u72b6\u6001\uff1a`{'VERIFIED' if checks['passed'] else 'ANALYZED'}`",
            f"- \u8f68\u9053\uff1a`{payload['track']}`",
            "- \u6570\u636e\uff1aESC-50 \u5b98\u65b9\u4e94\u6298",
            "",
        ]
        labels = {
            "track": "\u8f68\u9053",
            "verdict": "\u7ed3\u8bba",
            "table_mae": "Table I \u5e73\u5747\u7edd\u5bf9\u8bef\u5dee",
            "headline": "\u6807\u5fd7\u6027\u589e\u76ca",
            "sign": "\u589e\u76ca\u7b26\u53f7\u4e00\u81f4\u7387",
            "gain_spearman": "\u589e\u76ca Spearman \u76f8\u5173",
            "numerical": "\u5168\u90e8\u6570\u503c\u95e8\u69db\u901a\u8fc7",
            "diagnostics": "\u5168\u90e8\u8bca\u65ad\u901a\u8fc7",
        }
        figure_artifact = "\u5df2\u751f\u6210 CSV/PDF/PNG \u4ea7\u7269\u3002"
        not_run = "\u672a\u8fd0\u884c\u3002"
        assumptions = [
            "- \u4e25\u683c\u8f68\u9ed8\u8ba4\uff1a1000 Hz \u622a\u6b62\u9891\u7387\u3001\u6ee4\u6ce2\u540e\u4fe1\u53f7\u529f\u7387\u3001\u5305\u542b\u8fb9\u754c\u3001seed 0\u3002",
            "- \u8bba\u6587\u672a\u516c\u5f00\u622a\u6b62\u9891\u7387\u3001\u566a\u58f0\u53c2\u8003\u3001mask \u8fb9\u754c\u3001\u566a\u58f0 seed \u548c BEATs \u5fae\u8c03\u7ec6\u8282\u3002",
            "- \u5bf9\u9f50\u8f68\u59cb\u7ec8\u4e0e\u4e25\u683c\u8f68\u5206\u5f00\uff0c\u4e0d\u8986\u76d6\u4e25\u683c\u7ed3\u679c\u3002",
        ]
    else:
        heading = "# Trainingless Adaptation Reproduction Report"
        summary = "## Acceptance"
        clean_heading = "## Clean Baselines"
        controls_heading = "## Reproducibility Controls"
        assumptions_heading = "## Protocol and Limitations"
        passport = [
            "## Material Passport",
            "",
            "- Type: reproducibility experiment report",
            f"- Verification Status: `{'VERIFIED' if checks['passed'] else 'ANALYZED'}`",
            f"- Track: `{payload['track']}`",
            "- Data: official ESC-50 five-fold split",
            "",
        ]
        labels = {
            "track": "Track",
            "verdict": "Verdict",
            "table_mae": "Table I mean absolute error",
            "headline": "Headline gain",
            "sign": "Gain sign agreement",
            "gain_spearman": "Gain Spearman",
            "numerical": "All numerical gates passed",
            "diagnostics": "All diagnostics passed",
        }
        figure_artifact = "Generated CSV/PDF/PNG artifacts are available."
        not_run = "Not run."
        assumptions = [
            "- Strict defaults: 1000 Hz cutoff, filtered-signal noise power, inclusive mask boundary, seed 0.",
            "- The paper omits cutoff, noise reference, mask boundary, noise seed, and BEATs fine-tuning details.",
            "- Aligned results remain separate from strict results and never overwrite them.",
        ]
    figure_lines = []
    if "figure_spearman_mean" in metrics:
        figure_lines = [
            f"- Fig. 7 mean curve Spearman: {metrics['figure_spearman_mean']:.3f}",
            f"- Fig. 7 maximum ending-block error: {metrics['figure_ending_block_max_error']}",
        ]
    diagnostic_lines = []
    if "clean_passt_max_error_pp" in metrics:
        diagnostic_lines.extend(
            [
                f"- PaSST clean maximum fold error: {metrics['clean_passt_max_error_pp']:.3f} pp",
                f"- MS-CLAP clean accuracy: {metrics['clean_msclap_accuracy']:.3f}%",
                f"- BEATs clean five-fold accuracy: {metrics['clean_beats_accuracy']:.3f}%",
            ]
        )
    if "adaptation_integration_passed" in metrics:
        diagnostic_lines.append(f"- Adaptation integration passed: {metrics['adaptation_integration_passed']}")
    if "dsp_validation_passed" in metrics:
        diagnostic_lines.append(f"- DSP validation passed: {metrics['dsp_validation_passed']}")
    if "model_smoke_passed" in metrics:
        diagnostic_lines.append(f"- Parameter immutability smoke test passed: {metrics['model_smoke_passed']}")
    if "device_consistency_passed" in metrics:
        diagnostic_lines.append(f"- CPU/GPU consistency passed: {metrics['device_consistency_passed']}")
    lines = [
        heading,
        "",
        *passport,
        summary,
        "",
        f"- {labels['track']}: `{payload['track']}`",
        f"- {labels['verdict']}: `{payload['verdict']}`",
        f"- {labels['table_mae']}: {metrics['table_mae_pp']:.3f} pp",
        f"- {labels['headline']}: {metrics['headline_gain_pp']:.3f} pp",
        f"- {labels['sign']}: {metrics['gain_sign_agreement']:.3f}",
        f"- {labels['gain_spearman']}: {metrics['gain_spearman']:.3f}",
        *figure_lines,
        f"- {labels['numerical']}: {checks['numerical_passed']}",
        f"- {labels['diagnostics']}: {checks['diagnostics_passed']}",
        "",
        "## Table I",
        "",
        table.to_markdown(index=False, floatfmt=".2f"),
        "",
        clean_heading,
        "",
        clean.to_markdown(index=False, floatfmt=".2f") if clean is not None else not_run,
        "",
        "## Fig. 7",
        "",
        figure_artifact if figure is not None else not_run,
        "",
        controls_heading,
        "",
        *diagnostic_lines,
        "",
        assumptions_heading,
        "",
        *assumptions,
        "",
    ]
    return "\n".join(lines)
