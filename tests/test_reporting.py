import pandas as pd

from trainingless_adaptation.constants import PAPER_FIGURE7_RANKS, PAPER_TABLE_I
from trainingless_adaptation.reporting import figure_metrics, render_markdown, table_metrics


def test_reference_table_passes_exactly() -> None:
    rows = []
    for (model, method, snr_db, order), accuracy in PAPER_TABLE_I.items():
        rows.append(
            {
                "model": model,
                "method": method,
                "snr_db": snr_db,
                "order": order,
                "accuracy": accuracy,
                "paper_accuracy": accuracy,
                "absolute_error_pp": 0.0,
            }
        )
    metrics = table_metrics(pd.DataFrame(rows))
    assert metrics["table_mae_pp"] == 0.0
    assert metrics["headline_gain_error_pp"] < 1e-9
    assert metrics["gain_sign_agreement"] == 1.0
    assert metrics["gain_spearman"] > 1.0 - 1e-12


def test_reference_figure_passes_rank_check() -> None:
    rows = []
    for (model, snr_db), ranks in PAPER_FIGURE7_RANKS.items():
        for ending_block, accuracy in enumerate(ranks):
            rows.append(
                {
                    "model": model,
                    "snr_db": snr_db,
                    "ending_block": ending_block,
                    "accuracy": accuracy,
                }
            )
    metrics = figure_metrics(pd.DataFrame(rows))
    assert metrics["figure_spearman_mean"] > 1.0 - 1e-12


def test_chinese_report_has_no_replacement_characters() -> None:
    payload = {
        "track": "strict",
        "verdict": "strict_replication_passed",
        "metrics": {
            "table_mae_pp": 0.0,
            "headline_gain_pp": 20.4,
            "gain_sign_agreement": 1.0,
            "gain_spearman": 1.0,
        },
        "checks": {"passed": True, "numerical_passed": True, "diagnostics_passed": True},
    }
    report = render_markdown(payload, pd.DataFrame(), None, None, chinese=True)
    assert "\ufffd" not in report
    assert "\u590d\u73b0\u62a5\u544a" in report
