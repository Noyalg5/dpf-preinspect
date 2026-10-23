"""Stage ``evaluate``: calibrate and score every model, build the headline tables."""

from __future__ import annotations

import pandas as pd

from preinspect.config import Config, load_aux
from preinspect.evaluate.decision_step import (
    add_rationales,
    cost_curve_figure,
    decide,
    load_halves,
    test_report,
)
from preinspect.evaluate.explain_step import explain_step
from preinspect.evaluate.figures import reliability_figure
from preinspect.evaluate.scoring import evaluate_predictions
from preinspect.evaluate.tables import ablation_table, flatten, headline_table
from preinspect.runs import Run, open_run
from preinspect.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def run(cfg: Config, run_name: str | None = None) -> Run:
    """Evaluate the run's predictions and write tables, figures and metrics."""
    run_ = open_run(cfg, run_name)
    setup_logging(log_file=run_.path / "logs" / "evaluate.log")
    results = evaluate_predictions(cfg, run_.path)
    headline = headline_table(cfg, results)
    ablation = ablation_table(cfg, results)
    flatten(headline).to_csv(run_.sub("tables", "headline.csv"), index=False)
    flatten(ablation).to_csv(run_.sub("tables", "ablation.csv"), index=False)
    for row in headline:
        log.info(
            "%-38s macro-F1 %.3f  SCRAP recall %.3f  dP MAE %.3f  ECE %.3f",
            row["label"],
            row["macro_f1"]["mean"],
            row["scrap_recall"]["mean"],
            row["dp_mae"]["mean"],
            row["ece"]["mean"],
        )
    panels = {}
    for name in ("learned_flow_meta", "e2e_all_s0"):
        raw = run_.path / "predictions" / f"{name}.parquet"
        if raw.exists():
            from preinspect.evaluate.scoring import with_truth

            raw_df = with_truth(cfg, pd.read_parquet(raw))
            cal_df = pd.read_parquet(run_.path / "predictions_calibrated" / f"{name}.parquet")
            panels[name] = (raw_df[raw_df["split"] == "test"], cal_df[cal_df["split"] == "test"])
    if panels:
        reliability_figure(
            panels, cfg["evaluation"]["ece_bins"], run_.sub("figures", "reliability.png")
        )
    decisions = decision_step(cfg, run_, headline)
    run_.write_metrics(
        "evaluate",
        {"models": results, "headline": headline, "ablation": ablation, "decisions": decisions},
    )
    if (run_.path / "models" / "fusion" / "e2e_all_s0.pt").exists():
        run_.write_metrics("explain", explain_step(cfg, run_.path))
    return run_


def decision_step(cfg: Config, run_: Run, headline: list[dict]) -> dict:
    """Route test units with each headline model; full decision table for the deployed one."""
    features = pd.read_parquet(run_.path / "features" / "features.parquet")
    anomaly = features.set_index("unit_id")["anom_unexplained"]
    fit_ids, conf_ids = load_halves(run_.path)
    out = {}
    for row in headline:
        name = row["model"]
        path = run_.path / "predictions_calibrated" / f"{name}.parquet"
        if not path.exists():
            path = run_.path / "predictions_calibrated" / f"{name}_s0.parquet"
        d = decide(cfg, pd.read_parquet(path), anomaly, fit_ids, conf_ids)
        report = test_report(cfg, d)
        row["wrongful_scrap_op"] = {
            "mean": report["rates"]["wrongful_scrap_rate"],
            "std": None,
            "n": 1,
        }
        row["scrap_capture_op"] = {
            "mean": report["rates"]["scrap_capture_rate"],
            "std": None,
            "n": 1,
        }
        row["refer_share_op"] = {"mean": report["rates"]["share_REFER"], "std": None, "n": 1}
        row["saving_per_unit"] = {
            "mean": report["cost_per_unit"]["saving"],
            "std": None,
            "n": 1,
        }
        out[row["model"]] = report
        if row["model"] == cfg["deploy"]["model"]:
            table = d["table"]
            table["rationale"] = add_rationales(cfg, table, features)
            table.to_parquet(run_.sub("decisions", "deployed.parquet"), index=False)
            d["search"].to_csv(run_.sub("tables", "threshold_search.csv"), index=False)
            cap = load_aux("costs")["policy"]["wrongful_scrap_cap"]["value"]
            cost_curve_figure(d["search"], d["policy"], cap, run_.sub("figures", "cost_curve.png"))
            run_.write_json(
                "models/policy.json",
                {
                    "model": row["model"],
                    **d["policy"].as_dict(),
                    "qhat": {str(k): v for k, v in d["conformal"].qhat.items()},
                    "interval_q": {str(k): v for k, v in d["intervals"].q.items()},
                },
            )
    flatten(headline).to_csv(run_.sub("tables", "headline.csv"), index=False)
    return out
