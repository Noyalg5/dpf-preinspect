"""Headline and ablation tables built from evaluation results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from preinspect.evaluate.scoring import seed_summary


def wrongful_scrap_rate(confusion: list[list[int]]) -> float:
    """Share of truly serviceable units (PASS or REWORK) predicted SCRAP."""
    cm = np.asarray(confusion)
    serviceable = cm[:2].sum()
    return float(cm[:2, 2].sum() / serviceable) if serviceable else float("nan")


def _row(results: dict[str, Any], model: str) -> dict[str, Any]:
    keys = {
        "macro_f1": "truth.macro_f1",
        "scrap_recall": "truth.recall_SCRAP",
        "auc_scrap": "truth.auc_SCRAP",
        "dp_mae": "dp.mae",
        "ece": "ece",
    }
    row = {k: seed_summary(results, model, path) for k, path in keys.items()}
    first = next(k for k in results if k == model or k.startswith(f"{model}_s"))
    row["wrongful_scrap_argmax"] = {
        "mean": wrongful_scrap_rate(results[first]["calibrated"]["truth"]["confusion"]),
        "std": None,
        "n": 1,
    }
    row["ece_raw"] = {"mean": results[first]["raw"]["ece"], "std": None, "n": 1}
    return row


def headline_table(cfg: dict[str, Any], results: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per configured headline model (seeds aggregated as mean and std)."""
    rows = []
    for spec in cfg["headline"]:
        if any(k == spec["model"] or k.startswith(f"{spec['model']}_s") for k in results):
            rows.append(
                {"label": spec["label"], "model": spec["model"], **_row(results, spec["model"])}
            )
    return rows


def ablation_table(cfg: dict[str, Any], results: dict[str, Any]) -> list[dict[str, Any]]:
    """Concept-route modality ablation: each group alone, in pairs and all together."""
    return [
        {"groups": groups, "model": model, **_row(results, model)}
        for groups, model in cfg["ablation"].items()
        if model in results
    ]


def flatten(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten ``{mean, std}`` cells for CSV export."""
    flat = []
    for r in rows:
        out = {}
        for k, v in r.items():
            if isinstance(v, dict) and "mean" in v:
                out[k] = v["mean"]
                out[f"{k}_std"] = v["std"]
            else:
                out[k] = v
        flat.append(out)
    return pd.DataFrame(flat)
