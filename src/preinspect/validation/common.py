"""Shared helpers for the validation studies: inputs, outcomes and route metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from preinspect.service.pipeline import InspectionPipeline, UnitInput, unit_from_dataset

OUTCOMES = ["PASS", "REWORK", "SCRAP"]
ROUTE_FOR = {"PASS": "PROCEED", "REWORK": "EXTENDED_CLEAN", "SCRAP": "QUARANTINE"}


def stratified(units: pd.DataFrame, truth: pd.Series, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` units with the outcome mix balanced as far as the data allows."""
    rng = np.random.default_rng(seed)
    groups = {
        k: list(rng.permutation(g.index))
        for k, g in units.groupby(truth.loc[units["unit_id"]].values)
    }
    chosen: list[int] = []
    while len(chosen) < n and any(groups.values()):
        for idx in groups.values():
            if idx and len(chosen) < n:
                chosen.append(idx.pop())
    return units.loc[sorted(chosen)]


def inputs_for(ddir: Path, units: pd.DataFrame, signals: np.ndarray) -> list[UnitInput]:
    """Station inputs for dataset units (original images)."""
    return [unit_from_dataset(ddir, row, signals) for _, row in units.iterrows()]


def score_results(results: list[dict], truth: list[str]) -> dict[str, float]:
    """Outcome accuracy and routing safety for a batch of inspection results."""
    truth = np.asarray(truth)
    routes = np.array([r["route"] for r in results])
    pred = np.array([max(r["probabilities"], key=r["probabilities"].get) for r in results])
    serviceable = truth != "SCRAP"
    wrongful = (routes == "QUARANTINE") & serviceable
    missed = np.isin(routes, ["PROCEED", "EXTENDED_CLEAN"]) & ~serviceable
    auto = routes != "REFER"
    expected = np.array([ROUTE_FOR[t] for t in truth])
    return {
        "macro_f1": float(f1_score(truth, pred, labels=OUTCOMES, average="macro", zero_division=0)),
        "scrap_recall": float((pred[~serviceable] == "SCRAP").mean())
        if (~serviceable).any()
        else float("nan"),
        "unsafe_rate": float((wrongful | missed).mean()),
        "wrongful_scrap_rate": float(wrongful[serviceable].mean())
        if serviceable.any()
        else float("nan"),
        "scrap_capture_rate": float((~missed[~serviceable]).mean())
        if (~serviceable).any()
        else float("nan"),
        "refer_share": float((routes == "REFER").mean()),
        "anomaly_refer_share": float(np.mean([r["reason"] == "anomaly" for r in results])),
        "auto_route_accuracy": float((routes[auto] == expected[auto]).mean())
        if auto.any()
        else float("nan"),
        "n": int(len(truth)),
    }


def inspect_all(pipe: InspectionPipeline, inputs: list[UnitInput], chunk: int = 32) -> list[dict]:
    """Inspect many units in chunks to bound memory use."""
    out: list[dict] = []
    for s in range(0, len(inputs), chunk):
        out += pipe.inspect_batch(inputs[s : s + chunk])
    return out
