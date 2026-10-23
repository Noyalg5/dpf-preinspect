"""Measurement system analysis campaign.

(i) Attribute agreement: the system, re-imaged twice with acquisition
jitter, is treated as a fourth appraiser alongside the three simulated
inspectors, against the true outcome as the standard.

(ii) Gauge R&R on the predicted post-clean ΔP ratio: units x stations x
trials, where each station has its own camera profile and flow-bench bias,
and each trial re-seats the unit and re-measures its flow curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preinspect.service.pipeline import InspectionPipeline, UnitInput, unit_from_dataset
from preinspect.utils.seeding import make_rng
from preinspect.validation.common import inspect_all
from preinspect.validation.msa_stats import attribute_agreement, gauge_rr
from preinspect.validation.reimage import remeasure_flow, rerender

VIEWS = ("inlet", "outlet")


def reimaged_inputs(
    cfg: dict,
    ddir: Path,
    units: pd.DataFrame,
    signals: np.ndarray,
    scenes: dict,
    latent: pd.DataFrame,
    key: tuple,
    camera: str | None = None,
    flow_bias: float = 0.0,
) -> list[UnitInput]:
    """Station inputs for a repeat visit: new images and a new flow measurement."""
    jitter = cfg["campaign"]["msa"]["jitter"]
    out = []
    for _, row in units.iterrows():
        uid = row["unit_id"]
        rng = make_rng(cfg["seed"], "reimage", uid, *key)
        overrides = {"camera": camera} if camera else {}
        images = {v: rerender(cfg, scenes[uid], v, rng, jitter, **overrides) for v in VIEWS}
        unit = unit_from_dataset(ddir, row, signals, images)
        unit.curve = remeasure_flow(cfg, row.to_dict(), latent.loc[uid].to_dict(), rng, flow_bias)
        out.append(unit)
    return out


def attribute_study(
    cfg: dict,
    pipe: InspectionPipeline,
    ddir: Path,
    units: pd.DataFrame,
    signals: np.ndarray,
    scenes: dict,
    latent: pd.DataFrame,
    inspectors: pd.DataFrame,
) -> dict[str, Any]:
    """System versus simulated inspectors on the MSA units."""
    msa = units[units["msa"]]
    rows, routes = [], {}
    for trial in (1, 2):
        results = inspect_all(
            pipe, reimaged_inputs(cfg, ddir, msa, signals, scenes, latent, ("attribute", trial))
        )
        for r in results:
            call = max(r["probabilities"], key=r["probabilities"].get)
            rows.append(
                {"unit_id": r["unit_id"], "appraiser": "System", "trial": trial, "call": call}
            )
            routes.setdefault(r["unit_id"], []).append(r["route"])
    calls = pd.concat(
        [
            inspectors[inspectors["unit_id"].isin(msa["unit_id"])].rename(
                columns={"inspector": "appraiser"}
            )[["unit_id", "appraiser", "trial", "call"]],
            pd.DataFrame(rows),
        ],
        ignore_index=True,
    )
    standard = latent.loc[msa["unit_id"], "outcome_true"]
    stats = attribute_agreement(calls, standard)
    stats["route_repeatability"] = float(np.mean([len(set(v)) == 1 for v in routes.values()]))
    return stats


def grr_study(
    cfg: dict,
    pipe: InspectionPipeline,
    ddir: Path,
    units: pd.DataFrame,
    signals: np.ndarray,
    scenes: dict,
    latent: pd.DataFrame,
) -> dict[str, Any]:
    """Crossed gauge R&R on the predicted ΔP ratio."""
    m = cfg["campaign"]["msa"]
    test = units[units["split"] == "test"].copy()
    test["true_dp"] = latent.loc[test["unit_id"], "dp_ratio_post_clean_true"].to_numpy()
    test = test.sort_values("true_dp")
    picks = np.linspace(0, len(test) - 1, m["grr_units"]).round().astype(int)
    parts = test.iloc[picks]
    rows = []
    for station, spec in m["stations"].items():
        for trial in range(1, m["grr_trials"] + 1):
            inputs = reimaged_inputs(
                cfg,
                ddir,
                parts,
                signals,
                scenes,
                latent,
                ("grr", station, trial),
                spec["camera"],
                spec["flow_bias"],
            )
            for r in inspect_all(pipe, inputs):
                rows.append(
                    {
                        "part": r["unit_id"],
                        "operator": station,
                        "trial": trial,
                        "value": r["dp_ratio"]["prediction"],
                    }
                )
    data = pd.DataFrame(rows)
    return {**gauge_rr(data), "data": data.to_dict("records")}


def unit_inputs(ddir: Path, units: pd.DataFrame, signals: np.ndarray) -> list[UnitInput]:
    """Original-image inputs (for studies that do not re-image)."""
    return [unit_from_dataset(ddir, row, signals) for _, row in units.iterrows()]
