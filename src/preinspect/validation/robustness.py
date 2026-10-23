"""Robustness: performance and routing as image quality degrades.

Each corruption is applied to both faces of a fixed sample of test units at
five severities. A well-behaved system should lose accuracy gradually and,
as inputs become unreliable, send more units to REFER rather than make
confident wrong automatic decisions.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preinspect.report.style import SERIES, save_figure
from preinspect.service.pipeline import InspectionPipeline, UnitInput
from preinspect.utils.seeding import make_rng
from preinspect.validation.common import inspect_all, score_results
from preinspect.validation.corruptions import corrupt


def run_robustness(
    cfg: dict, pipe: InspectionPipeline, inputs: list[UnitInput], truth: list[str]
) -> pd.DataFrame:
    """Scores for the clean inputs and every (corruption, severity) pair."""
    rows = [
        {
            "kind": "none",
            "severity": 0,
            "level": 0.0,
            **score_results(inspect_all(pipe, inputs), truth),
        }
    ]
    rcfg = cfg["campaign"]["robustness"]
    for kind in rcfg["include"]:
        levels = rcfg["kinds"][kind]
        for sev, level in enumerate(levels, start=1):
            rng = make_rng(cfg["seed"], "robustness", kind, sev)
            corrupted = [
                replace(
                    u, images={v: corrupt(img, kind, level, rng) for v, img in u.images.items()}
                )
                for u in inputs
            ]
            rows.append(
                {
                    "kind": kind,
                    "severity": sev,
                    "level": float(level),
                    **score_results(inspect_all(pipe, corrupted), truth),
                }
            )
    return pd.DataFrame(rows)


def degradation_figure(df: pd.DataFrame, path: Path) -> Path:
    """Small multiples: macro-F1, REFER share and unsafe-route rate against severity."""
    kinds = [k for k in df["kind"].unique() if k != "none"]
    clean = df[df["kind"] == "none"].iloc[0]
    cols = 4
    rows = int(np.ceil(len(kinds) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.0 * cols, 2.6 * rows), sharey=True, squeeze=False
    )
    for ax in axes.ravel()[len(kinds) :]:
        ax.axis("off")
    for ax, kind in zip(axes.ravel(), kinds, strict=False):
        sub = pd.concat([pd.DataFrame([clean]), df[df["kind"] == kind]]).sort_values("severity")
        ax.plot(sub["severity"], sub["macro_f1"], color=SERIES[0], marker="o", label="macro-F1")
        ax.plot(
            sub["severity"], sub["refer_share"], color=SERIES[1], marker="o", label="REFER share"
        )
        ax.plot(
            sub["severity"],
            sub["unsafe_rate"],
            color=SERIES[2],
            marker="o",
            label="unsafe automatic route",
        )
        ax.set_title(kind.replace("_", " "))
        ax.set_xlabel("Severity")
        ax.set_ylim(0, 1)
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Degradation under image corruptions, test units (synthetic data)", fontsize=10)
    return save_figure(fig, path, df)
