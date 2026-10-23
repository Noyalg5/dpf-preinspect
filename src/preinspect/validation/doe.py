"""Imaging design of experiments: a replicated 2^3 full factorial.

Factors: lighting mode, exposure (EV) and sensor resolution (a lower
resolution is rendered and upsampled to the model's input size). Responses,
measured by the perception model on re-imaged held-out units:

- crack recall: share of faces with a real crack where the model reports one;
- false-reject rate: share of defect-free faces where it reports any defect.

Confound control: a model can favour the conditions it was trained on. Both
lighting levels and both exposure levels lie inside the training
distribution (lighting modes were mixed and exposure varied over a wider
range during data generation), units come only from splits the perception
model never saw, run order is randomised, and every replicate re-seats the
units. Resolution is the exception: the model only ever saw the higher
level, so the resolution effect includes a train/test mismatch; the report
says so rather than presenting it as a pure optics effect.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from statsmodels.stats.anova import anova_lm

from preinspect.models.perception import predict
from preinspect.report.style import INK, SERIES, save_figure
from preinspect.service.pipeline import InspectionPipeline
from preinspect.utils.seeding import make_rng
from preinspect.validation.reimage import rerender

FACTORS = ("lighting", "exposure_ev", "resolution")


def doe_images(units: pd.DataFrame, scenes: dict, n: int, seed: int) -> dict[str, list]:
    """Faces with a crack and defect-free faces from units unseen by perception training."""
    pool = units[units["split"].isin(["calib", "test"])]["unit_id"].tolist()
    rng = np.random.default_rng(seed)
    crack, clean = [], []
    for uid in rng.permutation(pool):
        defects = scenes[uid]["scene"]["defects"]
        for view in ("inlet", "outlet"):
            kinds = {d["cls"] for d in defects if d["view"] == view}
            if "crack" in kinds and len(crack) < n:
                crack.append((uid, view))
            elif not kinds and len(clean) < n:
                clean.append((uid, view))
    return {"crack": crack, "clean": clean}


def run_doe(cfg: dict, pipe: InspectionPipeline, units: pd.DataFrame, scenes: dict) -> pd.DataFrame:
    """Responses for every run of the replicated factorial, in randomised order."""
    d = cfg["campaign"]["doe"]
    images = doe_images(units, scenes, d["images_per_class"], cfg["seed"])
    classes = cfg["perception"]["classes"]
    k_crack = classes.index("crack")
    jitter = cfg["campaign"]["msa"]["jitter"]
    levels = [d["factors"][f] for f in FACTORS]
    runs = [(combo, rep) for combo in itertools.product(*levels) for rep in range(d["replicates"])]
    order = make_rng(cfg["seed"], "doe-order").permutation(len(runs))
    rows = []
    for run_index, i in enumerate(order):
        (light, ev, res), rep = runs[i]
        rng = make_rng(cfg["seed"], "doe", light, ev, res, rep)
        scores = {}
        for group, items in images.items():
            imgs = [
                rerender(cfg, scenes[u], v, rng, jitter, size=res, lighting=light, exposure_ev=ev)
                for u, v in items
            ]
            x = np.stack(imgs).astype(np.float32) / 255.0
            _, presence = predict(
                pipe.unets, torch.from_numpy(x.transpose(0, 3, 1, 2).copy()), pipe.device
            )
            scores[group] = presence
        thr = cfg["perception"]["mask_threshold"]
        rows.append(
            {
                "run_order": run_index + 1,
                "lighting": light,
                "exposure_ev": ev,
                "resolution": res,
                "replicate": rep + 1,
                "crack_recall": float((scores["crack"][:, k_crack] >= thr).mean()),
                "false_reject_rate": float((scores["clean"] >= thr).any(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("run_order")


def analyse(df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    """Coded-factor ANOVA per response and the recommended set-up."""
    d = cfg["campaign"]["doe"]
    coded = df.copy()
    for f in FACTORS:
        lo, hi = d["factors"][f]
        coded[f] = np.where(coded[f] == hi, 1.0, -1.0)
    out: dict[str, Any] = {}
    # Without replicates the full model is saturated; the three-way interaction
    # is then pooled into the error term.
    terms = "lighting * exposure_ev * resolution"
    if d["replicates"] < 2:
        terms += " - lighting:exposure_ev:resolution"
    for response in ("crack_recall", "false_reject_rate"):
        model = smf.ols(f"{response} ~ {terms}", data=coded).fit()
        effects = {k: 2 * v for k, v in model.params.items() if k != "Intercept"}
        p_values: dict[str, float] = {}
        if np.isfinite(model.ssr) and model.df_resid > 0 and coded[response].nunique() > 1:
            p_values = anova_lm(model, typ=2)["PR(>F)"].dropna().to_dict()
        out[response] = {"effects": effects, "p_values": p_values}
    means = df.groupby(list(FACTORS))[["crack_recall", "false_reject_rate"]].mean().reset_index()
    means["score"] = means["crack_recall"] - d["false_reject_weight"] * means["false_reject_rate"]
    best = means.sort_values("score", ascending=False).iloc[0]
    out["settings"] = means.to_dict("records")
    out["recommended"] = {
        f: (best[f].item() if hasattr(best[f], "item") else best[f]) for f in FACTORS
    }
    out["recommended_scores"] = {
        "crack_recall": float(best["crack_recall"]),
        "false_reject_rate": float(best["false_reject_rate"]),
    }
    return out


def effects_figure(df: pd.DataFrame, path: Path) -> Path:
    """Main-effects plots (top) and the lighting x exposure interaction (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    for col, factor in enumerate(FACTORS):
        for row, (response, colour) in enumerate(
            (("crack_recall", SERIES[0]), ("false_reject_rate", SERIES[1]))
        ):
            m = df.groupby(factor)[response].mean()
            axes[row, col].plot([str(x) for x in m.index], m.values, marker="o", color=colour)
            axes[row, col].set_title(
                f"{response.replace('_', ' ')} by {factor.replace('_', ' ')}", fontsize=9
            )
            axes[row, col].set_ylim(0, 1)
    fig.suptitle("Imaging DoE main effects, means over runs (synthetic data)", fontsize=10)
    save_figure(fig, path, df)
    inter = df.groupby(["lighting", "exposure_ev"])["crack_recall"].mean().unstack()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for k, ev in enumerate(inter.columns):
        ax.plot(
            inter.index.astype(str),
            inter[ev],
            marker="o",
            color=SERIES[k],
            label=f"exposure {ev:+.1f} EV",
        )
    ax.set_ylabel("Crack recall")
    ax.set_ylim(0, 1)
    ax.set_title("Lighting x exposure interaction")
    ax.legend()
    ax.tick_params(colors=INK["muted"])
    return save_figure(fig, path.with_name("doe_interaction.png"), inter.reset_index())
