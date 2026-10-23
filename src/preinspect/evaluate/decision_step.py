"""Conformal sets, threshold selection and routing applied to a run's predictions.

Data use:
- calibration-fit half: probability calibration, anomaly threshold, routing thresholds;
- calibration-conformal half: conformal quantiles;
- test split: every reported number.
Labels on calibration units are the *recorded* dispositions, as on a real line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preinspect.config import Config, load_aux
from preinspect.decision.conformal import ConformalInterval, ConformalOutcome, coverage
from preinspect.decision.cost import OUTCOMES, ROUTES, cost_matrix, expected_cost, route_rates
from preinspect.decision.rationale import rationale
from preinspect.decision.routing import Policy, choose_thresholds, route
from preinspect.evaluate.scoring import PROBA
from preinspect.report.style import INK, SERIES, save_figure


def load_halves(run_path: Path) -> tuple[set[str], set[str]]:
    """Calibration-fit and conformal unit IDs recorded by the evaluation step."""
    halves = pd.read_json(run_path / "calibration_units.json", typ="series")
    return set(halves["fit"]), set(halves["conformal"])


def decide(
    cfg: Config, preds: pd.DataFrame, anomaly: pd.Series, fit_ids: set[str], conf_ids: set[str]
) -> dict[str, Any]:
    """Fit conformal sets and routing thresholds, then route every unit.

    Parameters
    ----------
    preds : pandas.DataFrame
        Calibrated predictions with truth columns.
    anomaly : pandas.Series
        Unexplained-anomaly score indexed by unit ID.
    """
    rcfg = cfg["routing"]
    costs = load_aux("costs")
    matrix = cost_matrix(costs)
    proba = preds[PROBA].to_numpy()
    y_rec = preds["disposition"].map(OUTCOMES.index).to_numpy()
    in_fit = preds["unit_id"].isin(fit_ids).to_numpy()
    in_conf = preds["unit_id"].isin(conf_ids).to_numpy()
    alphas = sorted({*rcfg["alphas"], *rcfg["alpha_grid"], rcfg["interval_alpha"]})
    sets = ConformalOutcome(alphas).fit(proba[in_conf], y_rec[in_conf])
    intervals = ConformalInterval(alphas).fit(
        preds.loc[in_conf, "dp_ratio_post_clean"].to_numpy(),
        preds.loc[in_conf, "dp_pred"].to_numpy(),
    )
    anom = anomaly.reindex(preds["unit_id"]).to_numpy()
    anom_thr = float(np.nanquantile(anom[in_fit], rcfg["anomaly_quantile"]))
    in_set = {a: sets.predict(proba[in_fit], a)[:, 2] for a in rcfg["alpha_grid"]}
    policy, search = choose_thresholds(
        proba[in_fit],
        in_set,
        anom[in_fit],
        preds.loc[in_fit, "disposition"].to_numpy(),
        matrix,
        costs["policy"]["wrongful_scrap_cap"]["value"],
        rcfg,
        anom_thr,
        costs["policy"]["min_scrap_capture"]["value"],
    )
    member = sets.predict(proba, policy.alpha)
    routes, reasons = route(proba, member[:, 2], anom, np.ones(len(preds), bool), policy)
    out = preds.copy()
    out["route"], out["reason"] = routes, reasons
    out["prediction_set"] = [
        "|".join(c for c, m in zip(OUTCOMES, row, strict=True) if m) for row in member
    ]
    lo, hi = intervals.predict(preds["dp_pred"].to_numpy(), rcfg["interval_alpha"])
    out["dp_lo"], out["dp_hi"] = lo, hi
    out["anom_unexplained"] = anom
    return {
        "table": out,
        "policy": policy,
        "search": search,
        "conformal": sets,
        "intervals": intervals,
        "matrix": matrix,
    }


def test_report(cfg: Config, d: dict[str, Any]) -> dict[str, Any]:
    """Coverage, routing rates, cost and savings on the test split."""
    t = d["table"][d["table"]["split"] == "test"]
    y_true = t["outcome_true"].map(OUTCOMES.index).to_numpy()
    y_rec = t["disposition"].map(OUTCOMES.index).to_numpy()
    proba = t[PROBA].to_numpy()
    cov = {}
    for a in cfg["routing"]["alphas"]:
        member = d["conformal"].predict(proba, a)
        lo, hi = d["intervals"].predict(t["dp_pred"].to_numpy(), a)
        rec, true = t["dp_ratio_post_clean"].to_numpy(), t["dp_ratio_post_clean_true"].to_numpy()
        cov[str(a)] = {
            "sets_vs_truth": coverage(member, y_true),
            "sets_vs_recorded": coverage(member, y_rec),
            "interval_vs_recorded": float(np.mean((rec >= lo) & (rec <= hi))),
            "interval_vs_truth": float(np.mean((true >= lo) & (true <= hi))),
            "interval_half_width": d["intervals"].q[a],
            "qhat": d["conformal"].qhat[a],
        }
    outcomes = t["outcome_true"].to_numpy()
    baseline = expected_cost(np.full(len(t), "PROCEED"), outcomes, d["matrix"])
    gated = expected_cost(t["route"].to_numpy(), outcomes, d["matrix"])
    return {
        "policy": d["policy"].as_dict(),
        "coverage": cov,
        "rates": route_rates(t["route"].to_numpy(), outcomes),
        "route_confusion": pd.crosstab(t["route"], t["outcome_true"])
        .reindex(index=ROUTES, columns=OUTCOMES, fill_value=0)
        .to_dict("index"),
        "reasons": t["reason"].value_counts().to_dict(),
        "cost_per_unit": {"baseline": baseline, "gated": gated, "saving": baseline - gated},
        "n_test": int(len(t)),
    }


def add_rationales(cfg: Config, table: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Template rationale for every unit in a decision table."""
    feats = features.set_index("unit_id")
    texts = []
    for row in table.itertuples(index=False):
        f = feats.loc[row.unit_id].to_dict() if row.unit_id in feats.index else {}
        proba = {c: float(getattr(row, f"p_{c}")) for c in OUTCOMES}
        texts.append(
            rationale(
                f, proba, row.prediction_set.split("|"), row.route, row.reason, cfg["rationale"]
            )
        )
    return pd.Series(texts, index=table.index)


def cost_curve_figure(search: pd.DataFrame, policy: Policy, cap: float, path: Path) -> Path:
    """Expected cost and wrongful-scrap rate against the quarantine threshold."""
    best = search[
        (search["t_rework"] == policy.t_rework) & (search["alpha"] == policy.alpha)
    ].sort_values("t_quarantine")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    axes[0].plot(best["t_quarantine"], best["cost"], color=SERIES[0])
    axes[0].axvline(policy.t_quarantine, color=INK["muted"], linewidth=1)
    axes[0].set_xlabel("Quarantine threshold on P(SCRAP)")
    axes[0].set_ylabel("Expected cost per unit (GBP, assumed costs)")
    axes[0].set_title("Expected cost")
    axes[1].plot(best["t_quarantine"], best["wrongful_scrap_rate"], color=SERIES[1])
    axes[1].axhline(cap, color=INK["muted"], linewidth=1)
    axes[1].axvline(policy.t_quarantine, color=INK["muted"], linewidth=1)
    axes[1].set_xlabel("Quarantine threshold on P(SCRAP)")
    axes[1].set_ylabel("Wrongful-scrap rate")
    axes[1].set_title("Constraint (horizontal line: cap)")
    fig.suptitle(
        f"Calibration data; rework threshold fixed at its chosen value "
        f"{policy.t_rework:.2f} (synthetic data)",
        fontsize=9,
    )
    return save_figure(fig, path, best)
