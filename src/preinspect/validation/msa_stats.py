"""Measurement system analysis statistics (AIAG MSA Reference Manual, 4th ed.).

- Attribute agreement analysis: within-appraiser repeatability, agreement
  between appraisers, agreement with the standard, Cohen's kappa against the
  standard and Fleiss' kappa across appraisers.
- Variable gauge R&R by two-way crossed ANOVA (part x operator, with
  interaction): repeatability (equipment variation), reproducibility
  (appraiser variation), %GRR of total study variation and the number of
  distinct categories (ndc).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


def attribute_agreement(calls: pd.DataFrame, standard: pd.Series) -> dict[str, Any]:
    """Agreement statistics for repeated categorical calls.

    Parameters
    ----------
    calls : pandas.DataFrame
        Columns ``unit_id``, ``appraiser``, ``trial``, ``call``.
    standard : pandas.Series
        Reference call per unit ID.

    Returns
    -------
    dict
        Per-appraiser within, versus-standard and kappa figures; overall
        between-appraiser agreement and Fleiss' kappa.
    """
    wide = calls.pivot_table(
        index="unit_id", columns=["appraiser", "trial"], values="call", aggfunc="first"
    )
    std = standard.reindex(wide.index)
    per = {}
    for app in wide.columns.get_level_values(0).unique():
        trials = wide[app]
        same = trials.nunique(axis=1) == 1
        match = trials.eq(std, axis=0).all(axis=1)
        pooled_calls = trials.stack()
        pooled_std = std.reindex(pooled_calls.index.get_level_values(0)).to_numpy()
        per[app] = {
            "within_agreement": float(same.mean()),
            "vs_standard": float(match.mean()),
            "cohen_kappa_vs_standard": float(
                cohen_kappa_score(pooled_std, pooled_calls.to_numpy())
            ),
        }
    all_same = wide.nunique(axis=1) == 1
    first = calls[calls["trial"] == calls["trial"].min()].pivot(
        index="unit_id", columns="appraiser", values="call"
    )
    table, _ = aggregate_raters(first.to_numpy())
    return {
        "appraisers": per,
        "between_all_agree": float(all_same.mean()),
        "all_agree_with_standard": float(wide.eq(std, axis=0).all(axis=1).mean()),
        "fleiss_kappa": float(fleiss_kappa(table)),
        "n_units": int(len(wide)),
    }


def gauge_rr(df: pd.DataFrame, value: str = "value") -> dict[str, float]:
    """Two-way crossed ANOVA gauge R&R.

    Parameters
    ----------
    df : pandas.DataFrame
        Balanced design with columns ``part``, ``operator``, ``trial`` and ``value``.

    Returns
    -------
    dict
        Variance components, %GRR (share of total study variation, in
        standard-deviation terms), %EV, %AV, %PV and ndc.
    """
    p, o = df["part"].nunique(), df["operator"].nunique()
    r = int(df.groupby(["part", "operator"]).size().iloc[0])
    grand = df[value].mean()
    part_means = df.groupby("part")[value].mean()
    op_means = df.groupby("operator")[value].mean()
    cell_means = df.groupby(["part", "operator"])[value].mean()
    ss_part = o * r * ((part_means - grand) ** 2).sum()
    ss_op = p * r * ((op_means - grand) ** 2).sum()
    ss_cells = r * ((cell_means - grand) ** 2).sum()
    ss_int = ss_cells - ss_part - ss_op
    ss_err = ((df[value] - df.set_index(["part", "operator"]).index.map(cell_means)) ** 2).sum()
    ms_part = ss_part / (p - 1)
    ms_op = ss_op / (o - 1)
    ms_int = ss_int / ((p - 1) * (o - 1))
    ms_err = ss_err / (p * o * (r - 1))
    var_ev = ms_err
    var_int = max(0.0, (ms_int - ms_err) / r)
    var_op = max(0.0, (ms_op - ms_int) / (p * r))
    var_part = max(0.0, (ms_part - ms_int) / (o * r))
    var_av = var_op + var_int
    var_grr = var_ev + var_av
    var_total = var_grr + var_part
    sd = {
        k: math.sqrt(v)
        for k, v in {
            "ev": var_ev,
            "av": var_av,
            "grr": var_grr,
            "pv": var_part,
            "tv": var_total,
        }.items()
    }
    return {
        "sd_repeatability": sd["ev"],
        "sd_reproducibility": sd["av"],
        "sd_grr": sd["grr"],
        "sd_part": sd["pv"],
        "pct_ev": 100 * sd["ev"] / sd["tv"],
        "pct_av": 100 * sd["av"] / sd["tv"],
        "pct_grr": 100 * sd["grr"] / sd["tv"],
        "pct_pv": 100 * sd["pv"] / sd["tv"],
        "ndc": float(np.floor(1.41 * sd["pv"] / sd["grr"])) if sd["grr"] > 0 else float("inf"),
        "parts": p,
        "operators": o,
        "trials": r,
    }
