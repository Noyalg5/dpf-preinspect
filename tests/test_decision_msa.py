"""Conformal sets, routing rules, cost model and MSA statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from preinspect.config import load_aux
from preinspect.decision.conformal import ConformalOutcome, aps_scores, aps_sets, coverage
from preinspect.decision.cost import cost_matrix, route_rates
from preinspect.decision.routing import Policy, route
from preinspect.validation.msa_stats import attribute_agreement, gauge_rr


def test_aps_coverage_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4000, 3)) * 2
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in proba])
    cal, test = slice(0, 2000), slice(2000, None)
    conf = ConformalOutcome([0.1]).fit(proba[cal], y[cal])
    cov = coverage(conf.predict(proba[test], 0.1), y[test])
    assert 0.88 <= cov["coverage"] <= 0.93
    assert (aps_scores(proba, y) <= 1.0 + 1e-9).all()
    assert aps_sets(proba, 1.0).all()


def test_routing_rules_in_order() -> None:
    policy = Policy(t_quarantine=0.7, t_rework=0.5, alpha=0.1, anomaly_threshold=1.0)
    proba = np.array(
        [
            [0.1, 0.1, 0.8],
            [0.5, 0.2, 0.3],
            [0.2, 0.7, 0.1],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8],
        ]
    )
    scrap_in_set = np.array([True, True, False, False, False, True])
    anomaly = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0])
    valid = np.array([True, True, True, True, True, False])
    routes, reasons = route(proba, scrap_in_set, anomaly, valid, policy)
    assert routes.tolist() == ["QUARANTINE", "REFER", "EXTENDED_CLEAN", "PROCEED", "REFER", "REFER"]
    assert reasons.tolist() == [
        "quarantine",
        "ambiguous",
        "rework",
        "proceed",
        "anomaly",
        "invalid",
    ]


def test_cost_model_prefers_correct_routes() -> None:
    m = cost_matrix(load_aux("costs"))
    assert m[2, 2] < m[0, 2]  # quarantining true scrap is cheaper than cleaning it
    assert m[2, 0] > m[0, 0]  # quarantining a good unit costs more than proceeding
    rates = route_rates(
        np.array(["QUARANTINE", "PROCEED", "REFER"]), np.array(["PASS", "SCRAP", "SCRAP"])
    )
    assert rates["wrongful_scrap_rate"] == 1.0 and rates["scrap_capture_rate"] == 0.5


def test_gauge_rr_separates_part_and_gauge_variation() -> None:
    rng = np.random.default_rng(1)
    rows = [
        {"part": p, "operator": o, "trial": t, "value": p * 1.0 + 0.05 * o + rng.normal(0, 0.05)}
        for p in range(10)
        for o in range(3)
        for t in range(3)
    ]
    res = gauge_rr(pd.DataFrame(rows))
    assert res["pct_grr"] < 10 and res["ndc"] >= 5


def test_attribute_agreement_perfect_appraiser() -> None:
    units = [f"U{i}" for i in range(12)]
    std = pd.Series(["PASS", "REWORK", "SCRAP"] * 4, index=units)
    calls = pd.DataFrame(
        [
            {"unit_id": u, "appraiser": a, "trial": t, "call": std[u]}
            for u in units
            for a in ("A", "B")
            for t in (1, 2)
        ]
    )
    res = attribute_agreement(calls, std)
    assert res["appraisers"]["A"]["within_agreement"] == 1.0
    assert res["fleiss_kappa"] == 1.0
