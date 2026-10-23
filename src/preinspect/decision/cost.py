"""Expected-cost model for routing decisions.

Every figure comes from ``configs/costs.yaml``, where each value is labelled
as an assumption. The cost of a unit depends on the route it was given and
its true outcome:

- PROCEED: today's process (standard clean, test; REWORK adds an extended
  clean and a re-test; SCRAP is scrapped after the wasted clean and test).
- EXTENDED_CLEAN: extended clean and test straight away.
- QUARANTINE: an inspector confirms. True scrap is scrapped without
  cleaning; a serviceable unit is released with ``release_probability``
  (then processed as today) or else wrongly scrapped.
- REFER: an inspector reviews, catching a true scrap unit with probability
  ``refer_scrap_catch`` but also wrongly scrapping a serviceable unit with
  probability ``refer_false_scrap``; otherwise the unit is processed as today.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ROUTES = ["PROCEED", "EXTENDED_CLEAN", "QUARANTINE", "REFER"]
OUTCOMES = ["PASS", "REWORK", "SCRAP"]


def _v(costs: dict[str, Any], section: str, key: str) -> float:
    return float(costs[section][key]["value"])


def baseline_cost(costs: dict[str, Any]) -> dict[str, float]:
    """Cost per unit of today's process (no pre-inspection), by true outcome."""
    std, ext = _v(costs, "items", "standard_clean"), _v(costs, "items", "extended_clean")
    test, scrap = _v(costs, "items", "downstream_test"), _v(costs, "items", "scrap_handling")
    return {"PASS": std + test, "REWORK": std + test + ext + test, "SCRAP": std + test + scrap}


def cost_matrix(costs: dict[str, Any]) -> np.ndarray:
    """``(4 routes, 3 outcomes)`` expected cost per unit (currency units)."""
    it = "items"
    ext, test = _v(costs, it, "extended_clean"), _v(costs, it, "downstream_test")
    scrap, rate = _v(costs, it, "scrap_handling"), _v(costs, it, "inspector_rate")
    wrongful, release = _v(costs, it, "wrongful_scrap_value"), _v(costs, it, "release_probability")
    catch = _v(costs, it, "refer_scrap_catch")
    false_scrap = _v(costs, it, "refer_false_scrap")
    quarantine = rate * _v(costs, it, "quarantine_review_min")
    refer = rate * _v(costs, it, "refer_review_min")
    base = baseline_cost(costs)
    m = np.zeros((len(ROUTES), len(OUTCOMES)))
    for j, o in enumerate(OUTCOMES):
        m[0, j] = base[o]
        m[1, j] = ext + test + (scrap if o == "SCRAP" else 0.0)
        if o == "SCRAP":
            m[2, j] = quarantine + scrap
            m[3, j] = refer + catch * scrap + (1 - catch) * base[o]
        else:
            m[2, j] = quarantine + release * base[o] + (1 - release) * (wrongful + scrap)
            m[3, j] = refer + false_scrap * (wrongful + scrap) + (1 - false_scrap) * base[o]
    return m


def expected_cost(routes: np.ndarray, outcomes: np.ndarray, matrix: np.ndarray) -> float:
    """Mean cost per unit for arrays of route and outcome names."""
    r = np.array([ROUTES.index(x) for x in routes])
    o = np.array([OUTCOMES.index(x) for x in outcomes])
    return float(matrix[r, o].mean())


def route_rates(routes: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """Safety and workload rates for a set of routing decisions.

    - ``wrongful_scrap_rate``: serviceable units (PASS or REWORK) sent to QUARANTINE.
    - ``scrap_capture_rate``: true SCRAP units sent to QUARANTINE or REFER.
    - ``missed_scrap_rate``: true SCRAP units released to cleaning unchecked.
    """
    routes, outcomes = np.asarray(routes), np.asarray(outcomes)
    serviceable = outcomes != "SCRAP"
    scrap = ~serviceable
    out = {f"share_{r}": float((routes == r).mean()) for r in ROUTES}
    out["wrongful_scrap_rate"] = (
        float((routes[serviceable] == "QUARANTINE").mean()) if serviceable.any() else float("nan")
    )
    captured = np.isin(routes[scrap], ["QUARANTINE", "REFER"])
    out["scrap_capture_rate"] = float(captured.mean()) if scrap.any() else float("nan")
    out["missed_scrap_rate"] = 1.0 - out["scrap_capture_rate"] if scrap.any() else float("nan")
    return out


def energy_per_route(costs: dict[str, Any]) -> dict[str, float]:
    """Oven energy (kWh) consumed by a unit's first clean, by route."""
    std = _v(costs, "energy", "standard_clean_kwh")
    ext = _v(costs, "energy", "extended_clean_kwh")
    return {"PROCEED": std, "EXTENDED_CLEAN": ext, "QUARANTINE": 0.0, "REFER": std}
