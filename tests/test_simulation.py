"""Line simulation: conservation of units and sensible direction of effects."""

from __future__ import annotations

import numpy as np

from preinspect.config import load_aux
from preinspect.sim.line import simulate
from preinspect.sim.tornado import adjust, route_matrix

CONFUSION = {
    "PROCEED": {"PASS": 40, "REWORK": 10, "SCRAP": 3},
    "EXTENDED_CLEAN": {"PASS": 5, "REWORK": 15, "SCRAP": 2},
    "QUARANTINE": {"PASS": 1, "REWORK": 0, "SCRAP": 15},
    "REFER": {"PASS": 4, "REWORK": 2, "SCRAP": 5},
}


def _sim(days: int = 6) -> dict:
    sim = load_aux("line_sim")
    sim["horizon_days"], sim["warmup_days"] = days, 1
    return sim


def test_units_in_equal_units_out_plus_wip() -> None:
    mix = np.array([0.5, 0.27, 0.23])
    for gated in (False, True):
        k = simulate(_sim(), load_aux("costs"), mix, route_matrix(CONFUSION), gated, seed=3)
        assert k["arrived"] == k["dispatched"] + k["scrapped"] + k["wip_end"]
        assert k["arrived"] > 0


def test_gate_saves_oven_time_when_it_catches_scrap() -> None:
    mix = np.array([0.5, 0.27, 0.23])
    m = route_matrix(CONFUSION)
    base = simulate(_sim(), load_aux("costs"), mix, m, False, seed=5)
    gated = simulate(_sim(), load_aux("costs"), mix, adjust(m, scrap_capture=0.95), True, seed=5)
    assert gated["oven_hours"] < base["oven_hours"]
    assert gated["energy_kwh"] < base["energy_kwh"]


def test_adjust_keeps_rows_normalised() -> None:
    m = adjust(route_matrix(CONFUSION), scrap_capture=0.9, false_quarantine=0.03)
    for row in m.values():
        assert abs(row.sum() - 1.0) < 1e-9
    assert abs(m["SCRAP"][2] + m["SCRAP"][3] - 0.9) < 1e-9
    assert abs(m["PASS"][2] - 0.03) < 1e-9
