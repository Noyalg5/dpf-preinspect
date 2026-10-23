"""Discrete-event model of the cleaning line, with and without the pre-inspection gate.

Each unit arrives, (optionally) passes the pre-inspection station, and follows
its route through bake ovens, pneumatic cleaning, the test bench and
inspectors until it is dispatched or scrapped. Routes are drawn from the
deployed policy's measured route-given-outcome rates on the test split, so the
simulation inherits the classifier's real error pattern.

Simplifications: ovens are modelled as independent slots running around the
clock; there are no shift calendars, breakdowns or set-up times; a reworked
unit always passes after its extended clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import simpy

OUTCOMES = ["PASS", "REWORK", "SCRAP"]
ROUTES = ["PROCEED", "EXTENDED_CLEAN", "QUARANTINE", "REFER"]


@dataclass
class Tally:
    """Counters and accumulators for one simulated scenario."""

    arrived: int = 0
    dispatched: int = 0
    scrapped: int = 0
    wrongful_scraps: int = 0
    standard_bakes: int = 0
    extended_bakes: int = 0
    tests: int = 0
    inspector_min: float = 0.0
    in_system: int = 0
    lead_times_h: list[float] = field(default_factory=list)
    time_in_system_min: float = 0.0


class Line:
    """The line's shared resources and unit process."""

    def __init__(
        self,
        env: simpy.Environment,
        sim: dict[str, Any],
        costs: dict[str, Any],
        rng: np.random.Generator,
        gated: bool,
        route_given_outcome: dict,
    ) -> None:
        r = sim["resources"]
        self.env, self.sim, self.costs, self.rng, self.gated = env, sim, costs, rng, gated
        self.ovens = simpy.Resource(env, r["oven_slots"])
        self.bench = simpy.Resource(env, r["test_benches"])
        self.inspector = simpy.Resource(env, r["inspectors"])
        self.station = simpy.Resource(env, r["pre_inspection_stations"])
        self.route_given_outcome = route_given_outcome
        self.t = Tally()
        self.warmup = sim["warmup_days"] * 1440.0
        self.open_starts: dict[int, float] = {}
        self.next_key = 0

    def _use(self, resource: simpy.Resource, minutes: float):
        with resource.request() as req:
            yield req
            yield self.env.timeout(minutes)

    def _bake_and_test(self, extended: bool):
        tm = self.sim["times_min"]
        yield from self._use(self.ovens, tm["extended_bake" if extended else "standard_bake"])
        yield self.env.timeout(tm["pneumatic_clean"])
        yield from self._use(self.bench, tm["downstream_test"])
        if self.env.now >= self.warmup:
            self.t.tests += 1
            if extended:
                self.t.extended_bakes += 1
            else:
                self.t.standard_bakes += 1

    def _today(self, outcome: str):
        """Today's process: standard clean and test; rework once; scrap on failure."""
        yield from self._bake_and_test(extended=False)
        if outcome == "REWORK":
            yield from self._bake_and_test(extended=True)
        return outcome != "SCRAP"

    def _inspect(self, minutes: float):
        yield from self._use(self.inspector, minutes)
        if self.env.now >= self.warmup:
            self.t.inspector_min += minutes

    def unit(self, outcome: str):
        """Process one unit from arrival to dispatch or scrap."""
        tm, items = self.sim["times_min"], self.costs["items"]
        start = self.env.now
        counted = start >= self.warmup
        self.next_key += 1
        key = self.next_key
        if counted:
            self.open_starts[key] = start
        if counted:
            self.t.arrived += 1
            self.t.in_system += 1
        dispatched, wrongful = True, False
        if not self.gated:
            dispatched = yield from self._today(outcome)
        else:
            yield from self._use(self.station, tm["pre_inspection"])
            probs = self.route_given_outcome[outcome]
            route = ROUTES[self.rng.choice(len(ROUTES), p=probs)]
            if route == "PROCEED":
                dispatched = yield from self._today(outcome)
            elif route == "EXTENDED_CLEAN":
                yield from self._bake_and_test(extended=True)
                dispatched = outcome != "SCRAP"
            elif route == "QUARANTINE":
                yield from self._inspect(tm["quarantine_review"])
                if outcome == "SCRAP":
                    dispatched = False
                elif self.rng.random() < items["release_probability"]["value"]:
                    dispatched = yield from self._today(outcome)
                else:
                    dispatched, wrongful = False, True
            else:
                yield from self._inspect(tm["refer_review"])
                if outcome == "SCRAP" and self.rng.random() < items["refer_scrap_catch"]["value"]:
                    dispatched = False
                else:
                    dispatched = yield from self._today(outcome)
        if counted:
            self.open_starts.pop(key)
            self.t.in_system -= 1
            self.t.time_in_system_min += self.env.now - start
            if dispatched:
                self.t.dispatched += 1
                self.t.lead_times_h.append((self.env.now - start) / 60.0)
            else:
                self.t.scrapped += 1
                self.t.wrongful_scraps += int(wrongful)

    def arrivals(self, outcome_mix: np.ndarray):
        """Poisson arrivals; each unit's true outcome drawn from the outcome mix."""
        rate = self.sim["arrivals"]["units_per_day"] / 1440.0
        while True:
            yield self.env.timeout(self.rng.exponential(1.0 / rate))
            outcome = OUTCOMES[self.rng.choice(3, p=outcome_mix)]
            self.env.process(self.unit(outcome))


def simulate(
    sim: dict[str, Any],
    costs: dict[str, Any],
    outcome_mix: np.ndarray,
    route_given_outcome: dict,
    gated: bool,
    seed: int,
) -> dict[str, float]:
    """Run one replication and return its KPIs."""
    env = simpy.Environment()
    line = Line(env, sim, costs, np.random.default_rng(seed), gated, route_given_outcome)
    env.process(line.arrivals(outcome_mix))
    horizon = (sim["warmup_days"] + sim["horizon_days"]) * 1440.0
    env.run(until=horizon)
    return kpis(line, sim, costs, horizon)


def _open_time(line: Line, horizon: float) -> float:
    """Time already spent by units still in the system at the end (for mean WIP)."""
    return float(sum(horizon - start for start in line.open_starts.values()))


def kpis(
    line: Line, sim: dict[str, Any], costs: dict[str, Any], horizon: float
) -> dict[str, float]:
    """Throughput, oven use, energy, CO2e, cost, scrap errors, WIP and lead time."""
    t, it, en = line.t, costs["items"], costs["energy"]
    days = sim["horizon_days"]
    tm = sim["times_min"]
    oven_h = (t.standard_bakes * tm["standard_bake"] + t.extended_bakes * tm["extended_bake"]) / 60
    energy = (
        t.standard_bakes * en["standard_clean_kwh"]["value"]
        + t.extended_bakes * en["extended_clean_kwh"]["value"]
    )
    cost = (
        t.standard_bakes * it["standard_clean"]["value"]
        + t.extended_bakes * it["extended_clean"]["value"]
        + t.tests * it["downstream_test"]["value"]
        + t.inspector_min * it["inspector_rate"]["value"]
        + t.scrapped * it["scrap_handling"]["value"]
        + t.wrongful_scraps * it["wrongful_scrap_value"]["value"]
    )
    completed = t.dispatched + t.scrapped
    wip_end = t.in_system
    return {
        "arrived": float(t.arrived),
        "dispatched": float(t.dispatched),
        "scrapped": float(t.scrapped),
        "wip_end": float(wip_end),
        "wrongful_scraps": float(t.wrongful_scraps),
        "throughput_per_day": t.dispatched / days,
        "oven_hours": oven_h,
        "energy_kwh": energy,
        "co2e_kg": energy * en["grid_co2e_kg_per_kwh"]["value"],
        "cost_per_unit": cost / max(completed, 1),
        "lead_time_h_mean": float(np.mean(t.lead_times_h)) if t.lead_times_h else float("nan"),
        "lead_time_h_p90": float(np.percentile(t.lead_times_h, 90))
        if t.lead_times_h
        else float("nan"),
        "wip_mean": (t.time_in_system_min + _open_time(line, horizon)) / (days * 1440.0),
        "oven_utilisation": oven_h * 60 / (line.ovens.capacity * days * 1440.0),
    }
