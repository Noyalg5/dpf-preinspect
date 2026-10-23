"""Unit population: service metadata (observable) and latent condition.

Each unit's draws come from a generator keyed by its ``unit_id``, so a unit
is reproducible on its own and independent of how many others exist.
The relationships follow ``docs/domain_primer.md`` section 8.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import numpy as np

from preinspect.utils.seeding import make_rng

Meta = dict[str, Any]
Condition = dict[str, Any]


def application_shares(gcfg: dict[str, Any], t: float) -> dict[str, float]:
    """Application mix at normalised arrival time ``t`` in [0, 1], including drift."""
    shares = {name: spec["share"] for name, spec in gcfg["applications"].items()}
    for name, change in gcfg["drift"]["share_change"].items():
        shares[name] = max(0.0, shares[name] + change * (t - 0.5))
    total = sum(shares.values())
    return {name: share / total for name, share in shares.items()}


def _choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=float)
    return names[rng.choice(len(names), p=probs / probs.sum())]


def _beta(rng: np.random.Generator, params: list[float]) -> float:
    return float(rng.beta(params[0], params[1]))


def sample_metadata(
    gcfg: dict[str, Any], seed: int, unit_id: str, arrival_day: int, population: str
) -> tuple[Meta, np.random.Generator]:
    """Draw substrate and service-history fields for one unit.

    Returns
    -------
    meta : dict
        Observable metadata (``weight_gain_g`` is filled later from the condition).
    rng : numpy.random.Generator
        The unit's generator, to be reused for its latent condition.
    """
    rng = make_rng(seed, "unit", unit_id)
    sub = gcfg["substrate"]
    t = min(1.0, arrival_day / gcfg["arrival"]["main_days"])
    if population == "shift":
        cpsi = int(sub["shift_cell_density_cpsi"])
    else:
        cpsi = int(_choice(rng, sub["cell_density_cpsi"]))
    app = _choice(rng, application_shares(gcfg, t))
    spec = gcfg["applications"][app]
    hours = float(rng.lognormal(math.log(spec["hours_median"]), spec["hours_sigma"]))
    interval = gcfg["latent"]["clean_interval_h"] * rng.uniform(0.8, 1.2)
    start = date.fromisoformat(gcfg["arrival"]["start"])
    meta = {
        "unit_id": unit_id,
        "arrival_date": (start + timedelta(days=int(arrival_day))).isoformat(),
        "material": _choice(rng, sub["materials"]),
        "cell_density_cpsi": cpsi,
        "diameter_mm": float(rng.choice(sub["diameters_mm"])),
        "length_mm": float(rng.choice(sub["lengths_mm"])),
        "application": app,
        "operating_hours": round(hours, 0),
        "previous_cleans": int(hours // interval),
        "population": population,
    }
    meta["_hours_since_clean"] = max(500.0, hours - meta["previous_cleans"] * interval)
    return meta, rng


def _cracks(lat: dict[str, Any], meta: Meta, temp: float, regens: int, rng) -> dict[str, Any]:
    c = lat["crack"]
    logit = (
        c["intercept"]
        + c["per_100c_over_700"] * max(0.0, temp - 700.0) / 100.0
        + c["cordierite"] * (meta["material"] == "cordierite")
        + c["per_1000_regens"] * regens / 1000.0
    )
    out: dict[str, Any] = {"cracks": [], "internal_crack_severity": 0.0}
    if rng.random() >= 1.0 / (1.0 + math.exp(-logit)):
        return out
    ring_share = c[f"ring_off_share_{meta['material']}"]
    n_cracks = 1 + int(rng.random() < c["second_crack"])
    for _ in range(n_cracks):
        if rng.random() < c["visible_share"]:
            kind = "ring_off" if rng.random() < ring_share else "radial"
            out["cracks"].append(
                {
                    "type": kind,
                    "severity": _beta(rng, c["severity_beta"]),
                    "through": bool(rng.random() < c["through_share"]),
                }
            )
        else:
            lo, hi = lat["internal_crack_severity"]
            out["internal_crack_severity"] = max(
                out["internal_crack_severity"], float(rng.uniform(lo, hi))
            )
    return out


def sample_condition(gcfg: dict[str, Any], meta: Meta, rng: np.random.Generator) -> Condition:
    """Draw the latent condition of a unit from its service history.

    Parameters
    ----------
    gcfg : dict
        Generator configuration.
    meta : dict
        Metadata from :func:`sample_metadata`; ``active_regen_count``,
        ``max_regen_temp_c`` and ``weight_gain_g`` are added in place.
    rng : numpy.random.Generator
        The unit's generator.

    Returns
    -------
    dict
        Latent condition, including per-defect descriptors for the scene sampler.
    """
    lat = gcfg["latent"]
    spec = gcfg["applications"][meta["application"]]
    hours = meta["operating_hours"]
    ash = (
        spec["ash_rate"]
        * lat["ash_gL_per_1000h"]
        * meta.pop("_hours_since_clean")
        / 1000.0
        * rng.lognormal(0.0, lat["ash_noise_sigma"])
    )
    soot = min(
        lat["soot_max_gL"],
        rng.gamma(lat["soot_gamma_shape"], lat["soot_gamma_scale"] * spec["soot_rate"]),
    )
    regens = int(rng.poisson(hours / 100.0 * spec["regen_per_100h"]))
    event_p = lat["uncontrolled_event_rate"] * spec["thermal"] * (1.0 + soot / 10.0)
    uncontrolled = bool(rng.random() < event_p) or meta["population"] == "novel"
    temp = rng.normal(
        lat["regen_temp_mean_c"] + lat["regen_temp_per_thermal_c"] * (spec["thermal"] - 1.0),
        lat["regen_temp_sd_c"],
    )
    if uncontrolled:
        temp = max(temp, rng.normal(lat["uncontrolled_temp_mean_c"], lat["uncontrolled_temp_sd_c"]))
    temp = float(min(temp, lat["regen_temp_cap_c"]))

    oil = bool(rng.random() < spec["oil_risk"] * (1.0 + hours / 20000.0))
    coolant = bool(rng.random() < spec["coolant_risk"])
    if coolant:
        glazing = rng.uniform(*lat["glazing_coolant"])
    elif oil:
        glazing = rng.uniform(*lat["glazing_oil"])
    else:
        glazing = _beta(rng, lat["glazing_base_beta"])

    cond: Condition = {
        "soot_load_gL": float(soot),
        "ash_load_gL": float(ash),
        "glazing": float(glazing),
        "oil_contaminated": oil,
        "coolant_contaminated": coolant,
        "uncontrolled_regen": uncontrolled,
        "max_regen_temp_true_c": temp,
        **_cracks(lat, meta, temp, regens, rng),
    }
    plug_mean = lat["missing_plug_mean"] + lat["missing_plug_event_mean"] * uncontrolled
    for view in ("inlet", "outlet"):
        cond[f"missing_plugs_{view}"] = int(min(rng.poisson(plug_mean), lat["missing_plug_cap"]))
    cond["chip_severity"] = (
        _beta(rng, lat["chip_severity_beta"]) if rng.random() < lat["chip_prob"] else 0.0
    )
    p_face = (
        lat["face_plug_base"] + lat["face_plug_per_soot_gL"] * soot + lat["face_plug_oil"] * oil
    )
    face = rng.random() < p_face
    cond["face_plug_extent"] = _beta(rng, lat["face_plug_extent_beta"]) if face else 0.0
    cond["face_plug_deposit"] = _choice(rng, lat["face_plug_deposit"]) if face else "none"
    if (oil or coolant) and rng.random() < lat["stain_given_contamination"]:
        cond["stain"] = "coolant" if coolant else "oil"
    else:
        cond["stain"] = "oil" if rng.random() < lat["spurious_stain"] else "none"
    cond["melt_severity"] = (
        float(rng.uniform(*lat["melt_severity"])) if meta["population"] == "novel" else 0.0
    )
    cond["bypass_fraction"] = bypass_fraction(gcfg, cond)
    leak = cond["bypass_fraction"] > lat["leak_bypass_threshold"]
    if leak and rng.random() < lat["outlet_soot_given_leak"]:
        extent = (
            lat["outlet_soot_extent_base"]
            + cond["bypass_fraction"] / lat["outlet_soot_bypass_scale"]
        )
        cond["outlet_soot_extent"] = float(min(1.0, extent))
    elif rng.random() < lat["spurious_outlet_soot"]:
        cond["outlet_soot_extent"] = float(rng.uniform(*lat["spurious_outlet_soot_extent"]))
    else:
        cond["outlet_soot_extent"] = 0.0
    lo, hi = lat["debris_count"]
    has_debris = rng.random() < lat["debris_prob"]
    cond["debris_count"] = int(rng.integers(lo, hi + 1)) if has_debris else 0

    sub = gcfg["substrate"]
    volume_l = math.pi * (meta["diameter_mm"] / 2.0) ** 2 * meta["length_mm"] * 1e-6
    load_g = volume_l * (soot + ash + glazing * sub["glaze_mass_gL"])
    meta["active_regen_count"] = regens
    meta["max_regen_temp_c"] = (
        float("nan")
        if rng.random() < lat["max_regen_temp_missing"]
        else round(temp + rng.normal(0.0, lat["regen_log_sd_c"]), 1)
    )
    meta["weight_gain_g"] = round(load_g + rng.normal(0.0, sub["weight_noise_g"]), 1)
    cond["volume_l"] = volume_l
    return cond


def bypass_fraction(gcfg: dict[str, Any], cond: Condition) -> float:
    """Share of flow that bypasses the porous walls through leak paths."""
    f = gcfg["flow"]
    crack_sev = max((c["severity"] for c in cond["cracks"]), default=0.0)
    bypass = (
        f["bypass_per_missing_plug"] * (cond["missing_plugs_inlet"] + cond["missing_plugs_outlet"])
        + f["bypass_per_crack_severity"] * crack_sev
        + f["bypass_internal_crack"] * cond["internal_crack_severity"]
        + f["bypass_melt"] * cond["melt_severity"]
    )
    return float(min(bypass, f["bypass_max"]))
