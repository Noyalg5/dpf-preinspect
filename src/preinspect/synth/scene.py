"""Scene sampler: turns a unit's latent condition into face geometry.

All coordinates are in the substrate frame ``(u, v)``, where the unit disc is
the substrate's outer edge. The outlet face is seen from the opposite end,
so a feature that passes through the substrate appears mirrored (``u -> -u``)
on the outlet view.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from preinspect.utils.seeding import make_rng

VIEW_PARITY = {"inlet": 0, "outlet": 1}


def stylised_cells(gcfg: dict[str, Any], meta: dict[str, Any]) -> int:
    """Cells across the diameter in the stylised rendering (about 30 to 50)."""
    sub = gcfg["substrate"]
    scale = meta["diameter_mm"] / sub["reference_diameter_mm"]
    return int(
        round(sub["stylised_cells_per_sqrt_cpsi"] * math.sqrt(meta["cell_density_cpsi"]) * scale)
    )


def _harmonics(rng: np.random.Generator, geo: dict[str, Any]) -> list[list[float]]:
    amp = geo["shape_harmonic_amp"]
    return [
        [float(rng.uniform(0, amp) / (k + 1)), float(rng.uniform(0, 2 * math.pi))]
        for k in range(geo["shape_harmonics"])
    ]


def _point_in_disc(rng: np.random.Generator, r_max: float) -> list[float]:
    r = r_max * math.sqrt(rng.random())
    t = rng.uniform(0, 2 * math.pi)
    return [r * math.cos(t), r * math.sin(t)]


def _crack_points(rng: np.random.Generator, geo: dict[str, Any], kind: str) -> list[list[float]]:
    if kind == "radial":
        theta = rng.uniform(0, 2 * math.pi)
        radii = np.linspace(
            rng.uniform(*geo["radial_start_r"]),
            rng.uniform(*geo["radial_end_r"]),
            geo["radial_points"],
        )
        angles = theta + np.cumsum(rng.normal(0, geo["radial_angle_walk_sd"], radii.size))
    else:
        start = rng.uniform(0, 2 * math.pi)
        angles = start + np.linspace(0, rng.uniform(*geo["ring_span_rad"]), geo["ring_points"])
        radii = rng.uniform(*geo["ring_radius"]) + np.cumsum(
            rng.normal(0, geo["ring_radius_walk_sd"], angles.size)
        )
    return [
        [float(r * math.cos(a)), float(r * math.sin(a))] for r, a in zip(radii, angles, strict=True)
    ]


def _mirror(points: list[list[float]]) -> list[list[float]]:
    return [[-u, v] for u, v in points]


def _missing_plug_cells(
    rng: np.random.Generator, geo: dict[str, Any], n_cells: int, view: str, count: int
) -> tuple[list[list[int]], list[float]]:
    pitch = 2.0 / n_cells
    centre = _point_in_disc(rng, geo["plug_cluster_centre_r"])
    clustered = rng.random() < geo["plug_cluster_share"]
    ic, jc = int((centre[0] + 1) / pitch), int((centre[1] + 1) / pitch)
    span = geo["plug_cluster_radius_cells"] if clustered else n_cells
    cells: set[tuple[int, int]] = set()
    for _ in range(count * 50):
        if len(cells) >= count:
            break
        i, j = ic + int(rng.integers(-span, span + 1)), jc + int(rng.integers(-span, span + 1))
        u, v = (i + 0.5) * pitch - 1, (j + 0.5) * pitch - 1
        plugged = (i + j + VIEW_PARITY[view]) % 2 == 0
        if plugged and math.hypot(u, v) < geo["plug_max_r"]:
            cells.add((i, j))
    return [list(c) for c in sorted(cells)], centre


def _cracks(rng, geo, cond, pitch) -> list[dict[str, Any]]:
    out = []
    lo, hi = geo["crack_width_pitch"]
    for crack in cond["cracks"]:
        points = _crack_points(rng, geo, crack["type"])
        base = {
            "cls": "crack",
            "subtype": crack["type"],
            "severity": crack["severity"],
            "width": (lo + (hi - lo) * crack["severity"]) * pitch,
        }
        if crack["through"]:
            out.append({**base, "view": "inlet", "points": points})
            out.append({**base, "view": "outlet", "points": _mirror(points)})
        else:
            out.append({**base, "view": str(rng.choice(["inlet", "outlet"])), "points": points})
    return out


def sample_scene(
    gcfg: dict[str, Any], meta: dict[str, Any], cond: dict[str, Any], seed: int
) -> dict:
    """Sample defect and nuisance geometry consistent with the latent condition.

    Parameters
    ----------
    gcfg : dict
        Generator configuration.
    meta, cond : dict
        Unit metadata and latent condition.
    seed : int
        Global seed.

    Returns
    -------
    dict
        JSON-serialisable scene description used by the renderer.
    """
    rng = make_rng(seed, "scene", meta["unit_id"])
    geo, app = gcfg["geometry"], gcfg["appearance"]
    n_cells = stylised_cells(gcfg, meta)
    pitch = 2.0 / n_cells
    film_base, film_per = app["inlet_soot_film"]
    defects: list[dict[str, Any]] = _cracks(rng, geo, cond, pitch)
    leak_sites: dict[str, list[float]] = {}
    for crack in defects:
        leak_sites.setdefault(crack["view"], crack["points"][len(crack["points"]) // 2])

    for view in ("inlet", "outlet"):
        count = cond[f"missing_plugs_{view}"]
        if count:
            cells, centre = _missing_plug_cells(rng, geo, n_cells, view, count)
            defects.append({"cls": "missing_plug", "view": view, "cells": cells})
            leak_sites.setdefault(view, centre)
    if cond["chip_severity"] > 0:
        s = cond["chip_severity"]
        lo_w, hi_w = geo["chip_half_width_rad"]
        lo_d, hi_d = geo["chip_depth"]
        defects.append(
            {
                "cls": "chip",
                "view": str(rng.choice(["inlet", "outlet"])),
                "severity": s,
                "angle": float(rng.uniform(0, 2 * math.pi)),
                "half_width": lo_w + (hi_w - lo_w) * s,
                "depth": lo_d + (hi_d - lo_d) * s,
                "harm": _harmonics(rng, geo),
            }
        )
    if cond["face_plug_extent"] > 0:
        radius = math.sqrt(cond["face_plug_extent"])
        defects.append(
            {
                "cls": "face_plugging",
                "view": "inlet",
                "deposit": cond["face_plug_deposit"],
                "centre": _point_in_disc(rng, max(0.0, geo["face_plug_centre_max_r"] - radius)),
                "radius": radius,
                "harm": _harmonics(rng, geo),
            }
        )
    if cond["stain"] != "none":
        view = "inlet" if rng.random() < geo["stain_inlet_share"] else "outlet"
        defects.append(
            {
                "cls": "stain",
                "view": view,
                "fluid": cond["stain"],
                "centre": _point_in_disc(rng, geo["stain_centre_max_r"]),
                "radius": float(rng.uniform(*geo["stain_radius"])),
                "opacity": float(rng.uniform(*geo["stain_opacity"])),
                "harm": _harmonics(rng, geo),
            }
        )
    if cond["melt_severity"] > 0:
        view = "outlet" if rng.random() < geo["melt_outlet_share"] else "inlet"
        lo, hi = geo["melt_radius"]
        defects.append(
            {
                "cls": "melt",
                "view": view,
                "strength": cond["melt_severity"],
                "centre": _point_in_disc(rng, geo["melt_centre_max_r"]),
                "radius": lo + (hi - lo) * cond["melt_severity"],
            }
        )
    if cond["outlet_soot_extent"] > 0:
        if "outlet" in leak_sites:
            centre = leak_sites["outlet"]
        elif "inlet" in leak_sites:
            centre = _mirror([leak_sites["inlet"]])[0]
        else:
            centre = _point_in_disc(rng, geo["outlet_soot_centre_max_r"])
        e = cond["outlet_soot_extent"]
        lo_r, hi_r = geo["outlet_soot_radius"]
        lo_d, hi_d = geo["outlet_soot_density"]
        defects.append(
            {
                "cls": "outlet_soot",
                "view": "outlet",
                "centre": [float(c) for c in centre],
                "radius": lo_r + (hi_r - lo_r) * e,
                "density": lo_d + (hi_d - lo_d) * e,
                "harm": _harmonics(rng, geo),
            }
        )
    if cond["debris_count"]:
        view = "inlet" if rng.random() < geo["debris_inlet_share"] else "outlet"
        items = [
            {
                "centre": _point_in_disc(rng, geo["debris_centre_max_r"]),
                "size": float(rng.uniform(*geo["debris_size"])),
                "kind": str(rng.choice(["metal", "fibre"])),
                "harm": _harmonics(rng, geo),
                "seed": int(rng.integers(0, 2**31)),
            }
            for _ in range(cond["debris_count"])
        ]
        defects.append({"cls": "debris", "view": view, "items": items})

    nuisances: list[dict[str, Any]] = []
    for view in ("inlet", "outlet"):
        if rng.random() < gcfg["nuisance"]["dust"]:
            nuisances.append(
                {
                    "kind": "dust",
                    "view": view,
                    "count": int(rng.integers(*geo["dust_count"])),
                    "seed": int(rng.integers(0, 2**31)),
                }
            )
        if rng.random() < gcfg["nuisance"]["marker"]:
            on_face = rng.random() < geo["marker_on_face_share"]
            nuisances.append(
                {
                    "kind": "marker",
                    "view": view,
                    "angle": float(rng.uniform(0, 2 * math.pi)),
                    "radius": geo["marker_radius_face" if on_face else "marker_radius_can"],
                    "ink": str(rng.choice(["ink_blue", "ink_black"])),
                    "seed": int(rng.integers(0, 2**31)),
                }
            )
    return {
        "unit_id": meta["unit_id"],
        "material": meta["material"],
        "diameter_mm": meta["diameter_mm"],
        "n_cells": n_cells,
        "segment_cells": gcfg["substrate"]["sic_segment_cells"] if meta["material"] == "sic" else 0,
        "soot_film": {
            "inlet": min(app["inlet_soot_film_max"], film_base + film_per * cond["soot_load_gL"]),
            "outlet": float(rng.uniform(0, app["outlet_soot_film_max"])),
        },
        "texture_seed": int(rng.integers(0, 2**31)),
        "defects": defects,
        "nuisances": nuisances,
    }
