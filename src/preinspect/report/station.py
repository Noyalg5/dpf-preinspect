"""Imaging-station sizing: resolution budget and a schematic of the concept."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from preinspect.report.style import INK, SERIES, save_figure


def resolution_budget(scfg: dict[str, Any]) -> pd.DataFrame:
    """Pixels per cell pitch, wall and narrowest crack for each candidate sensor.

    Cell pitch of a real filter is ``25.4 / sqrt(cpsi)`` mm. Tiles are the
    number of model-sized crops needed to cover the field at full resolution.
    """
    rows = []
    for sensor in scfg["sensors"]:
        mm_per_px = scfg["fov_mm"] / sensor["px_short"]
        stride = scfg["tile_px"] - scfg["tile_overlap_px"]
        tiles = math.ceil((sensor["px_short"] - scfg["tile_overlap_px"]) / stride) ** 2
        row = {
            "sensor": sensor["name"],
            "px_across_field": sensor["px_short"],
            "mm_per_px": mm_per_px,
            "px_per_wall": scfg["wall_mm"] / mm_per_px,
            "px_per_crack": scfg["min_crack_mm"] / mm_per_px,
            "tiles_per_face": tiles,
        }
        for cpsi in scfg["cpsi"]:
            row[f"px_per_pitch_{cpsi}cpsi"] = 25.4 / math.sqrt(cpsi) / mm_per_px
        row["walls_resolved"] = row["px_per_wall"] >= scfg["min_px_per_wall"]
        row["crack_front_light"] = row["px_per_crack"] >= scfg["min_px_per_crack_front"]
        row["crack_backlight"] = row["px_per_crack"] >= scfg["min_px_per_crack_backlight"]
        rows.append(row)
    return pd.DataFrame(rows)


def station_schematic(path: Path) -> Path:
    """Side view of the proposed station: camera, lights, cradle and backlight."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, colour, label, text_dx=0.15):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02",
                linewidth=1,
                edgecolor=INK["secondary"],
                facecolor=colour,
            )
        )
        ax.text(x + w + text_dx, y + h / 2, label, va="center", fontsize=8, color=INK["primary"])

    box(
        4.3,
        8.6,
        1.4,
        0.9,
        SERIES[0],
        "Area-scan camera (20 MP class) with telecentric or\n"
        "low-distortion lens, fixed working distance",
    )
    box(3.3, 7.4, 3.4, 0.5, SERIES[3], "Diffuse dome / low-angle ring light (switchable)")
    box(3.0, 3.9, 4.0, 3.0, "#d9d6cc", "Used DPF in its can\n(inlet face up)")
    ax.add_patch(Rectangle((2.6, 3.3), 4.8, 0.5, facecolor=SERIES[2], edgecolor=INK["secondary"]))
    ax.text(
        7.55,
        3.55,
        "Cradle on turntable: repeatable seating,\nflip for the outlet face",
        va="center",
        fontsize=8,
    )
    box(
        3.3,
        2.2,
        3.4,
        0.5,
        SERIES[4],
        "Backlight panel: light leaks through cracks\nand channels missing a plug",
    )
    box(0.2, 0.4, 3.0, 1.1, "#ece9e2", "Load cell (weight gain)\nflow bench port", text_dx=-2.9)
    ax.annotate(
        "",
        xy=(5.0, 7.0),
        xytext=(5.0, 8.55),
        arrowprops={"arrowstyle": "-", "color": INK["muted"], "linestyle": "--"},
    )
    ax.text(5.1, 8.1, "field ~380 mm", fontsize=7, color=INK["secondary"])
    ax.set_title("Pre-inspection station concept (not to scale)", fontsize=10)
    return save_figure(fig, path)
