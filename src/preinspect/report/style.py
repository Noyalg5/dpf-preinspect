"""Shared figure style for every chart in the project.

One categorical palette in a fixed order (validated for colour-vision
deficiency on adjacent pairs; the first three slots validate on all pairs),
thin marks, recessive hairline grids and ink-coloured text. Each figure is
written together with a CSV of its data so the numbers are available as a
table, not only as colour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

OUTCOME_COLOURS = dict(zip(["PASS", "REWORK", "SCRAP"], SERIES[:3], strict=True))
ROUTE_COLOURS = dict(
    zip(["PROCEED", "EXTENDED_CLEAN", "QUARANTINE", "REFER"], SERIES[:4], strict=True)
)


def apply_style() -> None:
    """Set matplotlib defaults to the project style."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK["primary"],
            "axes.labelcolor": INK["secondary"],
            "axes.labelsize": 9,
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.markersize": 6,
            "font.family": "sans-serif",
            "text.color": INK["primary"],
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: Any, path: str | Path, data: pd.DataFrame | None = None) -> Path:
    """Save a figure as PNG and, if given, its data as a CSV next to it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    plt.close(fig)
    if data is not None:
        data.to_csv(target.with_suffix(".csv"), index=False)
    return target


apply_style()
