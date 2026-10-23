"""Evaluation figures: reliability diagrams and modality contributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from preinspect.evaluate.metrics import reliability_table
from preinspect.evaluate.scoring import PROBA
from preinspect.report.style import INK, SERIES, save_figure


def reliability_figure(
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]], bins: int, path: Path
) -> Path:
    """Reliability diagrams (test split) before and after calibration, one panel per model."""
    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.8), squeeze=False)
    rows = []
    for ax, (name, (raw, cal)) in zip(axes[0], tables.items(), strict=True):
        ax.plot([0, 1], [0, 1], color=INK["muted"], linewidth=1, label="perfect calibration")
        for label, df, colour in (("before", raw, SERIES[1]), ("after", cal, SERIES[0])):
            rel = pd.DataFrame(
                reliability_table(df["outcome_true"].to_numpy(), df[PROBA].to_numpy(), bins)
            ).dropna()
            ax.plot(rel["confidence"], rel["accuracy"], marker="o", color=colour, label=label)
            rows += [{"model": name, "stage": label, **r} for r in rel.to_dict("records")]
        ax.set_title(name)
        ax.set_xlabel("Predicted confidence (top class)")
        ax.set_ylabel("Observed accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
    fig.suptitle("Reliability on the test split (synthetic data)", fontsize=10)
    return save_figure(fig, path, pd.DataFrame(rows))
