"""Stage ``report``: render every result-bearing document from data and run outputs."""

from __future__ import annotations

import json
import shutil

from preinspect.config import Config, data_dir, docs_dir, reports_dir
from preinspect.report.context import build_context
from preinspect.report.render import render_to_file
from preinspect.report.station import station_schematic
from preinspect.runs import Run, open_run
from preinspect.utils.logging import get_logger

log = get_logger(__name__)
RUN_FIGURES = (
    "reliability.png",
    "cost_curve.png",
    "shap_global.png",
    "gradcam_examples.png",
    "deletion_insertion.png",
    "robustness.png",
    "doe_effects.png",
    "doe_interaction.png",
    "line_kpis.png",
    "tornado.png",
)
DOCS = (
    "technical_report",
    "model_card",
    "faq",
    "training_manual",
    "system_design",
    "demo_script",
    "real_data_plan",
)


def render_data_card(cfg: Config) -> None:
    """Render ``docs/data_card.md`` and copy the gallery and data-quality report."""
    ddir = data_dir(cfg)
    stats = json.loads((ddir / "stats.json").read_text())
    figures = reports_dir(cfg) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    shutil.copy(ddir / "figures" / "gallery.png", figures / "gallery.png")
    shutil.copy(
        ddir / "validation" / "data_quality_report.md", reports_dir(cfg) / "data_quality_report.md"
    )
    render_to_file("data_card.md.j2", {"s": stats}, docs_dir(cfg) / "data_card.md")


def copy_figures(cfg: Config, run: Run) -> None:
    """Copy the run's figures (and their CSV data) next to the rendered reports."""
    target = reports_dir(cfg) / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for name in RUN_FIGURES:
        src = run.path / "figures" / name
        for path in (src, src.with_suffix(".csv")):
            if path.exists():
                shutil.copy(path, target / path.name)
    station_schematic(target / "station.png")


def run(cfg: Config, run_name: str | None = None) -> Run:
    """Render the data card, every doc template and the README for the profile's run."""
    run_ = open_run(cfg, run_name)
    render_data_card(cfg)
    copy_figures(cfg, run_)
    context = build_context(cfg, run_)
    for name in DOCS:
        render_to_file(f"{name}.md.j2", context, docs_dir(cfg) / f"{name}.md")
    render_to_file("README.md.j2", context, docs_dir(cfg).parent / "README.md")
    log.info("rendered README, data card and %d documents from %s", len(DOCS), run_.name)
    return run_
