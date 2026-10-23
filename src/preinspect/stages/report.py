"""Stage ``report``: render result-bearing documents from data and run outputs."""

from __future__ import annotations

import json
import shutil

from preinspect.config import Config, data_dir, docs_dir, reports_dir
from preinspect.report.render import render_to_file
from preinspect.utils.logging import get_logger

log = get_logger(__name__)


def render_data_card(cfg: Config) -> None:
    """Render ``docs/data_card.md`` and copy the gallery into ``reports/figures``."""
    ddir = data_dir(cfg)
    stats = json.loads((ddir / "stats.json").read_text())
    figures = reports_dir(cfg) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    shutil.copy(ddir / "figures" / "gallery.png", figures / "gallery.png")
    shutil.copy(
        ddir / "validation" / "data_quality_report.md", reports_dir(cfg) / "data_quality_report.md"
    )
    render_to_file("data_card.md.j2", {"s": stats}, docs_dir(cfg) / "data_card.md")


def run(cfg: Config, run_name: str | None = None) -> None:
    """Render all documents for the profile in ``cfg``."""
    render_data_card(cfg)
    log.info("documents rendered")
