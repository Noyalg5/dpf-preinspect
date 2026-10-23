"""Stage ``data``: generate the synthetic dataset, validate it, summarise it."""

from __future__ import annotations

import json
import time

from preinspect.config import Config, data_dir
from preinspect.data.card import dataset_stats, gallery
from preinspect.data.validate import validate_dataset
from preinspect.synth.generate import generate_dataset
from preinspect.utils.logging import get_logger

log = get_logger(__name__)


def run(cfg: Config) -> None:
    """Generate, validate and summarise the dataset for ``cfg``."""
    t0 = time.perf_counter()
    generate_dataset(cfg)
    result = validate_dataset(cfg)
    if not result.integrity_ok:
        raise SystemExit("data failed integrity checks; see validation/data_quality_report.md")
    dataset_stats(cfg)
    gallery(cfg, data_dir(cfg) / "figures" / "gallery.png")
    seconds = time.perf_counter() - t0
    (data_dir(cfg) / "logs" / "stage.json").write_text(json.dumps({"seconds": round(seconds, 1)}))
    log.info("data stage finished in %.1f s", seconds)
