"""Jinja2 rendering of result-bearing documents.

Numbers in reports, docs and the README are never typed by hand: they are
passed in from run outputs and formatted by the filters defined here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100.0 * float(value):.{digits}f} %"


def _pm(values: Any, digits: int = 3) -> str:
    """Format a ``{"mean": m, "std": s}`` mapping as ``m ± s``."""
    if isinstance(values, dict) and "mean" in values:
        if values.get("std") is None or values.get("n", 2) < 2:
            return _fmt(values["mean"], digits)
        return f"{values['mean']:.{digits}f} ± {values['std']:.{digits}f}"
    return _fmt(values, digits)


def environment() -> Environment:
    """Jinja2 environment with the project's number-formatting filters."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["fmt"] = _fmt
    env.filters["pct"] = _pct
    env.filters["pm"] = _pm
    return env


def render_template(name: str, context: dict[str, Any]) -> str:
    """Render ``templates/<name>`` with ``context``."""
    return environment().get_template(name).render(**context)


def render_to_file(name: str, context: dict[str, Any], path: str | Path) -> Path:
    """Render a template and write it to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_template(name, context), encoding="utf-8")
    return target
