"""Shared fixtures: every test writes into a temporary project tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from preinspect.config import Config, load_config


@pytest.fixture()
def tiny_cfg(tmp_path: Path) -> Config:
    """Tiny profile with data, runs and reports redirected to a temp folder."""
    return load_config(
        "tiny",
        overrides={
            "paths": {
                "data_root": str(tmp_path / "data"),
                "runs_root": str(tmp_path / "runs"),
                "reports_root": str(tmp_path / "reports"),
            }
        },
    )
