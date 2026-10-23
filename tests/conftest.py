"""Shared fixtures. Data and runs are written to temporary folders."""

from __future__ import annotations

from pathlib import Path

import pytest

from preinspect.config import Config, load_config


def _tiny(root: Path) -> Config:
    return load_config(
        "tiny",
        overrides={
            "paths": {
                "data_root": str(root / "data"),
                "runs_root": str(root / "runs"),
                "reports_root": str(root / "reports"),
                "docs_root": str(root / "docs"),
            }
        },
    )


@pytest.fixture()
def tiny_cfg(tmp_path: Path) -> Config:
    """Tiny profile redirected to a per-test temporary folder."""
    return _tiny(tmp_path)


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Tiny profile with a generated and validated dataset, shared by the session."""
    from preinspect.stages import data as data_stage

    cfg = _tiny(tmp_path_factory.mktemp("shared"))
    data_stage.run(cfg)
    return cfg
