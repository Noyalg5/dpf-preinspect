"""Run folders: one per pipeline execution, holding provenance and outputs.

Layout::

    runs/<YYYYmmdd-HHMMSS>_<name>/
        config_resolved.yaml   the exact configuration used
        provenance.json        git hash, data-manifest hash, library versions
        metrics.json           metrics from every stage, keyed by stage
        models/  tables/  figures/  logs/
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from preinspect.config import Config, data_dir, dump_yaml, load_yaml, project_root, runs_dir
from preinspect.utils.hashing import sha256_file


def git_revision() -> dict[str, Any]:
    """Return the current commit hash and whether the tree has local changes."""

    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=project_root(), capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = _run("rev-parse", "HEAD") or "uncommitted"
    dirty = bool(_run("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit, "dirty": dirty}


def manifest_hash(cfg: Config) -> str | None:
    """Return the SHA-256 of the dataset manifest, or None if no data exists yet."""
    path = data_dir(cfg) / "manifest.json"
    return sha256_file(path) if path.exists() else None


@dataclass
class Run:
    """Handle to a run folder."""

    path: Path

    @property
    def name(self) -> str:
        """Folder name, e.g. ``20261023-101500_demo``."""
        return self.path.name

    def sub(self, *parts: str) -> Path:
        """Return a path inside the run, creating parent folders."""
        target = self.path.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def config(self) -> Config:
        """Configuration the run was created with."""
        return load_yaml(self.path / "config_resolved.yaml")

    def read_metrics(self) -> dict[str, Any]:
        """Return all metrics recorded so far."""
        path = self.path / "metrics.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def write_metrics(self, stage: str, metrics: dict[str, Any]) -> None:
        """Merge metrics for one stage into ``metrics.json``."""
        allm = self.read_metrics()
        allm[stage] = metrics
        (self.path / "metrics.json").write_text(json.dumps(allm, indent=2, default=_jsonable))

    def write_json(self, relpath: str, obj: Any) -> Path:
        """Write a JSON artefact inside the run and return its path."""
        target = self.sub(relpath)
        target.write_text(json.dumps(obj, indent=2, default=_jsonable))
        return target

    def read_json(self, relpath: str) -> Any:
        """Read a JSON artefact from the run."""
        return json.loads((self.path / relpath).read_text())


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def create_run(cfg: Config, name: str | None = None) -> Run:
    """Create a new run folder and record provenance.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    name : str, optional
        Suffix for the folder; defaults to the profile name.

    Returns
    -------
    Run
        Handle to the new folder.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = runs_dir(cfg) / f"{stamp}_{name or cfg['name']}"
    path.mkdir(parents=True, exist_ok=False)
    run = Run(path)
    dump_yaml(cfg, path / "config_resolved.yaml")
    import numpy
    import torch

    provenance = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_revision(),
        "data_manifest_sha256": manifest_hash(cfg),
        "dataset": cfg["dataset_name"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }
    run.write_json("provenance.json", provenance)
    return run


def latest_run(cfg: Config) -> Run:
    """Return the most recent run for the profile named in ``cfg``.

    Raises
    ------
    FileNotFoundError
        If no run exists for this profile.
    """
    candidates = sorted(runs_dir(cfg).glob(f"*_{cfg['name']}"))
    if not candidates:
        raise FileNotFoundError(
            f"No runs for profile '{cfg['name']}'. Run `preinspect train` first."
        )
    return Run(candidates[-1])


def open_run(cfg: Config, run: str | None) -> Run:
    """Open a named run folder, or the latest run for the profile when ``run`` is None."""
    if run is None:
        return latest_run(cfg)
    path = Path(run)
    if not path.exists():
        path = runs_dir(cfg) / run
    if not path.exists():
        raise FileNotFoundError(f"Run folder not found: {run}")
    return Run(path)
