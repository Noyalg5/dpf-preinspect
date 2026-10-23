"""Configuration loading.

Profiles (``tiny``, ``demo``, ``full``) are YAML files that ``extends`` a
base file and override only what differs. Auxiliary files (``costs``,
``acceptance``, ``line_sim``) are loaded separately by name.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def project_root() -> Path:
    """Return the repository root.

    ``PREINSPECT_ROOT`` takes precedence; otherwise the current directory is
    used if it holds a ``configs`` folder, else the source checkout root.
    """
    env = os.environ.get("PREINSPECT_ROOT")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd()
    if (cwd / "configs").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Return the directory holding YAML configuration files."""
    return project_root() / "configs"


def deep_merge(base: Config, override: Config) -> Config:
    """Recursively merge ``override`` into a copy of ``base``.

    Parameters
    ----------
    base, override : dict
        Mappings to merge; nested dicts merge, everything else is replaced.

    Returns
    -------
    dict
        New merged mapping.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return path
    candidate = config_dir() / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No config named '{name_or_path}' (looked for {candidate})")


def load_yaml(path: str | Path) -> Config:
    """Read a YAML mapping from disk."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(name_or_path: str | Path, overrides: Config | None = None) -> Config:
    """Load a profile config, following ``extends`` chains.

    Parameters
    ----------
    name_or_path : str or Path
        Profile name (e.g. ``"demo"``) or path to a YAML file.
    overrides : dict, optional
        Extra values merged last (used by tests).

    Returns
    -------
    dict
        Fully resolved configuration.
    """
    path = _resolve_path(name_or_path)
    raw = load_yaml(path)
    parent = raw.pop("extends", None)
    cfg = deep_merge(load_config(parent), raw) if parent else raw
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def load_aux(name: str) -> Config:
    """Load an auxiliary config file (``costs``, ``acceptance``, ``line_sim``)."""
    return load_yaml(_resolve_path(name))


def dump_yaml(cfg: Config, path: str | Path) -> None:
    """Write a mapping to YAML with stable key order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def data_dir(cfg: Config) -> Path:
    """Return the dataset folder for a profile, e.g. ``data/demo``."""
    root = Path(cfg["paths"]["data_root"])
    if not root.is_absolute():
        root = project_root() / root
    return root / cfg["dataset_name"]


def runs_dir(cfg: Config) -> Path:
    """Return the root folder for run outputs."""
    root = Path(cfg["paths"]["runs_root"])
    return root if root.is_absolute() else project_root() / root


def reports_dir(cfg: Config) -> Path:
    """Return the folder that rendered reports are written to."""
    root = Path(cfg["paths"]["reports_root"])
    return root if root.is_absolute() else project_root() / root


def docs_dir(cfg: Config) -> Path:
    """Return the folder that rendered documentation is written to."""
    root = Path(cfg["paths"]["docs_root"])
    return root if root.is_absolute() else project_root() / root
