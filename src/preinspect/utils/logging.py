"""Logging configuration shared by the CLI, training and service."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the root logger once, optionally mirroring to a file.

    Parameters
    ----------
    level : str
        Logging level name.
    log_file : str or Path, optional
        If given, log records are also appended to this file.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        root.addHandler(stream)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve()
            for h in root.handlers
        ):
            handler = logging.FileHandler(path)
            handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(handler)
    for noisy in ("matplotlib", "PIL", "numba", "shap", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
