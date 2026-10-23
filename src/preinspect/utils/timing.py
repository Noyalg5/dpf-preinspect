"""Wall-clock timing of pipeline steps."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from preinspect.utils.logging import get_logger

log = get_logger(__name__)


@contextmanager
def timed(store: dict[str, float], name: str) -> Iterator[None]:
    """Record the duration of a block, in seconds, under ``store[name]``."""
    start = time.perf_counter()
    yield
    store[name] = round(time.perf_counter() - start, 1)
    log.info("%s finished in %.1f s", name, store[name])
