"""SHA-256 helpers used for data manifests, audit records and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of a byte string.

    Parameters
    ----------
    data : bytes
        Payload to hash.

    Returns
    -------
    str
        64-character hexadecimal digest.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, streamed in chunks.

    Parameters
    ----------
    path : str or Path
        File to hash.
    chunk_size : int
        Read size in bytes.

    Returns
    -------
    str
        64-character hexadecimal digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialise an object to JSON with sorted keys and no whitespace.

    Canonical form makes hashes independent of dict insertion order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON form of ``obj``."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))
