"""Append-only audit trail with a SHA-256 hash chain.

Every inspection and every operator decision is a row in ``records``. Each
row stores the hash of the previous row, and its own hash covers its
content plus that previous hash, so changing, deleting or reordering any
row breaks the chain from that point on. SQLite triggers reject UPDATE and
DELETE; the chain catches tampering that bypasses them (for example editing
the database file with another tool).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from preinspect.utils.hashing import canonical_json, sha256_bytes

GENESIS = "0" * 64
FIELDS = (
    "id",
    "created_utc",
    "record_type",
    "unit_id",
    "payload",
    "model_version",
    "data_version",
    "code_version",
    "prev_hash",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    record_type TEXT NOT NULL
        CHECK (record_type IN ('inspection', 'operator_decision', 'downstream_result')),
    unit_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    model_version TEXT,
    data_version TEXT,
    code_version TEXT,
    prev_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS records_no_update BEFORE UPDATE ON records
BEGIN SELECT RAISE(ABORT, 'audit records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS records_no_delete BEFORE DELETE ON records
BEGIN SELECT RAISE(ABORT, 'audit records are append-only'); END;
"""


def record_hash(row: dict[str, Any]) -> str:
    """Hash of a record's content fields (including the previous hash)."""
    return sha256_bytes(canonical_json({k: row[k] for k in FIELDS}).encode("utf-8"))


@dataclass
class VerifyReport:
    """Result of a chain verification."""

    ok: bool
    n_records: int
    first_bad_id: int | None
    message: str

    def summary(self) -> str:
        """One-line summary for the command line."""
        state = "OK" if self.ok else "TAMPERING DETECTED"
        return f"audit chain {state}: {self.n_records} records. {self.message}"


class AuditTrail:
    """SQLite-backed, hash-chained log of inspections and operator decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def _last_hash(self) -> str:
        row = self.conn.execute(
            "SELECT record_hash FROM records ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS

    def append(
        self,
        record_type: str,
        unit_id: str,
        payload: dict[str, Any],
        versions: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        """Append a record and return its id and hash."""
        versions = versions or {}
        with self.conn:
            next_id = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM records").fetchone()[
                0
            ]
            row = {
                "id": next_id,
                "created_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "record_type": record_type,
                "unit_id": unit_id,
                "payload": canonical_json(payload),
                "model_version": versions.get("model_version"),
                "data_version": versions.get("data_version"),
                "code_version": versions.get("code_version"),
                "prev_hash": self._last_hash(),
            }
            row["record_hash"] = record_hash(row)
            self.conn.execute(
                f"INSERT INTO records ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                list(row.values()),
            )
        return next_id, row["record_hash"]

    def log_inspection(self, result: dict[str, Any], input_hash: str) -> int:
        """Record an inspection result (without image maps) and the hash of its inputs."""
        keep = (
            "route",
            "reason",
            "probabilities",
            "prediction_set",
            "dp_ratio",
            "anomaly",
            "rationale",
            "valid",
            "input_issues",
            "monitoring",
        )
        payload = {"input_sha256": input_hash, **{k: result[k] for k in keep if k in result}}
        return self.append("inspection", result["unit_id"], payload, result)[0]

    def log_operator(
        self,
        inspection_id: int,
        unit_id: str,
        decision: str,
        accepted: bool,
        reason: str | None,
        operator: str,
    ) -> int:
        """Record an operator's acceptance or override of an inspection."""
        payload = {
            "inspection_id": inspection_id,
            "decision": decision,
            "accepted": accepted,
            "override_reason": reason,
            "operator": operator,
        }
        return self.append("operator_decision", unit_id, payload)[0]

    def log_result(self, unit_id: str, disposition: str, dp_ratio: float | None) -> int:
        """Record the downstream test result once the unit has been cleaned and tested."""
        payload = {"disposition": disposition, "dp_ratio_post_clean": dp_ratio}
        return self.append("downstream_result", unit_id, payload)[0]

    def records(self, limit: int | None = None) -> pd.DataFrame:
        """All records (or the latest ``limit``) as a table with decoded payloads."""
        sql = "SELECT * FROM records ORDER BY id"
        df = pd.read_sql_query(sql, self.conn)
        if limit is not None:
            df = df.tail(limit)
        df["payload"] = df["payload"].map(json.loads)
        return df

    def verify(self) -> VerifyReport:
        """Recompute the chain and check the append-only triggers are in place."""
        cur = self.conn.execute(f"SELECT {', '.join(FIELDS)}, record_hash FROM records ORDER BY id")
        prev = GENESIS
        n = 0
        for values in cur:
            row = dict(zip((*FIELDS, "record_hash"), values, strict=True))
            n += 1
            if row["prev_hash"] != prev:
                return VerifyReport(False, n, row["id"], f"record {row['id']}: broken link")
            if record_hash(row) != row["record_hash"]:
                return VerifyReport(False, n, row["id"], f"record {row['id']}: content altered")
            prev = row["record_hash"]
        triggers = {
            r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        }
        if not {"records_no_update", "records_no_delete"} <= triggers:
            return VerifyReport(False, n, None, "append-only triggers have been removed")
        return VerifyReport(True, n, None, "every hash and link verified")
