"""Interface to the manufacturing execution system (MES) or line PLC.

The service publishes each unit's route to the line so that conveyors and
work instructions follow it. A real deployment would implement
:class:`MesAdapter` over OPC UA or MQTT (see ``docs/system_design.md`` for
the tag and topic map); :class:`MockMes` keeps a route queue in SQLite so
the behaviour can be demonstrated and tested without plant systems.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd


class MesAdapter(Protocol):
    """What the inspection service needs from the line control system."""

    def publish_route(self, unit_id: str, route: str, inspection_id: int, source: str) -> None:
        """Send the route chosen for a unit (by the system or an operator)."""

    def queue(self) -> pd.DataFrame:
        """Current route assignment per unit."""


class MockMes:
    """Route queue table standing in for the MES."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS route_queue (unit_id TEXT PRIMARY KEY, route TEXT, "
            "inspection_id INTEGER, source TEXT, status TEXT, updated_utc TEXT)"
        )

    def publish_route(self, unit_id: str, route: str, inspection_id: int, source: str) -> None:
        """Insert or update the unit's route; operator decisions replace system ones."""
        status = "awaiting_operator" if route in ("REFER", "QUARANTINE") else "released"
        if source == "operator":
            status = "confirmed"
        with self.conn:
            self.conn.execute(
                "INSERT INTO route_queue VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(unit_id) DO UPDATE "
                "SET route=excluded.route, inspection_id=excluded.inspection_id, "
                "source=excluded.source, status=excluded.status, updated_utc=excluded.updated_utc",
                (
                    unit_id,
                    route,
                    inspection_id,
                    source,
                    status,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )

    def queue(self) -> pd.DataFrame:
        """All units in the queue, most recent first."""
        return pd.read_sql_query("SELECT * FROM route_queue ORDER BY updated_utc DESC", self.conn)
