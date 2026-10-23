"""Audit trail: append-only behaviour and tamper detection."""

from __future__ import annotations

import sqlite3

import pytest

from preinspect.service.audit import AuditTrail

RESULT = {
    "unit_id": "DPF-1",
    "route": "PROCEED",
    "reason": "proceed",
    "probabilities": {"PASS": 0.9, "REWORK": 0.08, "SCRAP": 0.02},
    "prediction_set": ["PASS"],
    "model_version": "run-x",
    "data_version": "abc",
}


def _trail(tmp_path) -> AuditTrail:
    trail = AuditTrail(tmp_path / "audit.sqlite")
    first = trail.log_inspection(RESULT, "f" * 64)
    trail.log_operator(first, "DPF-1", "PROCEED", True, None, "op1")
    trail.log_inspection({**RESULT, "unit_id": "DPF-2", "route": "REFER"}, "e" * 64)
    return trail


def test_clean_chain_verifies(tmp_path) -> None:
    report = _trail(tmp_path).verify()
    assert report.ok and report.n_records == 3


def test_update_and_delete_are_rejected(tmp_path) -> None:
    trail = _trail(tmp_path)
    with pytest.raises(sqlite3.DatabaseError):
        trail.conn.execute("UPDATE records SET unit_id = 'X' WHERE id = 1")
    with pytest.raises(sqlite3.DatabaseError):
        trail.conn.execute("DELETE FROM records WHERE id = 2")


def test_tampering_is_detected(tmp_path) -> None:
    trail = _trail(tmp_path)
    trail.conn.executescript("DROP TRIGGER records_no_update;")
    trail.conn.execute(
        "UPDATE records SET payload = replace(payload, 'PROCEED', 'QUARANTINE') WHERE id = 1"
    )
    trail.conn.commit()
    report = AuditTrail(tmp_path / "audit.sqlite").verify()
    assert not report.ok and report.first_bad_id == 1
