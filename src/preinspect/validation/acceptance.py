"""Acceptance criteria AC-01 to AC-07 evaluated against measured values."""

from __future__ import annotations

import operator
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from preinspect.service.audit import AuditTrail
from preinspect.utils.hashing import sha256_json

OPS = {"<=": operator.le, ">=": operator.ge, "==": operator.eq, "<": operator.lt, ">": operator.gt}


def evaluate_criteria(
    acfg: dict[str, Any], values: dict[str, Any], evidence: dict[str, str]
) -> list[dict[str, Any]]:
    """PASS, FAIL or NOT MEASURED for each criterion."""
    rows = []
    for cid, c in acfg["criteria"].items():
        value = values.get(c["metric"])
        if value is None:
            status = "NOT MEASURED"
        else:
            status = "PASS" if OPS[c["op"]](value, c["target"]) else "FAIL"
        rows.append(
            {
                "id": cid,
                "name": c["name"],
                "description": c["description"],
                "metric": c["metric"],
                "op": c["op"],
                "target": c["target"],
                "value": value,
                "status": status,
                "evidence": evidence.get(cid, ""),
            }
        )
    return rows


def audit_integrity_check(decisions: pd.DataFrame, n: int = 20) -> dict[str, Any]:
    """Log decisions to a fresh audit trail, verify it, tamper with it, verify again."""
    with tempfile.TemporaryDirectory() as tmp:
        trail = AuditTrail(Path(tmp) / "audit.sqlite")
        for row in decisions.head(n).to_dict("records"):
            result = {
                "unit_id": row["unit_id"],
                "route": row["route"],
                "reason": row["reason"],
                "probabilities": {k[2:]: row[k] for k in row if k.startswith("p_")},
                "prediction_set": row["prediction_set"].split("|"),
            }
            trail.log_inspection(result, sha256_json(result))
        clean = trail.verify()
        trail.conn.executescript("DROP TRIGGER records_no_update;")
        trail.conn.execute(
            "UPDATE records SET payload = replace(payload, 'REFER', 'PROCEED') WHERE id = 1"
        )
        trail.conn.execute("UPDATE records SET unit_id = unit_id || 'X' WHERE id = 2")
        trail.conn.commit()
        tampered = trail.verify()
        trail.conn.close()
    return {
        "clean_ok": clean.ok,
        "records": clean.n_records,
        "tamper_detected": not tampered.ok,
        "tamper_message": tampered.message,
        "audit_integrity_ok": bool(clean.ok and not tampered.ok),
    }
