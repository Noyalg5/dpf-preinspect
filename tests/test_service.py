"""Inspection API end to end on the tiny run: inspect, decide, audit, monitor."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from preinspect.service.api import create_app

ROUTES = {"PROCEED", "EXTENDED_CLEAN", "QUARANTINE", "REFER"}


@pytest.fixture()
def client(tiny_data, tiny_run, tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("PREINSPECT_AUDIT_DB", str(tmp_path / "audit.sqlite"))
    monkeypatch.setenv("PREINSPECT_MES_DB", str(tmp_path / "mes.sqlite"))
    return TestClient(create_app(tiny_data, tiny_run))


def _sample(client: TestClient) -> tuple[str, dict, dict]:
    uid = client.get("/samples", params={"limit": 1}).json()[0]
    sample = client.get(f"/samples/{uid}").json()
    files = {
        v: (f"{v}.jpg", base64.b64decode(sample["images_jpeg_base64"][v]), "image/jpeg")
        for v in ("inlet", "outlet")
    }
    return uid, sample["request"], files


def test_inspect_decide_and_audit(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert "policy" in client.get("/model-info").json()
    uid, req, files = _sample(client)
    resp = client.post(
        "/inspect", data={"request": json.dumps({**req, "include_maps": True})}, files=files
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["route"] in ROUTES and len(result["rationale"]) > 20
    assert set(result["maps"]) == {"inlet", "outlet"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3
    override = {
        "inspection_id": result["inspection_id"],
        "unit_id": uid,
        "accepted": False,
        "route": "REFER",
        "operator": "op1",
    }
    assert client.post("/decision", json=override).status_code == 422
    ok = client.post("/decision", json={**override, "reason": "visible crack near the skin"})
    assert ok.status_code == 200
    client.post("/result", json={"unit_id": uid, "disposition": "PASS"}).raise_for_status()
    assert len(client.get("/records").json()) == 3
    assert client.get("/audit/verify").json()["ok"]
    assert client.get("/monitoring").json()["inspections"] == 1
    queue = client.get("/mes/queue").json()
    assert queue[0]["unit_id"] == uid and queue[0]["status"] == "confirmed"


def test_bad_requests_get_actionable_errors(client: TestClient) -> None:
    uid, req, files = _sample(client)
    bad = {**req, "flow": {"q_m3s": [0.1, 0.2, 0.3], "dp_kpa": [1.0, 2.0]}}
    resp = client.post("/inspect", data={"request": json.dumps(bad)}, files=files)
    assert resp.status_code == 422
    assert "pressure drop per flow rate" in json.dumps(resp.json())


def test_implausible_inputs_route_to_refer(client: TestClient) -> None:
    uid, req, files = _sample(client)
    req["metadata"]["operating_hours"] = 900000
    result = client.post("/inspect", data={"request": json.dumps(req)}, files=files).json()
    assert result["route"] == "REFER" and result["reason"] == "invalid"
    assert any("operating_hours" in issue for issue in result["input_issues"])
