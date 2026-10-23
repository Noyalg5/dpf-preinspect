"""Validation campaign and acceptance report on the tiny run."""

from __future__ import annotations

from preinspect.config import reports_dir


def test_validate_stage_reports_every_criterion(tiny_data, tiny_run) -> None:
    from preinspect.stages import validate

    run = validate.run(tiny_data, tiny_run.name)
    out = run.read_metrics()["validate"]
    criteria = {c["id"]: c for c in out["acceptance"]}
    assert set(criteria) == {f"AC-0{i}" for i in range(1, 8)}
    assert all(c["status"] in {"PASS", "FAIL", "NOT MEASURED"} for c in criteria.values())
    assert criteria["AC-07"]["status"] == "PASS"
    assert out["latency"]["onnx_max_abs_diff"] < 1e-3
    report = (reports_dir(tiny_data) / "validation_report.md").read_text()
    assert "Synthetic data" in report and "AC-06" in report
