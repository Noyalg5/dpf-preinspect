"""Every result-bearing document renders from a tiny run, with the synthetic-data disclaimer."""

from __future__ import annotations

from preinspect.config import docs_dir


def test_simulate_and_render_documents(tiny_data, tiny_run) -> None:
    from preinspect.stages import report, simulate

    simulate.run(tiny_data, tiny_run.name)
    assert tiny_run.read_metrics()["simulate"]["conservation_ok"]
    report.run(tiny_data, tiny_run.name)
    docs = docs_dir(tiny_data)
    readme = (docs.parent / "README.md").read_text()
    assert "synthetic" in readme and "| Metadata only |" in readme
    for name in (
        "technical_report",
        "model_card",
        "faq",
        "training_manual",
        "system_design",
        "demo_script",
        "real_data_plan",
        "data_card",
    ):
        text = (docs / f"{name}.md").read_text()
        assert len(text) > 500 and "nan %" not in text, name
