"""Stage ``validate``: the industrial validation campaign and acceptance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from preinspect.config import Config, data_dir, load_aux, reports_dir
from preinspect.data.io import load_latent, load_scenes, load_signals
from preinspect.report.render import render_to_file
from preinspect.runs import Run, open_run
from preinspect.service.pipeline import InspectionPipeline
from preinspect.utils.logging import get_logger, setup_logging
from preinspect.utils.seeding import seed_everything
from preinspect.utils.timing import timed
from preinspect.validation import doe, latency
from preinspect.validation.acceptance import audit_integrity_check, evaluate_criteria
from preinspect.validation.common import inputs_for, inspect_all, score_results, stratified
from preinspect.validation.domain_shift import commission, split_shift
from preinspect.validation.msa import attribute_study, grr_study
from preinspect.validation.robustness import degradation_figure, run_robustness

log = get_logger(__name__)


def _truth(latent: pd.DataFrame, units: pd.DataFrame) -> list[str]:
    return latent.loc[units["unit_id"], "outcome_true"].tolist()


def domain_shift_study(
    cfg: Config,
    pipe: InspectionPipeline,
    ddir: Path,
    units: pd.DataFrame,
    signals,
    latent: pd.DataFrame,
    reference: dict,
) -> dict[str, Any]:
    """Shift-set performance before and after commissioning."""
    n = cfg["campaign"]["domain_shift"]["commissioning_units"]
    comm, evaluation = split_shift(units, n)
    inputs = inputs_for(ddir, evaluation, signals)
    before = score_results(inspect_all(pipe, inputs), _truth(latent, evaluation))
    adapted = commission(cfg, pipe, ddir, comm["unit_id"].tolist())
    after = score_results(inspect_all(adapted, inputs), _truth(latent, evaluation))
    return {
        "reference": reference,
        "before": before,
        "after": after,
        "commissioning_units": int(len(comm)),
    }


def latency_study(
    cfg: Config, run: Run, pipe: InspectionPipeline, ddir: Path, units: pd.DataFrame, signals
) -> dict[str, Any]:
    """Per-unit latency on the accelerator and on CPU, with ONNX Runtime U-Nets on CPU."""
    sample = units[units["split"] == "test"].head(cfg["campaign"]["latency"]["units"] + 2)
    timings = {
        f"{pipe.device.type} (PyTorch)": latency.time_pipeline(
            pipe, inputs_for(ddir, sample, signals)
        )
    }
    cpu = InspectionPipeline(cfg, run, device=torch.device("cpu"))
    onnx_nets = latency.onnx_unets(cpu, run.path / "models" / "onnx")
    diff = latency.onnx_agreement(cpu, onnx_nets, cfg["generator"]["image_size"])
    timings["cpu (PyTorch)"] = latency.time_pipeline(cpu, inputs_for(ddir, sample, signals))
    cpu.unets = onnx_nets
    timings["cpu (ONNX Runtime U-Net)"] = latency.time_pipeline(
        cpu, inputs_for(ddir, sample, signals)
    )
    key = "cpu (ONNX Runtime U-Net)" if cfg["deploy"]["cpu_runtime"] == "onnx" else "cpu (PyTorch)"
    return {
        "timings": timings,
        "onnx_max_abs_diff": diff,
        "deployed_cpu_path": key,
        "latency_p95_cpu_s": timings[key]["p95_s"],
    }


def run(cfg: Config, run_name: str | None = None) -> Run:
    """Run every study, evaluate the acceptance criteria and render the report."""
    run_ = open_run(cfg, run_name)
    setup_logging(log_file=run_.path / "logs" / "validate.log")
    seed_everything(cfg["seed"])
    ddir = data_dir(cfg)
    units = pd.read_parquet(ddir / "validation" / "units_clean.parquet")
    latent = load_latent(ddir).set_index("unit_id")
    signals, scenes = load_signals(ddir), load_scenes(ddir)
    inspectors = pd.read_parquet(ddir / "inspectors.parquet")
    pipe = InspectionPipeline(cfg, run_)
    metrics = run_.read_metrics()
    deployed = cfg["deploy"]["model"]
    decision = metrics["evaluate"]["decisions"][deployed]
    timings: dict[str, float] = {}
    out: dict[str, Any] = {}
    test = units[units["split"] == "test"]
    with timed(timings, "robustness"):
        sample = stratified(
            test, latent["outcome_true"], cfg["campaign"]["robustness"]["n_units"], cfg["seed"]
        )
        rob = run_robustness(cfg, pipe, inputs_for(ddir, sample, signals), _truth(latent, sample))
        degradation_figure(rob, run_.sub("figures", "robustness.png"))
        out["robustness"] = rob.to_dict("records")
    with timed(timings, "domain_shift"):
        reference = score_results(
            inspect_all(pipe, inputs_for(ddir, test, signals)), _truth(latent, test)
        )
        out["domain_shift"] = domain_shift_study(cfg, pipe, ddir, units, signals, latent, reference)
    with timed(timings, "msa"):
        out["msa"] = {
            "attribute": attribute_study(
                cfg, pipe, ddir, units, signals, scenes, latent, inspectors
            ),
            "grr": grr_study(cfg, pipe, ddir, units, signals, scenes, latent),
        }
    with timed(timings, "doe"):
        runs = doe.run_doe(cfg, pipe, units, scenes)
        doe.effects_figure(runs, run_.sub("figures", "doe_effects.png"))
        out["doe"] = {**doe.analyse(runs, cfg), "runs": runs.to_dict("records")}
    with timed(timings, "latency"):
        out["latency"] = latency_study(cfg, run_, pipe, ddir, units, signals)
    decisions = pd.read_parquet(run_.path / "decisions" / "deployed.parquet")
    out["audit"] = audit_integrity_check(decisions)
    clean_unsafe = rob.loc[rob["kind"] == "none", "unsafe_rate"].iloc[0]
    values = {
        "wrongful_scrap_rate": decision["rates"]["wrongful_scrap_rate"],
        "scrap_capture_rate": decision["rates"]["scrap_capture_rate"],
        "latency_p95_cpu_s": out["latency"]["latency_p95_cpu_s"],
        "route_repeatability": out["msa"]["attribute"]["route_repeatability"],
        "coverage_alpha_0_10": decision["coverage"]["0.1"]["sets_vs_recorded"]["coverage"],
        "max_wrong_auto_route_increase": float(rob["unsafe_rate"].max() - clean_unsafe),
        "audit_integrity_ok": out["audit"]["audit_integrity_ok"],
        "refer_share": decision["rates"]["share_REFER"],
    }
    evidence = {
        "AC-01": "Decisions on the test split",
        "AC-02": "Decisions on the test split",
        "AC-03": "Latency",
        "AC-04": "Measurement system analysis",
        "AC-05": "Decisions on the test split",
        "AC-06": "Robustness",
        "AC-07": "Audit trail",
        "AC-08": "Decisions on the test split",
    }
    criteria = evaluate_criteria(load_aux("acceptance"), values, evidence)
    out["acceptance"] = criteria
    out["timings_s"] = timings
    run_.write_metrics("validate", out)
    render_validation_report(cfg, run_)
    for c in criteria:
        log.info("%s %-28s %s (measured %s)", c["id"], c["name"], c["status"], c["value"])
    return run_


def render_validation_report(cfg: Config, run_: Run) -> None:
    """Render ``reports/validation_report.md`` from the run's stored metrics."""
    metrics = run_.read_metrics()
    out = metrics["validate"]
    prov = run_.read_json("provenance.json")
    criteria = out["acceptance"]
    context = {
        "run_name": run_.name,
        "profile": cfg["name"],
        "code_version": prov["git"]["commit"],
        "data_version": prov["data_manifest_sha256"] or "",
        "deployed": cfg["deploy"]["model"],
        "criteria": criteria,
        "passed": sum(c["status"] == "PASS" for c in criteria),
        "decision": metrics["evaluate"]["decisions"][cfg["deploy"]["model"]],
        "robustness": out["robustness"],
        "shift": out["domain_shift"],
        "msa": out["msa"],
        "doe": out["doe"],
        "latency": out["latency"],
        "audit": out["audit"],
        "timings": out["timings_s"],
    }
    target = reports_dir(cfg)
    render_to_file("validation_report.md.j2", context, target / "validation_report.md")
    for fig in ("robustness.png", "doe_effects.png", "doe_interaction.png"):
        src = run_.path / "figures" / fig
        if src.exists():
            (target / "figures").mkdir(parents=True, exist_ok=True)
            (target / "figures" / fig).write_bytes(src.read_bytes())
