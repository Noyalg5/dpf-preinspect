"""Command-line interface: ``preinspect <command> --config <profile>``.

Each command maps to a Makefile target and delegates to a module in
``preinspect.stages``. Imports are deferred so ``--help`` stays fast.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from preinspect.config import load_config
from preinspect.utils.logging import setup_logging

app = typer.Typer(
    help="DPF vision pre-inspection research prototype (synthetic data).",
    no_args_is_help=True,
    add_completion=False,
)
audit_app = typer.Typer(help="Audit-trail tools.", no_args_is_help=True)
app.add_typer(audit_app, name="audit")

ConfigOpt = Annotated[str, typer.Option("--config", "-c", help="Profile name or YAML path.")]
RunOpt = Annotated[
    str | None, typer.Option("--run", help="Run folder name or path (default: latest).")
]


def _timed(stage: str, fn) -> None:
    """Run a stage and record its wall-clock time in the run's metrics."""
    import time

    start = time.perf_counter()
    run_ = fn()
    run_.record_timing(stage, time.perf_counter() - start)


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def info(config: ConfigOpt = "demo") -> None:
    """Show the resolved profile, compute device and dataset status."""
    from preinspect.config import data_dir
    from preinspect.utils.device import resolve_device

    cfg = load_config(config)
    ddir = data_dir(cfg)
    typer.echo(f"profile      : {cfg['name']}")
    typer.echo(f"device       : {resolve_device(cfg.get('device', 'auto'))}")
    status = "present" if (ddir / "manifest.json").exists() else "absent"
    typer.echo(f"dataset      : {ddir} ({status})")
    typer.echo(f"generator    : v{cfg['generator']['version']}")


@app.command()
def data(config: ConfigOpt = "demo") -> None:
    """Generate the synthetic dataset and run the data-validation gate."""
    from preinspect.stages import data as stage

    stage.run(load_config(config))


@app.command()
def train(config: ConfigOpt = "demo") -> None:
    """Train all models into a new run folder."""
    from preinspect.stages import train as stage

    _timed("train", lambda: stage.run(load_config(config)))


@app.command()
def evaluate(config: ConfigOpt = "demo", run: RunOpt = None) -> None:
    """Evaluate models, explanations, conformal sets and the routing policy."""
    from preinspect.stages import evaluate as stage

    _timed("evaluate", lambda: stage.run(load_config(config), run))


@app.command()
def validate(config: ConfigOpt = "demo", run: RunOpt = None) -> None:
    """Run the industrial validation campaign and acceptance tests."""
    from preinspect.stages import validate as stage

    _timed("validate", lambda: stage.run(load_config(config), run))


@app.command()
def simulate(config: ConfigOpt = "demo", run: RunOpt = None) -> None:
    """Run the process-impact line simulation."""
    from preinspect.stages import simulate as stage

    _timed("simulate", lambda: stage.run(load_config(config), run))


@app.command()
def report(config: ConfigOpt = "demo", run: RunOpt = None) -> None:
    """Render README, docs and reports from run outputs."""
    from preinspect.stages import report as stage

    _timed("report", lambda: stage.run(load_config(config), run))


@app.command()
def api(
    config: ConfigOpt = "demo",
    run: RunOpt = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve the inspection API."""
    import os

    import uvicorn

    os.environ["PREINSPECT_CONFIG"] = config
    if run:
        os.environ["PREINSPECT_RUN"] = run
    uvicorn.run("preinspect.service.api:create_app", host=host, port=port, factory=True)


@app.command(name="app")
def app_(port: int = 8501) -> None:
    """Launch the Streamlit decision-support UI (expects the API to be running)."""
    home = Path(__file__).parent / "app" / "home.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(home), "--server.port", str(port)],
        check=False,
    )


@app.command()
def replay(
    config: ConfigOpt = "demo",
    url: str = "http://127.0.0.1:8000",
    limit: int = 150,
    drift: Annotated[bool, typer.Option(help="Inject gradual acquisition drift.")] = False,
) -> None:
    """Stream test units through the API to populate the dashboard."""
    from preinspect.service.replay import replay as do_replay

    do_replay(load_config(config), url=url, limit=limit, drift=drift)


@app.command()
def smoke(config: ConfigOpt = "demo", run: RunOpt = None) -> None:
    """Exercise the API in-process: health, one inspection, a decision, audit verification."""
    import json
    import os
    import tempfile

    from fastapi.testclient import TestClient

    from preinspect.runs import open_run
    from preinspect.service.api import create_app

    cfg = load_config(config)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["PREINSPECT_AUDIT_DB"] = str(Path(tmp) / "audit.sqlite")
        os.environ["PREINSPECT_MES_DB"] = str(Path(tmp) / "mes.sqlite")
        client = TestClient(create_app(cfg, open_run(cfg, run)))
        assert client.get("/health").json()["status"] == "ok"
        uid = client.get("/samples", params={"limit": 1}).json()[0]
        sample = client.get(f"/samples/{uid}").json()
        import base64

        files = {
            v: (f"{v}.jpg", base64.b64decode(sample["images_jpeg_base64"][v]), "image/jpeg")
            for v in ("inlet", "outlet")
        }
        req = {**sample["request"], "include_maps": True}
        result = client.post("/inspect", data={"request": json.dumps(req)}, files=files).json()
        client.post(
            "/decision",
            json={
                "inspection_id": result["inspection_id"],
                "unit_id": uid,
                "accepted": True,
                "route": result["route"],
                "operator": "smoke-test",
            },
        ).raise_for_status()
        verified = client.get("/audit/verify").json()
        os.environ.pop("PREINSPECT_AUDIT_DB")
        os.environ.pop("PREINSPECT_MES_DB")
    typer.echo(
        f"smoke test: unit {uid} routed {result['route']} ({result['reason']}); "
        f"audit chain {'verified' if verified['ok'] else 'FAILED'}"
    )
    typer.echo(f"rationale: {result['rationale']}")
    if not verified["ok"]:
        raise typer.Exit(code=1)


@audit_app.command("verify")
def audit_verify(
    db: Annotated[Path, typer.Option(help="Audit database path.")] = Path("runs/audit.sqlite"),
) -> None:
    """Verify the audit trail hash chain; exit code 1 if tampering is found."""
    from preinspect.service.audit import AuditTrail

    report = AuditTrail(db).verify()
    typer.echo(report.summary())
    raise typer.Exit(code=0 if report.ok else 1)


if __name__ == "__main__":
    app()
