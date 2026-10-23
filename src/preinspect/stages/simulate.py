"""Stage ``simulate``: process-impact simulation of the pre-inspection gate."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from preinspect.config import Config, load_aux, reports_dir
from preinspect.report.style import INK, SERIES, save_figure
from preinspect.runs import Run, open_run
from preinspect.sim.tornado import compare, outcome_mix, route_matrix, tornado
from preinspect.utils.logging import get_logger, setup_logging

log = get_logger(__name__)
KPIS = [
    "throughput_per_day",
    "oven_hours",
    "energy_kwh",
    "co2e_kg",
    "cost_per_unit",
    "wrongful_scraps",
    "wip_mean",
    "lead_time_h_mean",
]


def kpi_figure(df: pd.DataFrame, path) -> None:
    """Baseline versus gated line, one small panel per KPI (mean over replications)."""
    means = df.groupby("scenario")[KPIS].mean()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for ax, kpi in zip(axes.ravel(), KPIS, strict=True):
        vals = [means.loc["baseline", kpi], means.loc["with gate", kpi]]
        ax.bar(["baseline", "with gate"], vals, color=[INK["muted"], SERIES[0]], width=0.5)
        ax.set_title(kpi.replace("_", " "), fontsize=9)
    fig.suptitle(
        "Line simulation: today's process versus the pre-inspection gate "
        "(assumed rates and costs, synthetic route rates)",
        fontsize=10,
    )
    save_figure(fig, path, means.reset_index())


def tornado_figure(tor: pd.DataFrame, path) -> None:
    """Swing of the saving per unit across each input's range."""
    tor = tor.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    base = tor["saving_base"].iloc[0]
    for i, row in enumerate(tor.itertuples()):
        lo, hi = sorted((row.saving_low, row.saving_high))
        ax.barh(i, hi - lo, left=lo, color=SERIES[0], height=0.5)
        ax.text(
            hi,
            i,
            f"  {row.low_value:g} to {row.high_value:g}",
            va="center",
            fontsize=7,
            color=INK["secondary"],
        )
    ax.axvline(base, color=INK["muted"], linewidth=1)
    ax.set_yticks(range(len(tor)), [p.replace("_", " ") for p in tor["parameter"]])
    ax.set_xlabel("Saving per unit (GBP, assumed costs)")
    ax.set_title("Sensitivity of the saving to uncertain inputs")
    save_figure(fig, path, tor)


def run(cfg: Config, run_name: str | None = None) -> Run:
    """Simulate baseline and gated lines and write KPIs, tornado and figures."""
    run_ = open_run(cfg, run_name)
    setup_logging(log_file=run_.path / "logs" / "simulate.log")
    sim, costs = load_aux("line_sim"), load_aux("costs")
    decision = run_.read_metrics()["evaluate"]["decisions"][cfg["deploy"]["model"]]
    matrix = route_matrix(decision["route_confusion"])
    mix = outcome_mix(decision["route_confusion"])
    df = compare(sim, costs, mix, matrix, sim["replications"], cfg["seed"])
    tor = tornado(sim, costs, mix, matrix, cfg["seed"])
    summary = df.groupby("scenario")[KPIS].agg(["mean", "std"])
    out = {
        "summary": {
            f"{k}|{stat}": summary[k][stat].to_dict() for k in KPIS for stat in ("mean", "std")
        },
        "route_given_outcome": {o: v.tolist() for o, v in matrix.items()},
        "outcome_mix": mix.tolist(),
        "tornado": tor.to_dict("records"),
        "conservation_ok": bool(
            (df["arrived"] == df["dispatched"] + df["scrapped"] + df["wip_end"]).all()
        ),
        "replications": int(sim["replications"]),
    }
    run_.write_metrics("simulate", out)
    df.to_csv(run_.sub("tables", "line_simulation.csv"), index=False)
    for name, fn, data in (("line_kpis.png", kpi_figure, df), ("tornado.png", tornado_figure, tor)):
        fn(data, run_.sub("figures", name))
        target = reports_dir(cfg) / "figures" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((run_.path / "figures" / name).read_bytes())
    means = df.groupby("scenario")["cost_per_unit"].mean()
    log.info("cost per unit: baseline %.2f, with gate %.2f", means["baseline"], means["with gate"])
    return run_
