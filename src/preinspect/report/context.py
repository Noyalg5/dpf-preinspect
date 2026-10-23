"""Everything the documentation templates need, gathered from run outputs and configs.

Templates receive numbers only from here, so no result in the README or docs
is typed by hand. Derived comparisons (which approach wins, whether fusion
beats the best single modality) are computed here too, and the templates
choose their wording from them.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from preinspect.config import Config, data_dir, load_aux, load_config
from preinspect.report.station import resolution_budget
from preinspect.runs import Run


def _f1(rows: list[dict], model: str) -> float | None:
    for r in rows:
        if r["model"] == model:
            return r["macro_f1"]["mean"]
    return None


def _safe(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def comparisons(headline: list[dict], ablation: list[dict]) -> dict[str, Any]:
    """Winner of the headline table and fusion-versus-single-modality gaps."""
    best = max(headline, key=lambda r: r["macro_f1"]["mean"])
    by_groups = {r["groups"]: r["macro_f1"]["mean"] for r in ablation}
    singles = {g: by_groups[g] for g in ("V", "F", "M") if g in by_groups}
    best_single = max(singles, key=singles.get) if singles else None
    concept_all = _f1(headline, "learned_flow_meta")
    e2e_all, e2e_img = _f1(headline, "e2e_all"), _f1(headline, "e2e_image")
    classical_all = _f1(headline, "classical_flow_meta")
    saving = {r["model"]: r["saving_per_unit"]["mean"] for r in headline if "saving_per_unit" in r}
    deployed_saving = saving.get("learned_flow_meta")
    flow_auc = next((r["auc_scrap"]["mean"] for r in headline if r["model"] == "flow_only"), None)
    return {
        "flow_scrap_auc": flow_auc,
        "deployed_best_saving": deployed_saving is not None
        and all(deployed_saving >= v for v in saving.values()),
        "best": best,
        "best_is_concept_route": best["model"] in ("learned_flow_meta", "classical_flow_meta"),
        "best_single_group": {"V": "vision", "F": "flow", "M": "metadata"}.get(
            best_single, best_single
        ),
        "best_single_f1": singles.get(best_single) if best_single else None,
        "fusion_gain": (by_groups.get("VFM", 0) - singles[best_single]) if best_single else None,
        "e2e_gain_over_image": (e2e_all - e2e_img) if e2e_all and e2e_img else None,
        "concept_minus_e2e": (concept_all - e2e_all) if concept_all and e2e_all else None,
        "learned_minus_classical": (concept_all - classical_all)
        if concept_all and classical_all
        else None,
    }


def build_context(cfg: Config, run: Run) -> dict[str, Any]:
    """Collect metrics, data statistics, configs and timings for rendering."""
    m = run.read_metrics()
    ddir = data_dir(cfg)
    prov = run.read_json("provenance.json")
    ev = m.get("evaluate", {})
    deployed = cfg["deploy"]["model"]
    headline, ablation = ev.get("headline", []), ev.get("ablation", [])
    timings = dict(m.get("timings", {}))
    stage_data = _safe(ddir / "logs" / "stage.json")
    if stage_data:
        timings["data"] = stage_data["seconds"]
    timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 1)
    return {
        "today": date.today().isoformat(),
        "profile": cfg["name"],
        "run_name": run.name,
        "commit": prov["git"]["commit"][:10],
        "dirty": prov["git"]["dirty"],
        "manifest": (prov["data_manifest_sha256"] or "")[:12],
        "generator_version": cfg["generator"]["version"],
        "seed": cfg["seed"],
        "stats": _safe(ddir / "stats.json"),
        "headline": headline,
        "ablation": ablation,
        "cmp": comparisons(headline, ablation) if headline else None,
        "models": ev.get("models", {}),
        "vision": _safe(run.path / "metrics" / "vision.json"),
        "train": m.get("train", {}),
        "explain": m.get("explain"),
        "decision": ev.get("decisions", {}).get(deployed),
        "deployed": deployed,
        "validate": m.get("validate"),
        "simulate": m.get("simulate"),
        "costs": load_aux("costs"),
        "acceptance": load_aux("acceptance"),
        "line_sim": load_aux("line_sim"),
        "cfg": cfg,
        "timings": timings,
        "budget": resolution_budget(cfg["station"]).to_dict("records"),
        "sample_sizes": sample_sizes(_safe(ddir / "stats.json")),
        "gates": fusion_gates(run),
        "tradeoff": capture_tradeoff(run, load_aux("costs")),
        "full": other_profile_summary(cfg, "full") if cfg["name"] != "full" else None,
    }


def other_profile_summary(cfg: Config, profile: str) -> dict[str, Any] | None:
    """Headline facts from the latest local run of another profile, if one exists."""
    from preinspect.config import runs_dir

    runs = sorted(runs_dir(cfg).glob(f"*_{profile}"))
    if not runs:
        return None
    run = Run(runs[-1])
    m = run.read_metrics()
    if "validate" not in m or "evaluate" not in m:
        return None
    best = max(m["evaluate"]["headline"], key=lambda r: r["macro_f1"]["mean"])
    timings = {k: v for k, v in m.get("timings", {}).items()}
    stage = _safe(data_dir(load_config(profile)) / "logs" / "stage.json")
    total = sum(timings.values()) + (stage["seconds"] if stage else 0.0)
    acc = m["validate"]["acceptance"]
    return {
        "run_name": run.name,
        "minutes": total / 60.0,
        "best": best,
        "passed": sum(c["status"] == "PASS" for c in acc),
        "criteria": len(acc),
        "failed": [c["id"] for c in acc if c["status"] != "PASS"],
    }


def capture_tradeoff(run: Run, costs: dict) -> dict | None:
    """Chosen policy against the cheapest one reaching 90 % scrap capture (calibration data)."""
    import pandas as pd

    path = run.path / "tables" / "threshold_search.csv"
    policy = _safe(run.path / "models" / "policy.json")
    if not path.exists() or policy is None:
        return None
    search = pd.read_csv(path)
    cap = costs["policy"]["wrongful_scrap_cap"]["value"]
    chosen = (
        search[
            (search["t_quarantine"] == policy["t_quarantine"])
            & (search["t_rework"] == policy["t_rework"])
            & (search["alpha"] == policy["alpha"])
        ]
        .iloc[0]
        .to_dict()
    )
    ok = search[(search["wrongful_scrap_rate"] <= cap) & (search["scrap_capture_rate"] >= 0.9)]
    target = ok.loc[ok["cost"].idxmin()].to_dict() if len(ok) else None
    return {"chosen": chosen, "capture90": target}


def fusion_gates(run: Run) -> dict[str, float] | None:
    """Mean gate weight per modality of the end-to-end network on the test split."""
    import pandas as pd

    path = run.path / "predictions" / "e2e_all_s0.parquet"
    if not path.exists():
        return None
    preds = pd.read_parquet(path)
    test = preds[preds["split"] == "test"]
    return {c[5:]: float(test[c].mean()) for c in test.columns if c.startswith("gate_")}


def sample_sizes(stats: dict | None) -> list[dict[str, float]]:
    """Units needed to estimate SCRAP recall to a given precision (normal approximation).

    n_scrap = z^2 p (1 - p) / h^2 at expected recall p and half-width h (95 %);
    total units = n_scrap / scrap prevalence.
    """
    import math

    prevalence = 0.25
    if stats:
        test = stats["disposition_by_split"].get("test", {})
        total = sum(test.values())
        prevalence = test.get("SCRAP", 0) / total if total else prevalence
    rows = []
    for recall in (0.85, 0.90, 0.95):
        for half_width in (0.05, 0.03):
            n_scrap = math.ceil(1.96**2 * recall * (1 - recall) / half_width**2)
            rows.append(
                {
                    "recall": recall,
                    "half_width": half_width,
                    "n_scrap": n_scrap,
                    "n_units": math.ceil(n_scrap / prevalence),
                    "prevalence": prevalence,
                }
            )
    return rows
