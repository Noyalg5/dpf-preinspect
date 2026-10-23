"""Data-validation gate.

Runs before any training. Two kinds of check:

1. **Integrity** (dataset level): every file in the manifest exists and its
   SHA-256 matches; no unlisted files; schema columns present. Failure
   blocks training entirely.
2. **Record checks** (row level): required fields, ranges, categories,
   duplicate IDs, flow-curve dropouts, disposition consistent with the
   recorded test results, readable images. Failing records are quarantined
   with reasons, never silently dropped.
3. **Field checks**: a value that cannot be trusted but does not invalidate
   the record (a temperature that may have been logged in Fahrenheit) is
   blanked for modelling and listed for verification at source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from preinspect.config import Config, data_dir
from preinspect.report.render import render_to_file
from preinspect.synth.generate import UNHASHED_DIRS, UNHASHED_FILES
from preinspect.synth.outcome import disposition_rule
from preinspect.utils.hashing import sha256_file
from preinspect.utils.logging import get_logger

log = get_logger(__name__)

FIELD_CHECKS = {"PROBABLE_FAHRENHEIT": "max_regen_temp_c"}
CHECKS = {
    "MISSING_REQUIRED": "A required field is blank.",
    "OUT_OF_RANGE": "A numeric field is outside its plausible range.",
    "UNKNOWN_CATEGORY": "A categorical field holds an unrecognised value.",
    "DUPLICATE_ID": "The unit ID appears on more than one record.",
    "PROBABLE_FAHRENHEIT": "Temperature may be a Fahrenheit value; field blanked, record kept.",
    "FLOW_DROPOUTS": "Flow curve has more sensor dropouts than can be imputed.",
    "DISPOSITION_MISMATCH": "Recorded disposition contradicts the recorded test results.",
    "IMAGE_UNREADABLE": "An image file is missing, unreadable or blank.",
}


class DataIntegrityError(RuntimeError):
    """Raised when data fails integrity checks or has not been validated."""


@dataclass
class ValidationResult:
    """Outcome of the validation gate."""

    integrity_ok: bool
    file_issues: list[dict[str, str]]
    quarantine: pd.DataFrame
    accepted: list[str]
    warnings: dict[str, int] = field(default_factory=dict)
    n_records: int = 0


def check_files(ddir: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Compare files on disk with the manifest hashes."""
    issues = []
    listed = manifest["files"]
    for rel, digest in listed.items():
        path = ddir / rel
        if not path.exists():
            issues.append({"file": rel, "issue": "missing"})
        elif sha256_file(path) != digest:
            issues.append({"file": rel, "issue": "hash mismatch"})
    for path in ddir.rglob("*"):
        rel_parts = set(path.relative_to(ddir).parts)
        if (path.is_file() and path.name not in UNHASHED_FILES and not UNHASHED_DIRS & rel_parts
                and str(path.relative_to(ddir)) not in listed):
            issues.append({"file": str(path.relative_to(ddir)), "issue": "not in manifest"})
    return issues


def record_checks(
    units: pd.DataFrame, signals: np.ndarray, cfg: Config
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Row-level checks.

    Returns
    -------
    quarantine : pandas.DataFrame
        One row per (record, failed check): ``row``, ``unit_id``, ``check``, ``detail``;
        the caller adds ``level`` (record or field).
    warnings : dict
        Counts of non-blocking findings (e.g. imputable dropouts).
    """
    v = cfg["validation"]
    flags: list[dict[str, Any]] = []

    def flag(rows: pd.Index, check: str, detail: Any) -> None:
        for r in rows:
            d = detail(r) if callable(detail) else detail
            flags.append({"row": int(r), "unit_id": units.at[r, "unit_id"], "check": check,
                          "detail": d})

    for col in v["required"]:
        flag(units.index[units[col].isna()], "MISSING_REQUIRED", f"{col} is blank")
    known = {"material": set(cfg["generator"]["substrate"]["materials"]),
             "application": set(cfg["generator"]["applications"])}
    for col, allowed in known.items():
        bad = units[col].notna() & ~units[col].isin(allowed)
        flag(units.index[bad], "UNKNOWN_CATEGORY", lambda r, c=col: f"{c}={units.at[r, c]}")
    dup = units["unit_id"].duplicated(keep=False)
    flag(units.index[dup], "DUPLICATE_ID", "unit_id booked in more than once")

    temp = units["max_regen_temp_c"]
    as_c = (temp - 32.0) * 5.0 / 9.0
    lo_t, hi_t = v["ranges"]["max_regen_temp_c"]
    fahr = (temp > v["fahrenheit_suspect_above"]) & as_c.between(lo_t, hi_t)
    flag(units.index[fahr], "PROBABLE_FAHRENHEIT",
         lambda r: f"{temp[r]} would be {as_c[r]:.0f} °C if logged in Fahrenheit")
    for col, (lo, hi) in v["ranges"].items():
        vals = pd.to_numeric(units[col], errors="coerce")
        bad = vals.notna() & ((vals < lo) | (vals > hi))
        if col == "max_regen_temp_c":
            bad &= ~fahr
        flag(units.index[bad], "OUT_OF_RANGE", lambda r, c=col: f"{c}={units.at[r, c]}")

    curves = signals[units["signal_index"].to_numpy(), 1, :]
    n_nan = np.isnan(curves).sum(axis=1)
    flag(units.index[n_nan > v["max_flow_dropouts"]], "FLOW_DROPOUTS",
         lambda r: f"{n_nan[units.index.get_loc(r)]} of {curves.shape[1]} points missing")

    spec = cfg["generator"]["outcome"]["spec_ratio"]
    implied = [
        disposition_rule(bool(i), float(c), float(e), spec)
        for i, c, e in zip(units["integrity_pass"], units["dp_ratio_post_clean"],
                           units["dp_ratio_post_extended"], strict=True)
    ]
    mismatch = units["disposition"].to_numpy() != np.asarray(implied)
    flag(units.index[mismatch], "DISPOSITION_MISMATCH",
         lambda r: f"recorded {units.at[r, 'disposition']}, tests imply {implied[r]}")

    warnings = {"flow_points_imputed": int(((n_nan > 0) & (n_nan <= v["max_flow_dropouts"])).sum())}
    return pd.DataFrame(flags, columns=["row", "unit_id", "check", "detail"]), warnings


def image_checks(ddir: Path, units: pd.DataFrame, min_std: float) -> pd.DataFrame:
    """Flag records whose images are missing, unreadable or blank."""
    flags = []
    for r, uid in units["unit_id"].items():
        for view in ("inlet", "outlet"):
            path = ddir / "images" / f"{uid}_{view}.jpg"
            img = cv2.imread(str(path)) if path.exists() else None
            if img is None or float(img.std()) < min_std:
                flags.append({"row": int(r), "unit_id": uid, "check": "IMAGE_UNREADABLE",
                              "detail": f"{view} image missing, unreadable or blank"})
    return pd.DataFrame(flags, columns=["row", "unit_id", "check", "detail"])


def validate_dataset(cfg: Config, write: bool = True) -> ValidationResult:
    """Run the gate and write ``validation/`` outputs next to the data.

    Parameters
    ----------
    cfg : dict
        Profile configuration.
    write : bool
        Write the markdown report, quarantine list and status file.

    Returns
    -------
    ValidationResult
        Integrity verdict, quarantined records and accepted unit IDs.
    """
    ddir = data_dir(cfg)
    manifest = json.loads((ddir / "manifest.json").read_text())
    file_issues = check_files(ddir, manifest)
    units = pd.read_parquet(ddir / "units.parquet")
    missing_cols = [c for c in cfg["validation"]["required"] if c not in units.columns]
    file_issues += [{"file": "units.parquet", "issue": f"column {c} missing"} for c in missing_cols]
    signals = np.load(ddir / "signals.npy")
    flags, warnings = record_checks(units, signals, cfg)
    flags = pd.concat([flags, image_checks(ddir, units, cfg["validation"]["image_min_std"])],
                      ignore_index=True)
    flags["level"] = np.where(flags["check"].isin(FIELD_CHECKS), "field", "record")
    bad_rows = set(flags.loc[flags["level"] == "record", "row"])
    accepted = units.loc[~units.index.isin(bad_rows), "unit_id"].tolist()
    result = ValidationResult(
        integrity_ok=not file_issues, file_issues=file_issues, quarantine=flags,
        accepted=accepted, warnings=warnings, n_records=len(units),
    )
    if write:
        _write_outputs(cfg, ddir, manifest, units, result)
    log.info("validation: integrity %s, %d records, %d quarantined, %d accepted",
             "OK" if result.integrity_ok else "FAILED", len(units), len(bad_rows), len(accepted))
    return result


def _write_outputs(cfg: Config, ddir: Path, manifest: dict, units: pd.DataFrame,
                   result: ValidationResult) -> None:
    out = ddir / "validation"
    out.mkdir(exist_ok=True)
    result.quarantine.to_csv(out / "quarantine.csv", index=False)
    pd.Series(result.accepted, name="unit_id").to_csv(out / "accepted.csv", index=False)
    by_check = result.quarantine.groupby("check")["row"].nunique().to_dict()
    clean = units.copy()
    for check, column in FIELD_CHECKS.items():
        rows = result.quarantine.loc[result.quarantine["check"] == check, "row"]
        clean.loc[rows, column] = np.nan
    accepted_units = clean[clean["unit_id"].isin(result.accepted)]
    accepted_units.to_parquet(out / "units_clean.parquet", index=False)
    record_flags = result.quarantine[result.quarantine["level"] == "record"]
    status = {
        "validated": datetime.now().isoformat(timespec="seconds"),
        "manifest_sha256": sha256_file(ddir / "manifest.json"),
        "integrity_ok": result.integrity_ok,
        "n_records": result.n_records,
        "n_quarantined_records": int(record_flags["row"].nunique()),
        "n_fields_blanked": int((result.quarantine["level"] == "field").sum()),
        "n_accepted": len(result.accepted),
        "units_clean_sha256": sha256_file(out / "units_clean.parquet"),
        "by_check": by_check,
        "warnings": result.warnings,
        "accepted_by_split": accepted_units["split"].value_counts().to_dict(),
    }
    (out / "status.json").write_text(json.dumps(status, indent=1))
    render_to_file("data_quality_report.md.j2", {
        "status": status, "checks": CHECKS, "manifest": manifest,
        "file_issues": result.file_issues, "field_checks": FIELD_CHECKS,
        "examples": result.quarantine.drop(columns="row").head(25).to_dict("records"),
    }, out / "data_quality_report.md")


def require_valid(cfg: Config) -> pd.DataFrame:
    """Return the accepted, field-cleaned units table, refusing unsafe data.

    Raises
    ------
    DataIntegrityError
        If validation has not run, the manifest changed since validation, a
        file no longer matches its hash, or integrity failed.
    """
    ddir = data_dir(cfg)
    status_path = ddir / "validation" / "status.json"
    if not status_path.exists():
        raise DataIntegrityError(f"{ddir} has not been validated; run `preinspect data` first.")
    status = json.loads(status_path.read_text())
    if not status["integrity_ok"]:
        raise DataIntegrityError("dataset failed integrity checks; see data_quality_report.md")
    if sha256_file(ddir / "manifest.json") != status["manifest_sha256"]:
        raise DataIntegrityError("manifest changed since validation; re-run the validation gate")
    issues = check_files(ddir, json.loads((ddir / "manifest.json").read_text()))
    clean_path = ddir / "validation" / "units_clean.parquet"
    if not clean_path.exists() or sha256_file(clean_path) != status["units_clean_sha256"]:
        issues.append({"file": "validation/units_clean.parquet", "issue": "missing or altered"})
    if issues:
        raise DataIntegrityError(f"{len(issues)} file(s) fail integrity, e.g. {issues[0]}")
    return pd.read_parquet(clean_path)
