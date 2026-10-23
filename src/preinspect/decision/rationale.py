"""Plain-English rationale built from measured concepts by fixed templates.

No language model is involved: every sentence comes from a template filled
with a measured value, so the same inputs always give the same wording and
every statement can be traced to a feature.
"""

from __future__ import annotations

import math
from typing import Any

FACE = {"inlet": "inlet face", "outlet": "outlet face"}
ROUTE_TEXT = {
    "PROCEED": "proceed to standard clean",
    "EXTENDED_CLEAN": "send to extended clean",
    "QUARANTINE": "quarantine for confirmation (scrap candidate; a person must confirm)",
    "REFER": "refer to an inspector",
}


def _hour(value: float) -> str:
    return f"{int(round(value)) % 12 or 12} o'clock"


def _finding(view: str, f: dict[str, Any], rc: dict[str, Any]) -> list[str]:
    """Sentences about one face from its learned concepts (``lrn_<view>_*``)."""
    p = f"lrn_{view}_"
    out = []
    face = FACE[view]

    def present(cls: str) -> bool:
        return (
            f.get(f"{p}{cls}_presence", 0.0) >= rc["presence_min"]
            or f.get(f"{p}{cls}_area", 0.0) >= rc["area_min"]
        )

    if present("crack"):
        length = f.get(f"{p}crack_length_mm", float("nan"))
        clock = f.get(f"{p}crack_clock", float("nan"))
        where = f" at {_hour(clock)}" if math.isfinite(clock) else ""
        size = f", about {length:.0f} mm" if math.isfinite(length) and length > 0 else ""
        out.append(f"Crack indication on the {face}{where}{size}.")
    if present("missing_plug"):
        n = int(f.get(f"{p}missing_plug_count", 0))
        out.append(f"Missing or failed plugs on the {face} (about {max(n, 1)} cells).")
    if present("chip"):
        clock = f.get(f"{p}chip_clock", float("nan"))
        where = f" at {_hour(clock)}" if math.isfinite(clock) else ""
        out.append(f"Edge chip or skin damage on the {face}{where}.")
    if present("face_plugging"):
        area = 100 * f.get(f"{p}face_plugging_area", 0.0)
        out.append(f"Deposits blocking about {area:.0f} % of the {face}.")
    if present("stain"):
        out.append(f"Stain on the {face} (possible oil or coolant contamination).")
    if present("outlet_soot") and view == "outlet":
        area = 100 * f.get(f"{p}outlet_soot_area", 0.0)
        out.append(f"Soot on the outlet face over about {area:.0f} % of it (symptom of leakage).")
    if present("debris"):
        out.append(f"Foreign debris on the {face}.")
    return out


def rationale(
    features: dict[str, Any],
    proba: dict[str, float],
    prediction_set: list[str],
    route: str,
    reason: str,
    rc: dict[str, Any],
) -> str:
    """Compose the rationale for one unit.

    Parameters
    ----------
    features : dict
        The unit's feature row (learned concepts, flow, cross-modal residual, anomaly).
    proba : dict
        Calibrated outcome probabilities.
    prediction_set : list of str
        Conformal prediction set.
    route, reason : str
        Route and reason code from the routing policy.
    rc : dict
        ``rationale`` configuration section.
    """
    sentences = _finding("inlet", features, rc) + _finding("outlet", features, rc)
    if not sentences:
        sentences.append("No defect indications on either face.")
    resid = features.get("xmod_load_residual", float("nan"))
    if math.isfinite(resid):
        if resid < rc["load_residual_low"]:
            sentences.append(
                "Flow restriction lower than expected for this weight gain "
                "(possible bypass through a crack or missing plugs)."
            )
        elif resid > rc["load_residual_high"]:
            sentences.append(
                "Flow restriction higher than expected for this weight gain "
                "(heavy loading or glazing)."
            )
        else:
            sentences.append("Flow restriction consistent with the weight gain.")
    if reason == "anomaly":
        sentences.append(
            "An area of the face does not match any known defect or normal appearance."
        )
    if reason == "invalid":
        sentences.append("Input checks failed, so no automatic decision is made.")
    top = max(proba, key=proba.get)
    sentences.append(
        f"Most likely outcome {top} ({100 * proba[top]:.0f} %); plausible outcomes: "
        f"{', '.join(prediction_set)}."
    )
    from preinspect.decision.routing import REASONS

    sentences.append(f"Decision: {ROUTE_TEXT[route]}, because {REASONS[reason]}.")
    return " ".join(sentences)
