"""Stream test units through a running API to populate the dashboard.

For each unit: the station inspection, then an operator decision (REFER and
QUARANTINE are resolved with the recorded outcome, as an inspector would
after review), then the downstream test result. ``drift=True`` darkens the
images and ages the service history progressively, so the drift panels
and control chart have something to show.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import httpx
import numpy as np

from preinspect.utils.logging import get_logger

log = get_logger(__name__)
ROUTE_FOR = {"PASS": "PROCEED", "REWORK": "EXTENDED_CLEAN", "SCRAP": "QUARANTINE"}


def _drifted(jpeg: bytes, gain: float) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def operator_decision(result: dict[str, Any], recorded: str) -> dict[str, Any]:
    """Simulated operator: accept automatic routes, resolve REFER/QUARANTINE on review."""
    route = result["route"]
    if route in ("PROCEED", "EXTENDED_CLEAN"):
        return {"accepted": True, "route": route, "reason": None}
    reviewed = ROUTE_FOR[recorded] if route == "REFER" else route
    if route == "QUARANTINE" and recorded != "SCRAP":
        reviewed = ROUTE_FOR[recorded]
    accepted = reviewed == route
    reason = None if accepted else f"inspector review: unit judged {recorded}"
    return {"accepted": accepted, "route": reviewed, "reason": reason}


def replay(cfg: dict[str, Any], url: str, limit: int, drift: bool = False) -> dict[str, int]:
    """Replay up to ``limit`` test units through the API at ``url``."""
    counts: dict[str, int] = {}
    with httpx.Client(base_url=url, timeout=300) as client:
        ids = client.get("/samples", params={"limit": limit}).json()
        for k, uid in enumerate(ids):
            sample = client.get(f"/samples/{uid}").json()
            req = sample["request"]
            files = {
                v: base64.b64decode(sample["images_jpeg_base64"][v]) for v in ("inlet", "outlet")
            }
            if drift:
                frac = k / max(1, len(ids) - 1)
                files = {v: _drifted(b, 1.0 - 0.45 * frac) for v, b in files.items()}
                req["metadata"]["operating_hours"] *= 1.0 + frac
            resp = client.post(
                "/inspect",
                data={"request": json.dumps(req)},
                files={v: (f"{uid}_{v}.jpg", b, "image/jpeg") for v, b in files.items()},
            )
            resp.raise_for_status()
            result = resp.json()
            recorded = sample["recorded"]["disposition"]
            decision = operator_decision(result, recorded)
            client.post(
                "/decision",
                json={
                    "inspection_id": result["inspection_id"],
                    "unit_id": uid,
                    "operator": "replay",
                    **decision,
                },
            ).raise_for_status()
            client.post(
                "/result",
                json={
                    "unit_id": uid,
                    "disposition": recorded,
                    "dp_ratio_post_clean": sample["recorded"]["dp_ratio_post_clean"],
                },
            )
            counts[result["route"]] = counts.get(result["route"], 0) + 1
            if (k + 1) % 25 == 0:
                log.info("replayed %d/%d units", k + 1, len(ids))
    log.info("replay finished: %s", counts)
    return counts
