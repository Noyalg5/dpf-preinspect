"""Thin HTTP client the Streamlit pages use to talk to the inspection API."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

API_URL = os.environ.get("PREINSPECT_API", "http://127.0.0.1:8000")


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=300)


def get(path: str, **params: Any) -> Any:
    """GET a JSON resource."""
    with _client() as c:
        r = c.get(path, params=params)
        r.raise_for_status()
        return r.json()


def post_json(path: str, body: dict[str, Any]) -> Any:
    """POST a JSON body; returns the JSON response or raises with the API's message."""
    with _client() as c:
        r = c.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()


def inspect(images: dict[str, bytes], request: dict[str, Any]) -> dict[str, Any]:
    """Send two face images and the inspection request."""
    files = {v: (f"{v}.jpg", b, "image/jpeg") for v, b in images.items()}
    with _client() as c:
        r = c.post("/inspect", data={"request": json.dumps(request)}, files=files)
        if r.status_code >= 400:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()
