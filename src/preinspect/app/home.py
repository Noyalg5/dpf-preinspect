"""Streamlit entry page for the decision-support UI."""

import streamlit as st

from preinspect.app.client import API_URL, get

st.set_page_config(page_title="DPF pre-inspection", layout="wide")
st.title("DPF pre-inspection")
st.caption(
    "Research prototype. Models are trained on synthetic data only; every QUARANTINE "
    "must be confirmed by a person."
)
try:
    info = get("/model-info")
    st.success(
        f"Connected to {API_URL}: model {info['model_version']}, "
        f"deployed route {info['deployed_model']}"
    )
    st.json({"policy": info["policy"], "inputs": info["inputs"]}, expanded=False)
except Exception as exc:  # noqa: BLE001 - show any connection problem to the user
    st.error(f"Cannot reach the API at {API_URL} ({exc}). Start it with `make api`.")
st.markdown(
    "- **Inspection Station**: inspect a unit, read the route and rationale, accept or override.\n"
    "- **Quality Engineer**: throughput, route mix, control chart, drift and calibration."
)
