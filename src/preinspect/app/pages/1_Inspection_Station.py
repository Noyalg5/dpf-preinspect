"""Inspection Station: one unit at a time, with accept or override."""

import base64
import json

import streamlit as st

from preinspect.app.client import get, inspect, post_json

ROUTES = ["PROCEED", "EXTENDED_CLEAN", "QUARANTINE", "REFER"]
st.set_page_config(page_title="Inspection Station", layout="wide")
st.title("Inspection Station")

source = st.radio("Input", ["Sample unit from the test set", "Upload"], horizontal=True)
images, request = None, None
if source.startswith("Sample"):
    uid = st.selectbox("Unit", get("/samples", limit=100))
    sample = get(f"/samples/{uid}")
    images = {v: base64.b64decode(sample["images_jpeg_base64"][v]) for v in ("inlet", "outlet")}
    request = sample["request"]
else:
    up_in = st.file_uploader("Inlet face image", type=["jpg", "jpeg", "png"])
    up_out = st.file_uploader("Outlet face image", type=["jpg", "jpeg", "png"])
    text = st.text_area("Request JSON (unit_id, metadata, flow, lighting)", height=200)
    if up_in and up_out and text:
        images = {"inlet": up_in.getvalue(), "outlet": up_out.getvalue()}
        request = json.loads(text)

if images and st.button("Inspect", type="primary"):
    try:
        st.session_state["result"] = inspect(images, {**request, "include_maps": True})
        st.session_state["images"] = images
    except RuntimeError as exc:
        st.error(f"The service rejected the request: {exc}")

result = st.session_state.get("result")
if result:
    route = result["route"]
    banner = {"REFER": st.error, "QUARANTINE": st.warning}.get(route, st.success)
    labels = {
        "REFER": "REFER: inspector review required",
        "QUARANTINE": "QUARANTINE: scrap candidate, confirm before any scrap decision",
    }
    banner(f"{labels.get(route, route)} ({result['reason']})")
    st.write(result["rationale"])
    layer = st.radio("Overlay", ["Original", "Defect outlines", "Anomaly heatmap"], horizontal=True)
    cols = st.columns(2)
    for col, view in zip(cols, ("inlet", "outlet"), strict=True):
        if layer == "Original":
            col.image(st.session_state["images"][view], caption=f"{view} face")
        else:
            key = "defects" if layer.startswith("Defect") else "anomaly"
            col.image(base64.b64decode(result["maps"][view][key]), caption=f"{view} face, {layer}")
    c1, c2, c3 = st.columns(3)
    c1.subheader("Outcome probabilities")
    for k, v in result["probabilities"].items():
        c1.progress(v, text=f"{k}: {100 * v:.0f} %")
    c1.caption(f"Plausible outcomes: {', '.join(result['prediction_set'])}")
    dp = result["dp_ratio"]
    c2.metric(
        "Predicted post-clean ΔP ratio",
        f"{dp['prediction']:.2f}",
        help=f"{100 * dp['coverage']:.0f} % interval {dp['lower']:.2f} to {dp['upper']:.2f}",
    )
    an = result["anomaly"]
    c3.metric(
        "Unexplained anomaly",
        f"{an['unexplained']:.2f}",
        delta=f"threshold {an['threshold']:.2f}",
        delta_color="off",
    )
    if result["input_issues"]:
        st.warning("Input issues: " + "; ".join(result["input_issues"]))
    with st.form("decision"):
        st.subheader("Operator decision")
        accept = st.radio("Decision", ["Accept", "Override"], horizontal=True) == "Accept"
        chosen = st.selectbox("Route", ROUTES, index=ROUTES.index(route), disabled=accept)
        reason = st.text_input("Reason (required to override)")
        operator = st.text_input("Operator ID", value="operator-1")
        if st.form_submit_button("Submit"):
            try:
                post_json(
                    "/decision",
                    {
                        "inspection_id": result["inspection_id"],
                        "unit_id": result["unit_id"],
                        "accepted": accept,
                        "route": route if accept else chosen,
                        "reason": reason or None,
                        "operator": operator,
                    },
                )
                st.success("Decision recorded in the audit trail.")
            except RuntimeError as exc:
                st.error(str(exc))
    st.caption(
        f"Model {result['model_version']}, data {str(result['data_version'])[:12]}. "
        f"{result['disclaimer']}"
    )
