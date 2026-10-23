"""Quality Engineer dashboard: throughput, route mix, control chart, drift, calibration."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from preinspect.app.client import get
from preinspect.report.style import INK, SERIES

st.set_page_config(page_title="Quality Engineer", layout="wide")
st.title("Quality Engineer")
window = st.slider("Recent inspections to analyse", 40, 2000, 300, step=20)
data = get("/monitoring", window=window)
if not data.get("inspections"):
    st.info("No inspections yet. Run `make replay` to stream test units through the service.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Inspections", data["inspections"])
rate = data.get("throughput_per_hour")
c2.metric("Throughput (units/h)", f"{rate:.0f}" if rate else "n/a")
c3.metric("Operator overrides", data["operator_overrides"])

left, right = st.columns(2)
left.subheader("Route mix")
left.bar_chart(pd.Series(data["route_mix"]), horizontal=True)
right.subheader("p-chart: REFER + QUARANTINE share per batch")
pc = pd.DataFrame(data["p_chart"])
if len(pc):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(pc["batch"], pc["p"], marker="o", color=SERIES[0])
    for col, style in (("centre", "-"), ("ucl", "--"), ("lcl", "--")):
        ax.plot(pc["batch"], pc[col], color=INK["muted"], linestyle=style, linewidth=1)
    out = pc[pc["out_of_control"]]
    ax.scatter(out["batch"], out["p"], color=SERIES[7], zorder=3, label="outside limits")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Share")
    right.pyplot(fig)
else:
    right.info("Not enough inspections for one batch yet.")

st.subheader("Input drift against the training data (PSI)")
d1, d2 = st.columns(2)
d1.caption("Service metadata")
d1.dataframe(pd.DataFrame(data["metadata_drift"]).round(3), hide_index=True)
d2.caption("Image embedding (principal components of the perception encoder)")
d2.dataframe(pd.DataFrame(data["embedding_drift"]).round(3), hide_index=True)
st.caption("PSI below 0.10 stable, 0.10 to 0.25 watch, above 0.25 alert (conventional bands).")

st.subheader("Calibration over time (needs downstream results)")
cal = pd.DataFrame(data["calibration"])
if len(cal):
    st.line_chart(cal.set_index("batch")[["confidence", "accuracy"]])
else:
    st.info("No downstream results logged yet.")

st.subheader("Line route queue (mock MES)")
st.dataframe(pd.DataFrame(get("/mes/queue")).head(50), hide_index=True)
if st.button("Verify audit trail"):
    v = get("/audit/verify")
    (st.success if v["ok"] else st.error)(f"{v['records']} records: {v['message']}")
