"""Dev demo for the ship3d component (uses synthetic DEMO data, not the real dataset).

Run from the repo root:
    streamlit run components/ship3d/demo_app.py

Pick the mode in the sidebar, or open with ?mode=hero|explorer|passenger.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.ship3d import ship3d  # noqa: E402

st.set_page_config(page_title="ship3d demo", layout="wide")
st.markdown(
    "<style>.stApp{background:#050b1a;color:#eaf0fa}</style>",
    unsafe_allow_html=True,
)


@st.cache_data
def demo_data(seed: int = 7):
    """Synthetic passengers shaped like the Titanic training set (DEMO data)."""
    rng = random.Random(seed)
    rates = {1: 0.63, 2: 0.47, 3: 0.24}
    counts = {1: 216, 2: 184, 3: 491}
    rows = []
    pid = 1
    for pclass, n in counts.items():
        for _ in range(n):
            actual = 1 if rng.random() < rates[pclass] else 0
            # a fake model that is right ~82% of the time
            prob = min(0.98, max(0.02, rng.gauss(0.72 if actual else 0.28, 0.18)))
            pred = 1 if prob >= 0.5 else 0
            rows.append([pid, pclass, actual, pred, round(prob, 3)])
            pid += 1
    rng.shuffle(rows)
    zones = []
    for pclass, label in [(1, "First class"), (2, "Second class"), (3, "Third class")]:
        sub = [r for r in rows if r[1] == pclass]
        surv = sum(r[2] for r in sub)
        zones.append({"id": f"class{pclass}", "label": label, "count": len(sub), "survived": surv, "rate": surv / len(sub)})
    stats = {"total": len(rows), "survived": sum(r[2] for r in rows),
             "accuracy": sum(r[2] == r[3] for r in rows) / len(rows)}
    return rows, zones, stats


rows, zones, stats = demo_data()
qp_mode = st.query_params.get("mode", "explorer")
modes = ["hero", "explorer", "passenger"]
mode = st.sidebar.radio("Mode", modes, index=modes.index(qp_mode) if qp_mode in modes else 1)
quality = st.sidebar.selectbox("Quality", ["auto", "high", "low"], index=["auto", "high", "low"].index(st.query_params.get("quality", "auto")))
tod = st.sidebar.selectbox("Initial time of day", ["night", "dusk", "day"])
st.sidebar.caption("All passenger numbers on this page are DEMO data.")

if mode == "hero":
    st.title("TITANIC ML LAB")
    st.caption("Hero banner · demo")
    ev = ship3d(mode="hero", height=560, time_of_day=tod, quality=quality, key="hero")
    st.write("event:", ev)

elif mode == "explorer":
    st.subheader("3D Ship Explorer · demo data")
    no_pred = st.sidebar.checkbox("Hide model predictions", value=False)
    pax = [[r[0], r[1], r[2], None, None] for r in rows] if no_pred else rows
    color_by = st.sidebar.selectbox("Initial colour-by", ["actual", "class", "predicted", "correct"])
    ev = ship3d(
        mode="explorer", height=560, zones=zones, passengers=pax, color_by=color_by,
        stats=stats, time_of_day=tod, quality=quality, key="explorer",
    )
    st.markdown("**Last event returned to Python:**")
    st.json(ev if ev is not None else {"event": None})
    if ev and ev.get("event") == "zone_click":
        z = next((z for z in zones if z["id"] == ev["zone"]), None)
        if z:
            st.success(f"Zone clicked: {z['label']} — {z['survived']}/{z['count']} survived ({z['rate']:.0%})")

else:
    st.subheader("Passenger prediction · demo")
    pclass = st.sidebar.selectbox("Passenger class", [1, 2, 3], index=0)
    prob = st.sidebar.slider("Estimated survival probability", 0.0, 1.0, 0.57, 0.01)
    ev = ship3d(
        mode="passenger", height=380,
        beacon={"pclass": pclass, "prob": prob, "label": "Your passenger"},
        time_of_day=tod, quality=quality, key="passenger",
    )
    st.write("event:", ev)
