"""Fleet health demo: replay a held-out engine cycle by cycle and see when each policy would alert.

Reads reports/ written by src/run.py.   streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REP = Path(__file__).resolve().parent / "reports"

st.set_page_config(page_title="Predictive maintenance", layout="wide")
st.title("Predictive maintenance on NASA C-MAPSS turbofan engines")
st.caption("Simulated run-to-failure data (public domain). Engines below were never used to fit the models.")

metrics = json.loads((REP / "metrics.json").read_text())
fd = st.sidebar.selectbox("Sub-dataset", [k for k in metrics if k.startswith("FD")])
tr = pd.read_csv(REP / f"engine_traces_{fd}.csv")
m = metrics[fd]
th_rul = m["policies"]["predicted_rul_alert"]["alert_when_predicted_rul_le"]
th_h = m["policies"]["health_score_alert"]["threshold"]
fixed = m["policies"]["fixed_interval"]["maintenance_cycle"]
lead = m["costs_assumed"]["lead_time"]

unit = st.sidebar.selectbox("Held-out engine", sorted(tr[tr.split == "test"].unit.unique()))
e = tr[tr.unit == unit].reset_index(drop=True)
cycle = st.sidebar.slider("Replay up to cycle", 1, int(e.cycle.max()), int(e.cycle.max()))
now = e[e.cycle <= cycle]
row = now.iloc[-1]

c = st.columns(4)
c[0].metric("Predicted RUL", f"{row.pred_rul:.0f} cycles")
c[1].metric("Health score", f"{row.health_score:.3f}", help=f"alert threshold {th_h:.3f} (3 cycles in a row)")
alert_rul = bool((now.pred_rul <= th_rul).any())
run3 = (now.health_score >= th_h).astype(int).groupby((now.health_score < th_h).cumsum()).cumsum().max() >= 3
c[2].metric("RUL policy", "ALERT" if alert_rul else "ok")
c[3].metric("Health-score policy", "ALERT" if run3 else "ok")
st.write(f"Fixed-interval schedule would service this engine at cycle **{fixed}**; it actually fails at cycle "
         f"**{int(e.cycle.max())}**. Maintenance needs **{lead}** cycles' notice.")
st.line_chart(now.set_index("cycle")[["pred_rul"]].assign(alert_level=th_rul))
st.line_chart(now.set_index("cycle")[["health_score"]].assign(threshold=th_h))
sens = [s for s in e.columns if s.startswith("s") and s[1:].isdigit()]
st.line_chart(now.set_index("cycle")[sens])
