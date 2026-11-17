"""
Streamlit equivalent of the Power BI manufacturing KPI dashboard.
SYNTHETIC data for a fictional automotive electronics plant.

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import kpi_lib as K  # noqa: E402


@st.cache_data
def load():
    t = K.load_star()
    return t, K.production_view(t)


def card(col, label, value, target=None):
    delta = None if target is None else f"{(value - target) * 100:+.1f} pp vs {target:.0%}"
    col.metric(label, f"{value:.1%}", delta)


def main() -> None:
    st.set_page_config(page_title="Plant KPI dashboard", page_icon="🏭", layout="wide")
    st.title("Plant KPI dashboard — automotive electronics (SYNTHETIC)")
    st.caption("Fictional plant: two SMT lines → final assembly → end-of-line test, three shifts, Jul–Oct 2026. "
               "All numbers are invented example data. Same measures as `powerbi/measures.dax`.")
    if not (K.STAR / "fact_production.csv").exists():
        st.error("Run `python src/generate_data.py` first."); st.stop()
    t, p = load()

    with st.sidebar:
        st.header("Filters")
        dmin, dmax = p["date"].min().date(), p["date"].max().date()
        rng = st.date_input("Dates", (dmin, dmax), min_value=dmin, max_value=dmax)
        lines = st.multiselect("Line", sorted(p["line"].unique()), default=sorted(p["line"].unique()))
        shifts = st.multiselect("Shift", ["Early", "Late", "Night"], default=["Early", "Late", "Night"])
        products = st.multiselect("Product", sorted(p["product_code"].unique()), default=sorted(p["product_code"].unique()))
    if not (lines and shifts and products) or len(rng) != 2:
        st.warning("Pick at least one line, shift and product, and a start and end date."); st.stop()
    start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    f = p[p["date"].between(start, end) & p["line"].isin(lines) & p["shift"].isin(shifts) & p["product_code"].isin(products)]
    if f.empty:
        st.warning("No production in this selection."); st.stop()
    k = K.kpis(f)

    c = st.columns(6)
    card(c[0], "OEE", k["oee"], 0.85)
    card(c[1], "Availability", k["availability"])
    card(c[2], "Performance", k["performance"])
    card(c[3], "Quality (FPY)", k["quality"])
    card(c[4], "Output vs target", k["output_vs_target"], 1.0)
    c[5].metric("Unplanned downtime", f"{k['unplanned_downtime_minutes'] / 60:,.0f} h")

    left, right = st.columns(2)
    wk = K.kpis_by(f.assign(week=f["date"].dt.to_period("W-SUN").dt.start_time), ["week", "line"])
    trend = alt.Chart(wk).mark_line(point=True).encode(
        x=alt.X("week:T", title="Week starting"), y=alt.Y("oee:Q", title="OEE", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
        color=alt.Color("line:N", title="Line"), tooltip=["line", "week:T", alt.Tooltip("oee:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"y": [0.85]})).mark_rule(strokeDash=[4, 4], color="grey").encode(y="y:Q")
    left.subheader("Weekly OEE by line (dashed = 85%)")
    left.altair_chart(trend + rule, width="stretch")

    keys = dict(line_keys=t["dim_line"].loc[t["dim_line"]["line"].isin(lines), "line_key"],
                date_keys=f["date_key"].unique(),
                shift_keys=t["dim_shift"].loc[t["dim_shift"]["shift"].isin(shifts), "shift_key"])
    par = K.downtime_pareto(t, **keys)
    par["hours"] = par["minutes"] / 60
    base = alt.Chart(par).encode(x=alt.X("reason:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=200)))
    bars = base.mark_bar(color="#3b6ea5").encode(y=alt.Y("hours:Q", title="Downtime (h)"),
                                                  tooltip=["reason", alt.Tooltip("hours:Q", format=".1f"), "events"])
    line = base.mark_line(color="#d1495b", point=True).encode(
        y=alt.Y("cumulative_share:Q", title="Cumulative %", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])))
    right.subheader("Downtime Pareto")
    right.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), width="stretch")

    a, b = st.columns(2)
    by_line = K.kpis_by(f, "line")
    out = by_line.melt(id_vars="line", value_vars=["good_count", "target_units"], var_name="measure", value_name="units")
    a.subheader("Good output vs target")
    a.altair_chart(alt.Chart(out).mark_bar().encode(
        y=alt.Y("line:N", title=None), x=alt.X("units:Q", title="Units"), yOffset="measure:N",
        color=alt.Color("measure:N", scale=alt.Scale(range=["#00798c", "#c9c9c9"]), title=None)), width="stretch")
    m = K.kpis_by(f, ["line", "shift"])
    b.subheader("OEE: line × shift")
    heat = alt.Chart(m).mark_rect().encode(
        x=alt.X("shift:N", sort=["Early", "Late", "Night"], title=None, axis=alt.Axis(labelAngle=0, orient="top")), y=alt.Y("line:N", title=None),
        color=alt.Color("oee:Q", scale=alt.Scale(scheme="redyellowgreen"), legend=None)).properties(height=300)
    text = heat.mark_text(fontSize=14).encode(text=alt.Text("oee:Q", format=".1%"), color=alt.value("black"))
    b.altair_chart(heat + text, width="stretch")

    with st.expander("Loss breakdown and quality detail"):
        loss = K.oee_losses_minutes(f)
        st.dataframe(pd.DataFrame({"minutes": loss}).assign(share=lambda d: d["minutes"] / loss["planned_production"]).style.format({"minutes": "{:,.0f}", "share": "{:.1%}"}))
        st.write(f"Yield after rework {k['yield_after_rework']:.2%} · scrap rate {k['scrap_rate']:.2%} · "
                 f"rolled throughput yield {K.rty(f):.2%} (only meaningful with all route steps selected)")


if __name__ == "__main__":
    main()
