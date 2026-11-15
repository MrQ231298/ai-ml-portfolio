"""
Streamlit dashboard: Irish retail sales by sector - history, back-test
accuracy and a 6-month forecast with an empirical 80% interval.

Data: CSO Retail Sales Index (RSM05 + RSM08, value, unadjusted), CC BY 4.0.

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


@st.cache_data
def load():
    hist = pd.read_csv(ROOT / "data" / "irish_retail_sales_index_monthly.csv", parse_dates=["date"])
    bt = pd.read_csv(REPORTS / "backtest_forecasts.csv", parse_dates=["origin", "target_date"])
    fut = pd.read_csv(REPORTS / "future_forecast.csv", parse_dates=["date"])
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    return hist, bt, fut, metrics


def main() -> None:
    st.set_page_config(page_title="Irish retail sales forecast", page_icon="📈", layout="wide")
    st.title("Irish retail sales — sector forecasts")
    st.caption("CSO Retail Sales Index (value, not seasonally adjusted; chain-linked to the 2021 base). "
               "Source: Central Statistics Office, CC BY 4.0. Index values, not euro amounts.")
    if not (REPORTS / "metrics.json").exists():
        st.error("Run `python src/fetch_cso_retail.py` and `python src/backtest.py` first."); st.stop()
    hist, bt, fut, m = load()
    best = m["best_model"]

    series = sorted(hist.series.unique())
    with st.sidebar:
        s = st.selectbox("Retail sector", series, index=series.index("Clothing, footwear & textiles"))
        start = st.slider("Show history from", 2015, 2025, 2021)
        h = st.select_slider("Back-test horizon to display (months ahead)", options=list(range(1, 7)), value=1)
        models = st.multiselect("Back-test models", sorted(bt.model.unique()), default=[best, "seasonal_naive"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model used for the forecast", best)
    c2.metric("Back-test MASE (all sectors)", f"{m['overall'][best]['MASE']:.3f}")
    c3.metric("Seasonal-naive MASE", f"{m['overall']['seasonal_naive']['MASE']:.3f}")
    c4.metric("80% interval coverage (later origins)", f"{m['interval_80_coverage_later_half']:.0%}")

    hs = hist[(hist.series == s) & (hist.date.dt.year >= start)].assign(kind="actual (CSO)")
    fs = fut[fut.series == s]
    bts = bt[(bt.series == s) & (bt.h == h) & (bt.model.isin(models))]
    base = alt.Chart(hs).mark_line(color="black").encode(x="date:T", y=alt.Y("value:Q", title="index (2021=100)"))
    band = alt.Chart(fs).mark_area(opacity=0.25, color="#2b6cb0").encode(x="date:T", y="lo80:Q", y2="hi80:Q")
    fc = alt.Chart(fs).mark_line(color="#2b6cb0", point=True).encode(x="date:T", y="forecast:Q")
    btl = alt.Chart(bts).mark_line(strokeDash=[4, 3]).encode(
        x=alt.X("target_date:T", title="month"), y="forecast:Q", color=alt.Color("model:N", title=f"{h}-month-ahead back-test"))
    st.altair_chart((base + band + fc + btl).properties(height=420), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Next 6 months")
        st.dataframe(fs[["date", "forecast", "lo80", "hi80"]].assign(date=fs.date.dt.strftime("%b %Y")).round(1),
                     hide_index=True, use_container_width=True)
    with right:
        st.subheader(f"Back-test MASE for {s}")
        per = pd.Series(m["mase_by_series"][s]).sort_values().rename("MASE").to_frame()
        st.dataframe(per.round(3), use_container_width=True)
    st.info("MASE < 1 means the model's out-of-sample error was smaller than the typical in-sample "
            "error of a 'same month last year' forecast. Intervals come from past back-test errors, "
            "not from a statistical model, and covered the actual value less often than 80%.")


if __name__ == "__main__":
    main()
