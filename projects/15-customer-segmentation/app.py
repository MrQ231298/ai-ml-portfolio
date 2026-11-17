"""Streamlit explorer for the customer segmentation results.

Run:  streamlit run app.py   (after python src/fetch_and_features.py && python src/segment.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))  # so the pickled preprocessing function can be found
from seg_lib import FEATURES  # noqa: E402

REPORTS, MODELS = HERE / "reports", HERE / "models"

st.set_page_config(page_title="Customer segmentation", layout="wide")


@st.cache_data
def load():
    m = json.loads((REPORTS / "metrics.json").read_text())
    grid = pd.read_csv(REPORTS / "k_selection.csv")
    prof = pd.read_csv(REPORTS / "segment_profiles.csv")
    cust = pd.read_csv(REPORTS / "customer_segments.csv")
    cross = pd.read_csv(REPORTS / "kmeans_vs_rfm_crosstab.csv", index_col=0)
    return m, grid, prof, cust, cross


@st.cache_resource
def model():
    return joblib.load(MODELS / "segmenter.joblib")


m, grid, prof, cust, cross = load()
hv = m["holdout_validation"]
st.title("Customer segmentation: UK online gift retailer (UCI Online Retail, CC BY 4.0)")
st.caption(f"{m['data']['customers']:,} customers. Features from {m['data']['calibration_period'][0]} to "
           f"{m['data']['calibration_period'][1]}; checked against what they did from {m['data']['holdout_period'][0]} "
           f"to {m['data']['holdout_period'][1]}. All numbers come from reports/metrics.json.")

c = st.columns(4)
c[0].metric("Segments chosen (k)", m["k_selected"])
c[1].metric("Silhouette", f"{m['selected']['silhouette']:.3f}")
c[2].metric("Bootstrap stability (ARI)", f"{m['selected']['bootstrap_ari_mean']:.2f} ± {m['selected']['bootstrap_ari_sd']:.2f}")
c[3].metric("Bought again in hold-out", f"{hv['buyers_share']:.1%}")

t1, t2, t3, t4 = st.tabs(["Segments", "Choosing k", "Hold-out check", "Score a customer"])

with t1:
    show = prof[["name", "customers", "customer_share", "recency_days", "frequency", "monetary", "tenure_days",
                 "cancel_rate", "revenue_share", "future_90d_purchase_rate", "future_90d_revenue_share"]].copy()
    st.dataframe(show.style.format({"customer_share": "{:.1%}", "revenue_share": "{:.1%}", "cancel_rate": "{:.1%}",
                                    "future_90d_purchase_rate": "{:.1%}", "future_90d_revenue_share": "{:.1%}",
                                    "monetary": "£{:,.0f}"}), hide_index=True, use_container_width=True)
    st.caption("Medians per segment, except cancel_rate (mean) and the share columns.")
    long = prof.melt(id_vars="name", value_vars=["customer_share", "revenue_share", "future_90d_revenue_share"],
                     var_name="measure", value_name="share")
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("share:Q", axis=alt.Axis(format="%")), y=alt.Y("name:N", title=None), color="measure:N",
        yOffset="measure:N").properties(height=260), use_container_width=True)
    st.image(str(REPORTS / "segments_pca.png"), caption="Segments on the first two principal components", width=650)

with t2:
    st.write(f"**Rule (fixed before running):** {m['selection_rule']}.")
    gl = grid.melt(id_vars="k", value_vars=["silhouette", "bootstrap_ari_mean", "davies_bouldin"])
    st.altair_chart(alt.Chart(gl).mark_line(point=True).encode(x="k:O", y="value:Q", color="variable:N")
                    .properties(height=300), use_container_width=True)
    st.dataframe(grid.round(3), hide_index=True, use_container_width=True)
    st.warning(f"k = 4 only just passes the stability bar (mean ARI {m['selected']['bootstrap_ari_mean']:.2f}, "
               f"sd {m['selected']['bootstrap_ari_sd']:.2f}); across 10 random starts the lowest agreement was "
               f"{m['seed_stability_ari_min']:.2f}. Treat the boundaries as soft.")

with t3:
    auc = pd.DataFrame({"method": list(hv["auc_predict_purchase_next_90d"].keys()),
                        "AUC": list(hv["auc_predict_purchase_next_90d"].values())})
    st.altair_chart(alt.Chart(auc).mark_bar().encode(x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.5, 0.8])),
                                                     y=alt.Y("method:N", sort="-x", title=None)).properties(height=200),
                    use_container_width=True)
    st.write(f"Share of variation in next-90-day revenue explained by segment (η²): k-means "
             f"{hv['eta_squared_future_revenue']['kmeans']:.3f}, RFM rules {hv['eta_squared_future_revenue']['rfm_rules']:.3f}.")
    st.info("The simple RFM rules predict who buys next quarter better than the k-means segments. "
            "k-means is useful for describing customers, not as a replacement for a scoring rule.")
    st.write("How the k-means segments line up with the RFM rule segments:")
    st.dataframe(cross, use_container_width=True)

with t4:
    bundle = model()
    st.write("Enter a customer's activity over the calibration window to see which segment the saved model gives.")
    cols = st.columns(4)
    defaults = cust[FEATURES].median()
    vals = {}
    for i, f in enumerate(FEATURES):
        step = 0.01 if f == "cancel_rate" else 1.0
        vals[f] = cols[i % 4].number_input(f, min_value=0.0, value=float(defaults[f]), step=step)
    X = np.array([[vals[f] for f in FEATURES]])
    seg = int(bundle["kmeans"].predict(bundle["preprocess"].transform(X))[0])
    st.success(f"Segment: **{bundle['names'][seg]}**")
