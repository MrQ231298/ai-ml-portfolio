"""
Streamlit demo: score one telecom customer's churn risk, explain it, and
show whether the cost-optimised policy would contact them.

Trained on the public IBM Telco Customer Churn sample (Apache 2.0).

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from churn_lib import BUNDLE_PATH, CAT_COLS, NUM_COLS  # noqa: E402


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH) if BUNDLE_PATH.exists() else None


@st.cache_resource
def load_explainer(_hgb):
    try:
        import shap
        return shap.TreeExplainer(_hgb.named_steps["clf"])
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Churn risk explainer", page_icon="📉", layout="wide")
    st.title("Customer churn risk — with reasons and a cost-based decision")
    st.caption("Portfolio demo on the public IBM Telco Customer Churn sample (7,043 customers).")
    st.info(
        "Gender and senior-citizen status are **not** model inputs; they are used only "
        "for a fairness audit in the README. Do not paste real customer records into a public demo."
    )

    b = load_bundle()
    if b is None:
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()

    with st.sidebar:
        st.header("Customer profile")
        row = {}
        defaults = {"Contract": "Month-to-month", "InternetService": "Fiber optic",
                    "PaymentMethod": "Electronic check", "OnlineSecurity": "No", "TechSupport": "No"}
        for c in CAT_COLS:
            opts = b["cat_choices"][c]
            row[c] = st.selectbox(c, opts, index=opts.index(defaults[c]) if c in defaults else 0)
        lo, hi, med = b["num_ranges"]["tenure"]
        row["tenure"] = st.slider("tenure (months)", int(lo), int(hi), 5)
        lo, hi, med = b["num_ranges"]["MonthlyCharges"]
        row["MonthlyCharges"] = st.slider("MonthlyCharges", float(lo), float(hi), 85.0)
        row["TotalCharges"] = float(row["tenure"] * row["MonthlyCharges"])
        st.caption(f"TotalCharges approximated as tenure x monthly = {row['TotalCharges']:.2f}")

    X = pd.DataFrame([row])[b["feature_cols"]]
    p = float(b["model"].predict_proba(X)[0, 1])
    t = b["threshold"]
    cfg = b["cost_cfg"]
    exp_value = p * cfg["save_rate"] * cfg["value_months"] * row["MonthlyCharges"] - cfg["offer_cost"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Calibrated churn probability", f"{p:.1%}")
    c2.metric("Cost-optimal contact threshold", f"{t:.2f}")
    c3.metric("Expected value of a retention offer", f"{exp_value:,.2f}")
    if p >= t:
        st.success("Policy: **contact** this customer with a retention offer.")
    else:
        st.warning("Policy: **do not contact** — expected saving is below the offer cost.")

    st.subheader("Why? Top factors for this customer (SHAP, log-odds)")
    hgb = b["hgb_uncalibrated"]
    explainer = load_explainer(hgb)
    if explainer is None:
        st.write("SHAP is not installed; see reports/permutation_importance.png instead.")
    else:
        prep = hgb.named_steps["prep"]
        Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
        sv = np.asarray(explainer.shap_values(Xt))
        sv = sv[..., 1] if sv.ndim == 3 else sv
        contrib = pd.Series(sv[0], index=Xt.columns)
        # Only show one-hot columns that are "on" for this customer, plus numerics.
        active = [c for c in Xt.columns if c in NUM_COLS or Xt.iloc[0][c] == 1]
        top = contrib[active].reindex(contrib[active].abs().sort_values(ascending=False).index).head(8)
        st.bar_chart(top.rename("SHAP (+ raises risk, - lowers risk)"), horizontal=True)
    st.caption(
        f"Assumptions: offer costs {cfg['offer_cost']:.0f}, {cfg['save_rate']:.0%} of contacted "
        f"would-be churners are saved, a saved customer is worth {cfg['value_months']} months of charges."
    )


if __name__ == "__main__":
    main()
