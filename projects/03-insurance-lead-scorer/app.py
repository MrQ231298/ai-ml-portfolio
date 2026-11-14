"""
Streamlit demo: score a SYNTHETIC commercial insurance lead.

This is not an underwriting engine, not a pricing tool, and not a
recommendation to offer or refuse cover. Training data is invented.

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "lead_scorer.joblib"


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


def main() -> None:
    st.set_page_config(
        page_title="Insurance lead scorer (demo)",
        page_icon="📋",
        layout="centered",
    )
    st.title("Commercial insurance lead scorer")
    st.caption("Portfolio demo — trained on SYNTHETIC example data only.")

    st.warning(
        "**Not underwriting, not pricing, and not a recommendation to "
        "offer or refuse cover.** This model estimates the chance that a "
        "*fictional* sales lead would convert. It was not trained on real "
        "prospects. Do not use it on live customer or applicant data."
    )

    bundle = load_bundle()
    if bundle is None:
        st.error(
            "Saved model not found. From this folder run "
            "`python src/generate_synthetic_leads.py` and `python src/train.py`, "
            "then reload."
        )
        st.stop()

    choices = bundle["choices"]
    ranges = bundle["numeric_ranges"]

    c1, c2 = st.columns(2)
    with c1:
        source = st.selectbox("Lead source", choices["source"], index=choices["source"].index("web_form") if "web_form" in choices["source"] else 0)
        coverage_type = st.selectbox(
            "Coverage type",
            choices["coverage_type"],
            index=choices["coverage_type"].index("combined_package") if "combined_package" in choices["coverage_type"] else 0,
        )
        industry = st.selectbox("Industry", choices["industry"])
        region = st.selectbox("Region (Ireland)", choices["region"])
        revenue_band = st.selectbox("Revenue band", choices["revenue_band"])
    with c2:
        n_employees = st.slider("Employees", ranges["n_employees"]["min"], min(200, ranges["n_employees"]["max"]), 12)
        years_trading = st.slider("Years trading", ranges["years_trading"]["min"], ranges["years_trading"]["max"], 8)
        n_locations = st.slider("Number of locations", ranges["n_locations"]["min"], min(8, ranges["n_locations"]["max"]), 1)
        response_time_hours = st.slider(
            "Hours until first follow-up",
            0.5,
            72.0,
            6.0,
            step=0.5,
        )
        has_claims_last_3y = st.checkbox("Claims in the last 3 years", value=False)

    row = pd.DataFrame(
        [
            {
                "industry": industry,
                "revenue_band": revenue_band,
                "coverage_type": coverage_type,
                "source": source,
                "region": region,
                "n_employees": int(n_employees),
                "years_trading": int(years_trading),
                "n_locations": int(n_locations),
                "response_time_hours": float(response_time_hours),
                "has_claims_last_3y": int(has_claims_last_3y),
            }
        ]
    )

    # Always show a score so a screenshot captures a prediction.
    primary = bundle["model_name"]
    display = {
        "dummy_prior": "Dummy (training prevalence)",
        "logistic_regression": "Logistic regression",
        "gradient_boosting": "Gradient boosting",
    }
    preds = {name: float(model.predict_proba(row)[0, 1]) for name, model in bundle["models"].items()}
    headline = preds[primary]
    st.subheader(f"{headline:.0%} conversion chance")
    st.caption(
        f"{display.get(primary, primary)} estimate for this fictional lead. "
        "The headline model is the one with the highest PR-AUC on the held-out test set."
    )
    st.progress(min(max(headline, 0.0), 1.0))

    table = pd.DataFrame(
        {
            "model": [display[k] for k in preds],
            "estimated_conversion_probability": [round(preds[k], 3) for k in preds],
            "headline": ["Yes" if k == primary else "" for k in preds],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    m = bundle["metrics"]["models"]
    st.info(
        "On the SYNTHETIC test set: "
        f"{display[primary]} ROC-AUC **{m[primary]['roc_auc']:.3f}**, "
        f"PR-AUC **{m[primary]['pr_auc']:.3f}**, "
        f"precision@50 **{m[primary]['precision_at_k']['50']:.0%}**. "
        "These scores describe invented data, not a live sales book."
    )

    st.divider()
    st.caption(
        "Training data: SYNTHETIC_commercial_insurance_leads.csv "
        "(SYNTHETIC_EXAMPLE_DATA). Not real prospects."
    )


if __name__ == "__main__":
    main()
