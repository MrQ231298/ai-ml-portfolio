"""
Streamlit demo: estimate a published RTB average monthly rent.

The figures behind this app are official averages (CSO / RTB table
RIA02), licensed CC BY 4.0 — not asking prices for a named property,
and not a valuation. This is not financial, legal or letting advice.

Run from this folder (virtualenv activated):

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# joblib needs PersistenceBaseline on the path (same module name as at train time).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from baseline import PersistenceBaseline  # noqa: E402,F401

BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "rent_predictor.joblib"


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


def predict_all(bundle, row: pd.DataFrame) -> dict[str, float]:
    out = {}
    for name, model in bundle["models"].items():
        pred = model.predict(row)[0]
        out[name] = float(pred)
    return out


def main() -> None:
    st.set_page_config(
        page_title="Irish rent predictor (demo)",
        page_icon="🏠",
        layout="centered",
    )
    st.title("Irish rent predictor")
    st.caption(
        "Estimates the published RTB *average* monthly rent for a "
        "location, property type and bedroom count."
    )

    st.warning(
        "**Not a valuation, and not letting, financial or legal advice.** "
        "The model predicts official *average* rents from the Residential "
        "Tenancies Board, hosted by the CSO (table RIA02, CC BY 4.0). "
        "It cannot tell you what a specific dwelling should list or let for."
    )

    bundle = load_bundle()
    if bundle is None:
        st.error(
            "Saved model not found. From this folder run "
            "`python src/fetch_cso_ria02.py` and `python src/train.py`, "
            "then reload."
        )
        st.stop()

    with st.expander("Data source and credit", expanded=False):
        src = bundle.get("source", {})
        st.markdown(
            f"""
Based on the **RTB Average Monthly Rent Report (table RIA02)**,
hosted by the Central Statistics Office and licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- API: {src.get("api_url", "")}
- Attribution: {src.get("attribution", "")}

County-wide totals and overlapping bedroom bands were dropped before
training. The numbers you see here are model *estimates of those
published averages*, not a new official statistic.
"""
        )

    counties = bundle["counties"]
    loc_to_county = bundle["location_to_county"]
    loc_to_level = bundle["location_to_level"]
    default_county = "Dublin" if "Dublin" in counties else counties[0]
    county = st.selectbox("County", counties, index=counties.index(default_county))
    locations = sorted(loc for loc, cty in loc_to_county.items() if cty == county)
    # Prefer a well-known Dublin example if present.
    preferred = "Ranelagh, Dublin 6"
    loc_index = locations.index(preferred) if preferred in locations else 0
    location = st.selectbox("Location (town, neighbourhood or postal district)", locations, index=loc_index)

    types = bundle["property_types"]
    type_index = types.index("Apartment") if "Apartment" in types else 0
    property_type = st.selectbox("Property type", types, index=type_index)

    bedroom_map = bundle["bedroom_map"]
    bedroom_labels = list(bedroom_map.keys())
    bed_index = bedroom_labels.index("Two bed") if "Two bed" in bedroom_labels else 0
    bedrooms = st.selectbox("Bedrooms", bedroom_labels, index=bed_index)

    year = st.slider(
        "Year",
        min_value=int(bundle["year_min"]),
        max_value=max(int(bundle["year_max"]) + 1, 2026),
        value=int(bundle["year_max"]),
        help="The test hold-out used 2024–2025. Years after the last "
        "published table are an extrapolation.",
    )

    row = pd.DataFrame(
        [
            {
                "location": location,
                "county": loc_to_county[location],
                "location_level": loc_to_level[location],
                "property_type": property_type,
                "year": int(year),
                "n_bedrooms": int(bedroom_map[bedrooms]),
            }
        ]
    )

    if st.button("Estimate average monthly rent", type="primary") or True:
        preds = predict_all(bundle, row)
        display = {
            "persistence_baseline": "Persistence baseline (last observed training year)",
            "linear_regression": "Log-linear regression",
            "gradient_boosting": "Gradient boosting",
        }
        headline_key = bundle.get("model_name", "persistence_baseline")
        primary = preds[headline_key]
        st.subheader(f"€{primary:,.0f} per month")
        st.caption(
            f"{display.get(headline_key, headline_key)} estimate of the published "
            f"average for {bedrooms.lower()} {property_type.lower()}s in "
            f"{location}, {year}. The headline model is the one with the "
            f"lowest mean absolute error on the 2024–2025 hold-out."
        )

        compare = pd.DataFrame(
            {
                "model": [display[k] for k in preds],
                "estimated_avg_rent_eur": [round(preds[k], 0) for k in preds],
                "headline": ["Yes" if k == headline_key else "" for k in preds],
            }
        )
        st.markdown("**All three models**")
        st.dataframe(compare, hide_index=True, use_container_width=True)

        m = bundle["metrics"]["models"]
        st.info(
            "On the 2024–2025 hold-out: "
            f"persistence MAE **€{m['persistence_baseline']['mae']:.0f}**, "
            f"log-linear MAE €{m['linear_regression']['mae']:.0f}, "
            f"gradient boosting MAE €{m['gradient_boosting']['mae']:.0f}. "
            "Treat a single estimate as a range of roughly that width, "
            "not as an exact figure. In this run the simple baseline was "
            "the most accurate of the three."
        )

    st.divider()
    st.caption(
        "Source: RTB Average Monthly Rent Report (RIA02), hosted by the CSO, "
        "CC BY 4.0. Models trained on 2008–2023 cells; headline pick is the lowest hold-out MAE."
    )


if __name__ == "__main__":
    main()
