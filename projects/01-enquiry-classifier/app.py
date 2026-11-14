"""
Streamlit demo: paste a customer enquiry and see the predicted routing class.

The model was trained only on SYNTHETIC example messages. Predictions on
real customer text are a demonstration of the workflow, not a production
decision engine.

Run from this folder (with the project virtualenv activated):

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "tfidf_logistic_regression.joblib"


@st.cache_resource
def load_bundle():
    """Load the joblib bundle once and cache it for the Streamlit session."""
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


def predict(bundle, text: str) -> pd.DataFrame:
    """Return a table of class probabilities, highest first."""
    # Same transform used at train time (the vectorizer is already fitted).
    X = bundle["vectorizer"].transform([text])
    proba = bundle["model"].predict_proba(X)[0]
    classes = list(bundle["model"].classes_)
    display = bundle["class_display"]
    rows = [
        {
            "class_id": cls,
            "class": display.get(cls, cls),
            "confidence": float(p),
        }
        for cls, p in zip(classes, proba)
    ]
    return pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)


def main() -> None:
    st.set_page_config(
        page_title="Tyre desk enquiry classifier (demo)",
        page_icon="🛞",
        layout="centered",
    )

    st.title("Tyre desk enquiry classifier")
    st.caption("Portfolio demo — trained on SYNTHETIC example data only.")

    st.info(
        "Every training message in this project is invented example text, "
        "not real customer or company correspondence. Do not paste personal "
        "data you would not want stored in your browser session."
    )

    bundle = load_bundle()
    if bundle is None:
        st.error(
            "Saved model not found. From the project folder, run "
            "`python src/generate_synthetic_data.py` and then `python src/train.py`, "
            "then reload this app."
        )
        st.stop()

    with st.expander("What this model does", expanded=False):
        st.markdown(
            """
            A sales desk receives mixed emails and messages. This baseline
            model suggests one of six routing classes:

            | Class | Typical intent |
            | --- | --- |
            | Quote request | Asking for a price or formal quote |
            | Stock / availability | Asking whether goods are on the shelf |
            | Order status | Chasing an existing order or delivery |
            | Complaint / return | Damage, wrong goods, warranty, returns |
            | Account / billing | Invoices, credit, statements, account stop |
            | Other | Hours, catalogues, account opening, everything else |

            It is a **TF-IDF + Logistic Regression** baseline. Confidence is
            the model's predicted probability for each class. High confidence
            is not the same as being correct — see the project README for
            limitations.
            """
        )

    examples = {
        "— choose an example —": "",
        "Quote request": (
            "Hi, can you price 8 x 225/45R17 Michelin summer tyres, "
            "trade rate, delivered to Harbour Road Motors please?"
        ),
        "Stock / availability": (
            "Have you got 4 x 205/55R16 all-season tyres in stock today "
            "for collection this afternoon?"
        ),
        "Order status": (
            "Any update on SO-48211? We were expecting the 315/80R22.5 "
            "steer tyres yesterday and the vehicle is on the ramp."
        ),
        "Complaint / return": (
            "The tyres on ORD-19304 arrived with scuffed sidewalls. "
            "Please arrange a collection and replacements today."
        ),
        "Account / billing": (
            "Can you resend INV-2026-4412? We think we've been double "
            "charged and accounts need a copy of the VAT breakdown."
        ),
        "Other": (
            "What are your opening hours on Saturday, and do you have a "
            "current trade catalogue you can email over?"
        ),
    }

    choice = st.selectbox("Or start from a synthetic example", list(examples.keys()))
    default_text = examples[choice]
    text = st.text_area(
        "Customer message",
        value=default_text,
        height=160,
        placeholder="Paste an enquiry here…",
    )

    clicked = st.button("Predict class", type="primary")
    if clicked or (choice != "— choose an example —" and text.strip()):
        if not text.strip():
            st.warning("Please paste or type a message first.")
            return

        results = predict(bundle, text.strip())
        top = results.iloc[0]
        st.subheader(top["class"])
        st.metric("Confidence", f"{top['confidence'] * 100:.1f}%")

        st.markdown("**All class probabilities**")
        chart_df = results.set_index("class")[["confidence"]]
        st.bar_chart(chart_df, horizontal=True)
        pretty = results[["class", "confidence"]].copy()
        pretty["confidence"] = pretty["confidence"].map(lambda x: f"{x * 100:.1f}%")
        st.dataframe(pretty, hide_index=True, use_container_width=True)

        if top["confidence"] < 0.45:
            st.warning(
                "Low confidence — a real desk would likely leave this for "
                "a person to route, or ask a clarifying question."
            )

    st.divider()
    st.caption(
        f"Model: {bundle['model_name']} · trained on {bundle['trained_on']} "
        f"· data_source={bundle['data_source']}"
    )


if __name__ == "__main__":
    main()
