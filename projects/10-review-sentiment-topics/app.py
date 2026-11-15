"""
Streamlit demo: score a customer review for sentiment and topic, and
explore the topic insights computed on 22,641 real clothing reviews.

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load():
    s = joblib.load(ROOT / "models" / "sentiment_tfidf.joblib")
    t = joblib.load(ROOT / "models" / "topics_nmf.joblib")
    summary = pd.read_csv(ROOT / "reports" / "topic_summary.csv")
    sm = json.loads((ROOT / "reports" / "sentiment_metrics.json").read_text())
    return s, t, summary, sm


def main() -> None:
    st.set_page_config(page_title="Review sentiment & topics", page_icon="💬", layout="wide")
    st.title("Customer review sentiment & topics")
    st.caption("Trained on the public Women's E-Commerce Clothing Reviews dataset (22,641 reviews, CC0).")
    s, t, summary, sm = load()
    best = sm["best_model"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test macro-F1 (3 classes)", f"{sm['test'][best]['macro_f1']:.3f}")
    c2.metric("Negative vs positive ROC-AUC", f"{sm['binary_neg_vs_pos_roc_auc_best']:.3f}")
    c3.metric("Topics (NMF)", f"{len(summary)}")

    txt = st.text_area("Paste a review", "Beautiful colour but the fabric is thin and it runs very small. "
                       "I had to send it back.")
    if txt.strip():
        proba = s["model"].predict_proba([txt])[0]
        classes = list(s["model"].classes_)
        w = t["nmf"].transform(t["vectorizer"].transform([txt]))[0]
        left, right = st.columns(2)
        with left:
            st.subheader(f"Sentiment: {classes[int(np.argmax(proba))]}")
            st.bar_chart(pd.Series(proba, index=classes, name="probability"))
        with right:
            st.subheader("Closest topics")
            if w.sum() == 0:
                st.write("No topic words recognised.")
            else:
                top = np.argsort(-w)[:3]
                for k in top:
                    if w[k] > 0:
                        st.write(f"**T{k}** ({w[k] / w.sum():.0%}): {', '.join(t['terms'][k][:8])}")

    st.subheader("Topic insights (dominant topic per review)")
    show = summary[["topic", "top_terms", "share", "mean_rating", "negative_share", "negative_lift", "recommend_rate", "top_departments"]]
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption("Negativity lift = topic's share of 1–2 star reviews divided by the corpus share. "
               "Topics show where to read first; they do not prove causes.")


if __name__ == "__main__":
    main()
