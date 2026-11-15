"""
Streamlit demo: upload a supplier invoice PDF and get structured fields.

Uses the hybrid extractor (rules first, ML candidate ranker as fallback)
trained on SYNTHETIC invoices only. Optional "simulate a scan" runs the
page through image degradation + RapidOCR first.

    streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from extract_lib import FIELDS, RuleExtractor, arithmetic_ok, ocr_text, pdf_text  # noqa: E402


@st.cache_resource
def load_models():
    return RuleExtractor(), joblib.load(ROOT / "models" / "candidate_ranker.joblib")


def main() -> None:
    st.set_page_config(page_title="Invoice field extractor", page_icon="🧾", layout="wide")
    st.title("Invoice → structured data")
    st.caption("Portfolio demo. Trained and evaluated on SYNTHETIC invoices only.")
    st.warning("Do not upload real invoices containing personal data to a shared or public deployment.")

    rules, ml = load_models()
    samples = sorted((ROOT / "data" / "invoices_pdf").glob("*.pdf"))
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Upload an invoice PDF", type=["pdf"])
    with c2:
        pick = st.selectbox("…or pick a synthetic sample", [p.name for p in samples],
                            index=[p.name for p in samples].index("D_010.pdf") if samples else 0)
        scan = st.checkbox("Simulate a scanned copy (OCR)", value=False)

    if up is not None:
        tmp = Path(tempfile.mkstemp(suffix=".pdf")[1]); tmp.write_bytes(up.read()); pdf = tmp
    else:
        pdf = ROOT / "data" / "invoices_pdf" / pick
    with st.spinner("Reading the document…"):
        text = ocr_text(pdf, seed=0) if scan else pdf_text(pdf)
    r, m = rules.extract(text), ml.extract(text)
    out = {f: (r[f] if r[f] not in (None, "") else m[f]) for f in FIELDS}
    source = {f: ("rule" if r[f] not in (None, "") else "ML ranker") for f in FIELDS}

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Extracted fields")
        tab = pd.DataFrame({"field": FIELDS, "value": [out[f] for f in FIELDS], "found by": [source[f] for f in FIELDS]})
        st.dataframe(tab, hide_index=True, use_container_width=True)
        if arithmetic_ok(out):
            st.success("Arithmetic check passed: net + VAT = total.")
        else:
            st.error("Arithmetic check failed — send to a person for review.")
        st.download_button("Download as CSV", pd.DataFrame([out]).to_csv(index=False), "invoice_fields.csv")
    with right:
        st.subheader("Text the extractor saw")
        st.code(text[:3000], language=None)


if __name__ == "__main__":
    main()
