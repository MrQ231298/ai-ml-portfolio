"""
Streamlit demo: ask questions about a folder of company documents.

Default corpus: the SYNTHETIC Brightwater Facilities document pack.
You can upload your own .md / .txt / .pdf files for a private, local
session (nothing is sent anywhere unless you configure the optional LLM).

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rag import DOCS_DIR, Index, answer, chunk_documents, load_threshold  # noqa: E402


@st.cache_resource(show_spinner="Building the index (first run downloads two small models)…")
def build_index(folder: str):
    return Index(chunk_documents(Path(folder)))


def main() -> None:
    st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📚", layout="wide")
    st.title("Ask your documents")
    st.caption("Retrieval-augmented Q&A: hybrid search + cross-encoder re-ranking, answers quoted from the source with citations.")

    with st.sidebar:
        st.header("Documents")
        ups = st.file_uploader("Upload your own (.md, .txt, .pdf)", type=["md", "txt", "pdf"], accept_multiple_files=True)
        if ups:
            folder = Path(tempfile.mkdtemp())
            for u in ups:
                (folder / u.name).write_bytes(u.read())
            st.success(f"Using {len(ups)} uploaded file(s) for this session.")
        else:
            folder = DOCS_DIR
            st.info("Using the SYNTHETIC Brightwater Facilities pack (fictional company).")
            for p in sorted(DOCS_DIR.iterdir()):
                st.write("•", p.name)
        thr = st.slider("Abstain if best match score is below", -12.0, 4.0, float(load_threshold()), 0.25,
                        help="Default was chosen on the dev questions in eval/questions.json.")
        llm = bool(os.environ.get("RAG_LLM_BASE_URL") and os.environ.get("RAG_LLM_MODEL"))
        st.write("Answer mode:", f"LLM ({os.environ.get('RAG_LLM_MODEL')})" if llm else "extractive (no LLM, no API key)")

    idx = build_index(str(folder))
    q = st.text_input("Question", "How long do I have to submit an expense claim?")
    if q:
        res = answer(idx, q, thr)
        if res["abstained"]:
            st.warning(f"I couldn't find this in the documents (best match score {res['best_score']:.2f}). "
                       "Try rephrasing, or check with the policy owner.")
        else:
            st.success(res["answer"])
            st.caption("Sources: " + "; ".join(res["citations"]) + f" · match score {res['best_score']:.2f}")
        with st.expander("Passages retrieved"):
            for p in res["passages"]:
                st.markdown(f"**{p['doc']} § {p['section']}** — {p['text']}")


if __name__ == "__main__":
    main()
