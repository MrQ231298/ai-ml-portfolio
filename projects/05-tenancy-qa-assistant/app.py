"""
Streamlit demo: extractive Q&A over a snapshot of public Irish tenancy guidance.

    streamlit run app.py

This app does not generate legal advice. It retrieves the closest passages
from pages fetched from Citizens Information and the RTB, and shows their
source URL and fetch date.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from retrieve import available_backends, load_index, search  # noqa: E402

DISCLAIMER = (
    "**This is not legal advice.** The answers below are **extractive "
    "passages** copied from a snapshot of public guidance pages. They are "
    "for an educational portfolio demo only. Irish rental law can change "
    "(it did, for example, on 14 September 2026). Always read the live "
    "source page, and get advice from the RTB, Citizens Information, "
    "Threshold, or a qualified solicitor if you have a real case."
)

SAMPLE_QUESTIONS = [
    "How much can a landlord charge as a security deposit in Ireland?",
    "What is the annual RTB registration fee for a private rented tenancy?",
    "What changed on 14 September 2026 about how a landlord may send a notice of termination?",
    "How many days' notice must a landlord give to end a tenancy that has lasted less than six months?",
    "Can a landlord end a tenancy in the first six months without giving a reason?",
    "Do I have to register with the RTB if I only rent a room in my own home to a lodger?",
]


@st.cache_resource
def _index():
    try:
        return load_index()
    except FileNotFoundError:
        return None


def main() -> None:
    st.set_page_config(page_title="Tenancy Q&A assistant (demo)", page_icon="🏠", layout="wide")
    st.title("Irish tenancy guidance Q&A")
    st.caption(
        "Portfolio demo — retrieval over a public Citizens Information and RTB snapshot. "
        "Not a solicitor and not the RTB."
    )

    st.error(DISCLAIMER)

    index = _index()
    if index is None:
        st.warning("Index not found. Run `python src/fetch_pages.py` and `python src/chunk_and_index.py`, then reload.")
        st.stop()

    backends = available_backends(index)
    preferred = index.get("preferred_backend") or backends[0]

    with st.sidebar:
        st.subheader("About this snapshot")
        st.write(f"Pages indexed: **{index['n_pages']}**")
        st.write(f"Passages: **{index['n_chunks']}**")
        st.write(f"Index built: **{index.get('built_at_dublin', 'unknown')}**")
        label = {
            "minilm": "MiniLM (all-MiniLM-L6-v2)",
            "tfidf": "TF-IDF baseline",
        }
        backend = st.radio(
            "Retrieval backend",
            backends,
            index=backends.index(preferred) if preferred in backends else 0,
            format_func=lambda b: label.get(b, b) + (" — default" if b == preferred else ""),
        )
        st.caption(
            "Passages are ranked by cosine similarity. The app does not call a paid API "
            "and does not invent wording that is not in the snapshot."
        )
        k = st.slider("Passages to show", min_value=1, max_value=8, value=3)

    st.subheader("Ask a question")
    preset = st.selectbox("Try a sample question, or write your own below", [""] + SAMPLE_QUESTIONS)
    query = st.text_input("Your question", value=preset or "")
    go = st.button("Find passages", type="primary")

    if go or (query and preset and query == preset):
        hits = search(index, query, k=k, backend=backend)
        if not hits:
            st.info("Type a question to retrieve passages.")
            return
        st.subheader("Top passages")
        st.caption("Quoted from the snapshot, with the source link and the date the page was fetched.")
        for hit in hits:
            with st.container(border=True):
                st.markdown(f"**{hit['rank']}. {hit['title']}**  \nScore: `{hit['score']:.3f}`")
                st.write(hit["text"])
                st.markdown(
                    f"Source: [{hit['url']}]({hit['url']})  \n"
                    f"Fetched: {hit['fetched_at_dublin']}  \n"
                    f"Passage id: `{hit['chunk_id']}`"
                )

        st.info(
            "If these passages do not match your situation, that is expected — "
            "the index is a small snapshot, not a complete statement of the law. "
            "Follow the source link and check the live page."
        )


if __name__ == "__main__":
    main()
