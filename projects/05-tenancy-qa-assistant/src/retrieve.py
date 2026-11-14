"""
Shared retrieval helpers for the tenancy Q&A assistant.

Two backends live in the same index bundle:

- sentence-transformers ``all-MiniLM-L6-v2`` (preferred, if it built)
- a TF-IDF cosine baseline (always built, used for comparison and as a
  fallback)

Neither backend generates new text. They only rank existing passages
and return them with their source URL and fetch date.

Usage (from the project root, after building the index)::

    from src.retrieve import load_index, search
    index = load_index()
    hits = search(index, "How large can a security deposit be?", k=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "models" / "tenancy_index.joblib"

# Loaded lazily so importing this module is cheap (the Streamlit app and
# the evaluation script both import it).
_st_model = None


def project_paths() -> dict[str, Path]:
    return {
        "root": ROOT,
        "pages": ROOT / "data" / "pages",
        "chunks": ROOT / "data" / "chunks.json",
        "index": INDEX_PATH,
        "reports": ROOT / "reports",
        "eval": ROOT / "eval" / "questions.json",
    }


def load_index(path: Path | None = None) -> dict[str, Any]:
    import joblib

    path = path or INDEX_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No index at {path}. Run `python src/chunk_and_index.py` first."
        )
    return joblib.load(path)


def _minilm_model(model_name: str):
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(model_name)
    return _st_model


def _search_minilm(index: dict[str, Any], query: str, k: int) -> list[dict[str, Any]]:
    model = _minilm_model(index["minilm_model_name"])
    q = model.encode([query], normalize_embeddings=True)
    scores = (index["minilm_embeddings"] @ q[0]).astype(float)
    return _top_hits(index, scores, k)


def _search_tfidf(index: dict[str, Any], query: str, k: int) -> list[dict[str, Any]]:
    from sklearn.metrics.pairwise import cosine_similarity

    q_vec = index["tfidf_vectorizer"].transform([query])
    scores = cosine_similarity(q_vec, index["tfidf_matrix"]).ravel()
    return _top_hits(index, scores, k)


def _top_hits(index: dict[str, Any], scores: np.ndarray, k: int) -> list[dict[str, Any]]:
    k = min(k, len(scores))
    # argpartition then sort the shortlist so we do not fully sort a long list
    top_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    hits = []
    for rank, i in enumerate(top_idx, start=1):
        chunk = dict(index["chunks"][int(i)])
        chunk["score"] = float(scores[int(i)])
        chunk["rank"] = rank
        hits.append(chunk)
    return hits


def search(
    index: dict[str, Any],
    query: str,
    k: int = 5,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """Return the top-k passages for ``query``.

    ``backend`` is ``"minilm"`` or ``"tfidf"``. If omitted, the bundle's
    preferred backend is used (MiniLM when it was built successfully).
    """
    query = (query or "").strip()
    if not query:
        return []
    backend = (backend or index.get("preferred_backend") or "tfidf").lower()
    if backend == "minilm" and index.get("minilm_embeddings") is not None:
        return _search_minilm(index, query, k)
    if backend == "minilm":
        # Caller asked for MiniLM but it is not in the bundle.
        backend = "tfidf"
    return _search_tfidf(index, query, k)


def available_backends(index: dict[str, Any]) -> list[str]:
    out = []
    if index.get("minilm_embeddings") is not None:
        out.append("minilm")
    if index.get("tfidf_vectorizer") is not None:
        out.append("tfidf")
    return out
