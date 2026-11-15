"""
Retrieval-augmented question answering over a folder of client documents.

Pipeline
--------
1. Ingest  : read .md / .txt / .pdf files, detect section headings, split
             each section into overlapping word-window chunks.
2. Retrieve: four interchangeable retrievers over the same chunks
               - BM25 (lexical, implemented here, no extra dependency)
               - TF-IDF cosine (lexical baseline)
               - dense: sentence-transformers all-MiniLM-L6-v2 embeddings
               - hybrid: reciprocal-rank fusion (RRF) of BM25 + dense
3. Re-rank : optional cross-encoder (ms-marco-MiniLM-L-6-v2) over the
             hybrid top-20.
4. Answer  : by default an *extractive* answer - the best 1-2 sentences
             from the top passages, with citations - and an abstention
             ("not found in the documents") when the best cross-encoder
             score is below a threshold tuned on the dev questions.
             If RAG_LLM_BASE_URL and RAG_LLM_MODEL are set, the top
             passages are instead sent to any OpenAI-compatible chat
             endpoint (for example a local Ollama server) with an
             instruction to answer only from the context and cite.

Everything runs locally without a paid API key. The two models are
downloaded once from Hugging Face into the local cache.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "client_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_WORDS, OVERLAP = 90, 30
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to and or in on for is are be by with at as it this that from your you we our "
           "i do does can what how when which who if any my me".split())


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str


# ----------------------------------------------------------------- ingest
def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def split_sections(text: str, is_pdf: bool) -> list[tuple[str, str]]:
    """Return (heading, body) pairs. Markdown uses '#'; PDFs use a heuristic:
    a short line without a final full stop that is followed by body text."""
    lines = [l.rstrip() for l in text.splitlines()
             if l.strip() and not l.startswith("<!--") and "SYNTHETIC EXAMPLE DOCUMENT" not in l]
    sections, head, buf = [], "Introduction", []

    def is_heading(i: int, l: str) -> bool:
        if l.startswith("#"):
            return True
        return is_pdf and len(l) < 60 and not l.endswith((".", ",", ";", ":")) and i + 1 < len(lines)

    for i, l in enumerate(lines):
        if is_heading(i, l):
            if buf:
                sections.append((head, " ".join(buf)))
            head, buf = l.lstrip("# ").strip(), []
        else:
            buf.append(l.strip())
    if buf:
        sections.append((head, " ".join(buf)))
    return sections


def chunk_documents(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        for s_idx, (head, body) in enumerate(split_sections(read_document(path), path.suffix == ".pdf")):
            words = body.split()
            step = CHUNK_WORDS - OVERLAP
            for k, start in enumerate(range(0, max(len(words) - OVERLAP, 1), step)):
                piece = " ".join(words[start:start + CHUNK_WORDS])
                chunks.append(Chunk(f"{path.stem}:{s_idx}:{k}", path.stem, head, f"{head}. {piece}"))
    return chunks


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


# -------------------------------------------------------------- retrievers
class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = np.mean([len(t) for t in self.toks])
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.toks))
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            dl = len(toks)
            for w in q:
                if w in tf:
                    f = tf[w]
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


class Index:
    """Holds the chunks and every retriever built over them."""

    def __init__(self, chunks: list[Chunk], load_models: bool = True):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.bm25 = BM25(texts)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.tfidf_m = self.tfidf.fit_transform(texts)
        self.embedder = self.reranker = None
        if load_models:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            self.embedder = SentenceTransformer(EMBED_MODEL)
            self.emb = self.embedder.encode(texts, normalize_embeddings=True, batch_size=32)
            self.reranker = CrossEncoder(RERANK_MODEL)

    # each retriever returns chunk indices in rank order
    def rank_bm25(self, q: str) -> np.ndarray:
        return np.argsort(-self.bm25.scores(q), kind="stable")

    def rank_tfidf(self, q: str) -> np.ndarray:
        s = (self.tfidf_m @ self.tfidf.transform([q]).T).toarray().ravel()
        return np.argsort(-s, kind="stable")

    def rank_dense(self, q: str) -> np.ndarray:
        qv = self.embedder.encode([q], normalize_embeddings=True)[0]
        return np.argsort(-(self.emb @ qv), kind="stable")

    def rank_hybrid(self, q: str, k: int = 60) -> np.ndarray:
        """Reciprocal-rank fusion (Cormack et al., 2009) of BM25 and dense."""
        fused = np.zeros(len(self.chunks))
        for ranking in (self.rank_bm25(q), self.rank_dense(q)):
            fused[ranking] += 1.0 / (k + np.arange(1, len(ranking) + 1))
        return np.argsort(-fused, kind="stable")

    def rank_rerank(self, q: str, pool: int = 20) -> tuple[np.ndarray, np.ndarray]:
        cand = self.rank_hybrid(q)[:pool]
        s = self.reranker.predict([(q, self.chunks[i].text) for i in cand])
        order = np.argsort(-s, kind="stable")
        return cand[order], np.asarray(s)[order]


# ------------------------------------------------------------------ answer
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if len(s.split()) >= 4]


def extractive_answer(index: Index, q: str, threshold: float, top_passages: int = 3) -> dict:
    ranked, scores = index.rank_rerank(q)
    best = float(scores[0])
    passages = [index.chunks[i] for i in ranked[:top_passages]]
    if best < threshold:
        return {"answer": None, "abstained": True, "best_score": best, "citations": [],
                "passages": [asdict(p) for p in passages]}
    sents = []
    for p in passages[:2]:
        body = p.text.split(". ", 1)[1] if ". " in p.text else p.text   # drop the heading prefix
        sents += [(s, p) for s in split_sentences(body)]
    s_scores = np.asarray(index.reranker.predict([(q, s) for s, _ in sents]))
    order = list(np.argsort(-s_scores))
    best_i = order[0]
    chosen_idx = [best_i]
    # Add a second sentence only if it comes from the same passage as the
    # best one (keeps the answer coherent instead of mixing two topics).
    for i in order[1:]:
        if sents[i][1].chunk_id == sents[best_i][1].chunk_id:
            chosen_idx.append(i)
            break
    chosen = [sents[i] for i in sorted(chosen_idx)]
    answer = " ".join(s for s, _ in chosen)
    cites = sorted({f"{p.doc} § {p.section}" for _, p in chosen})
    return {"answer": answer, "abstained": False, "best_score": best, "citations": cites,
            "passages": [asdict(p) for p in passages]}


def llm_answer(index: Index, q: str, threshold: float) -> dict:
    """Optional generative answer via an OpenAI-compatible endpoint.

    Set RAG_LLM_BASE_URL (e.g. http://localhost:11434/v1 for Ollama),
    RAG_LLM_MODEL (e.g. llama3.1) and, if needed, RAG_LLM_API_KEY.
    Not used in the recorded evaluation (no key or local LLM was used).
    """
    import requests

    base = extractive_answer(index, q, threshold)
    if base["abstained"]:
        return base
    ctx = "\n\n".join(f"[{i + 1}] ({p['doc']} § {p['section']}) {p['text']}" for i, p in enumerate(base["passages"]))
    msgs = [
        {"role": "system", "content": "Answer ONLY from the numbered context. Cite sources like [1]. "
                                      "If the context does not contain the answer, say you cannot find it."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
    ]
    r = requests.post(
        os.environ["RAG_LLM_BASE_URL"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {os.environ.get('RAG_LLM_API_KEY', 'none')}"},
        json={"model": os.environ["RAG_LLM_MODEL"], "messages": msgs, "temperature": 0},
        timeout=120,
    )
    r.raise_for_status()
    base["answer"] = r.json()["choices"][0]["message"]["content"]
    base["generator"] = os.environ["RAG_LLM_MODEL"]
    return base


def answer(index: Index, q: str, threshold: float) -> dict:
    if os.environ.get("RAG_LLM_BASE_URL") and os.environ.get("RAG_LLM_MODEL"):
        return llm_answer(index, q, threshold)
    return extractive_answer(index, q, threshold)


def load_threshold() -> float:
    p = ROOT / "models" / "abstain_threshold.json"
    return json.loads(p.read_text())["threshold"] if p.exists() else 0.0
