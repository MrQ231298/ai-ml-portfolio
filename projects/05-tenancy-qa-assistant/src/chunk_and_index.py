"""
Chunk the fetched guidance pages and build a local retrieval index.

Preferred embeddings: sentence-transformers ``all-MiniLM-L6-v2`` (runs
locally, no paid API key). A TF-IDF cosine index is always built as a
baseline and as a fallback if the transformer model cannot be loaded.

Usage
-----
    python src/chunk_and_index.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))

from retrieve import project_paths

DUBLIN_TZ = ZoneInfo("Europe/Dublin")
MINILM_NAME = "all-MiniLM-L6-v2"

# Character windows. These are deliberately modest: a passage should be
# short enough to read in the demo and long enough to keep a heading
# plus a couple of paragraphs together.
MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 180
OVERLAP_CHARS = 150

# Boilerplate we do not want in the index.
DROP_FROM_LINE = re.compile(
    r"^(You are here:|### Follow Us|### Filter by tag|### Years|### Archives)\b"
)


def load_pages(pages_dir: Path) -> list[dict]:
    pages = []
    for path in sorted(pages_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("text"):
            pages.append(rec)
    if not pages:
        raise RuntimeError(f"No page JSON files in {pages_dir}. Run src/fetch_pages.py first.")
    return pages


def clean_text(text: str) -> str:
    """Strip breadcrumbs and obvious site-chrome tails."""
    lines = []
    for line in text.splitlines():
        if DROP_FROM_LINE.search(line.strip()):
            # Cut the Citizens Information news-page chrome (Follow Us / archives).
            if line.strip().startswith("### Follow Us"):
                break
            continue
        lines.append(line)
    # Collapse leftover blank lines.
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text


def _split_long(section: str, page: dict, start_idx: int) -> list[dict]:
    """Sliding windows over a section that is longer than MAX_CHUNK_CHARS."""
    heading = ""
    body = section
    m = re.match(r"^(#{1,4} .+)\n", section)
    if m:
        heading = m.group(1).strip()
        body = section[m.end():].strip()

    chunks = []
    start = 0
    local = 0
    while start < len(body):
        end = min(len(body), start + MAX_CHUNK_CHARS)
        # Prefer to break on a paragraph or sentence boundary.
        if end < len(body):
            break_at = body.rfind("\n", start + MIN_CHUNK_CHARS, end)
            if break_at == -1:
                break_at = body.rfind(". ", start + MIN_CHUNK_CHARS, end)
                if break_at != -1:
                    break_at += 1
            if break_at != -1 and break_at > start:
                end = break_at
        piece = body[start:end].strip()
        if heading:
            text = heading + "\n" + piece
        else:
            text = piece
        if len(text) >= 80:
            chunks.append(_make_chunk(page, start_idx + local, text))
            local += 1
        if end >= len(body):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return chunks


def _make_chunk(page: dict, idx: int, text: str) -> dict:
    return {
        "chunk_id": f"{page['page_id']}__{idx:03d}",
        "page_id": page["page_id"],
        "url": page["url"],
        "title": page["title"],
        "fetched_at_dublin": page["fetched_at_dublin"],
        "n_chars": len(text),
        "text": text,
    }


def chunk_page(page: dict) -> list[dict]:
    text = clean_text(page["text"])
    sections = re.split(r"\n(?=#{1,4} )", text)
    sections = [s.strip() for s in sections if s.strip()]
    chunks: list[dict] = []
    buf = ""
    for sec in sections:
        # Skip a lone table-of-contents style heading block that is very short
        # and only lists other headings.
        if buf and len(buf) + 2 + len(sec) <= MAX_CHUNK_CHARS:
            buf = buf + "\n\n" + sec
            continue
        if buf:
            if len(buf) > MAX_CHUNK_CHARS:
                chunks.extend(_split_long(buf, page, len(chunks)))
            elif len(buf) >= MIN_CHUNK_CHARS or not chunks:
                chunks.append(_make_chunk(page, len(chunks), buf))
            else:
                # Very short leftover: append to the previous chunk.
                chunks[-1]["text"] = chunks[-1]["text"] + "\n\n" + buf
                chunks[-1]["n_chars"] = len(chunks[-1]["text"])
            buf = ""
        if len(sec) > MAX_CHUNK_CHARS:
            chunks.extend(_split_long(sec, page, len(chunks)))
        else:
            buf = sec
    if buf:
        if len(buf) > MAX_CHUNK_CHARS:
            chunks.extend(_split_long(buf, page, len(chunks)))
        elif len(buf) >= MIN_CHUNK_CHARS or not chunks:
            chunks.append(_make_chunk(page, len(chunks), buf))
        else:
            chunks[-1]["text"] = chunks[-1]["text"] + "\n\n" + buf
            chunks[-1]["n_chars"] = len(chunks[-1]["text"])
    return chunks


def build_tfidf(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Unigrams + bigrams keep phrases such as "notice of termination"
    # and "90 days" intact. Digits are kept on purpose.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def try_minilm(texts: list[str]) -> tuple[np.ndarray | None, str | None]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"sentence-transformers import failed ({exc}). Falling back to TF-IDF.")
        return None, f"import_failed: {type(exc).__name__}: {exc}"
    try:
        print(f"Loading {MINILM_NAME} (downloaded locally on first run)...")
        model = SentenceTransformer(MINILM_NAME)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        return np.asarray(embeddings, dtype=np.float32), None
    except Exception as exc:
        print(f"MiniLM encode failed ({exc}). Falling back to TF-IDF.")
        return None, f"encode_failed: {type(exc).__name__}: {exc}"


def main() -> None:
    paths = project_paths()
    paths["index"].parent.mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    pages = load_pages(paths["pages"])
    chunks: list[dict] = []
    for page in pages:
        page_chunks = chunk_page(page)
        print(f"  {page['page_id']}: {len(page_chunks)} chunks "
              f"({page['n_chars']} chars on page)")
        chunks.extend(page_chunks)
    if not chunks:
        raise RuntimeError("Chunking produced no passages.")

    texts = [c["text"] for c in chunks]
    print(f"Total chunks: {len(chunks)} from {len(pages)} pages")

    vectorizer, matrix = build_tfidf(texts)
    embeddings, minilm_error = try_minilm(texts)
    preferred = "minilm" if embeddings is not None else "tfidf"
    built_at = datetime.now(DUBLIN_TZ).strftime("%Y-%m-%d %H:%M %Z")

    bundle = {
        "built_at_dublin": built_at,
        "preferred_backend": preferred,
        "minilm_model_name": MINILM_NAME if embeddings is not None else None,
        "minilm_embeddings": embeddings,
        "minilm_error": minilm_error,
        "tfidf_vectorizer": vectorizer,
        "tfidf_matrix": matrix,
        "chunks": chunks,
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "chunking": {
            "max_chars": MAX_CHUNK_CHARS,
            "min_chars": MIN_CHUNK_CHARS,
            "overlap_chars": OVERLAP_CHARS,
        },
    }
    joblib.dump(bundle, paths["index"], compress=3)
    # Also write a plain-JSON copy of the chunks so the index is inspectable
    # without unpickling.
    slim = [{k: c[k] for k in c} for c in chunks]
    paths["chunks"].write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Preferred backend: {preferred}")
    print(f"Wrote {paths['index']} and {paths['chunks']}")


if __name__ == "__main__":
    main()
