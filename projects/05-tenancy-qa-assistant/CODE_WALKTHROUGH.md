# Code walkthrough — tenancy Q&A assistant

This note is for someone who can run a command and open a file, but
is still new to retrieval. It explains the files, the run order, and
the few lines that matter.

The metrics table and the critical evaluation live in
[README.md](README.md). The short client write-up is in
[CASE_STUDY.md](CASE_STUDY.md).

**Data reminder:** the index is built from a snapshot of *public*
Citizens Information and RTB pages. The text remains the publishers'
copyright. This is not legal advice.

---

## The idea in one minute

We save a modest set of official guidance pages, split them into
passages, and embed those passages with a small local model
(`all-MiniLM-L6-v2`). A question is embedded the same way. The
nearest passages are shown *as they were written*, with a link and
the fetch date. A TF-IDF baseline is built for comparison. Nothing
is sent to a paid API.

---

## Run order

```text
1. python src/fetch_pages.py
        ↓ writes data/pages/*.json, data/raw/*.html, data/manifest.json

2. python src/chunk_and_index.py
        ↓ writes data/chunks.json
        ↓ writes models/tenancy_index.joblib
        ↓ downloads all-MiniLM-L6-v2 on the first run

3. python src/evaluate.py
        ↓ writes reports/metrics.json, eval_results.csv, retrieval_metrics.png

4. streamlit run app.py
```

If you only want to try the committed snapshot, you can skip step 1.

---

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/fetch_pages.py`](src/fetch_pages.py) | Hand-picked fetch of public CI and RTB pages; records URL and fetch time. |
| [`data/pages/`](data/pages/) | Extracted text plus provenance for each page. |
| [`data/manifest.json`](data/manifest.json) | Snapshot inventory. |
| [`src/chunk_and_index.py`](src/chunk_and_index.py) | Chunks pages; builds MiniLM and TF-IDF indexes. |
| [`src/retrieve.py`](src/retrieve.py) | Shared search helper used by the evaluator and the app. |
| [`src/evaluate.py`](src/evaluate.py) | hit@k and MRR on the 16-question set. |
| [`eval/questions.json`](eval/questions.json) | Questions and expected source pages. |
| [`models/tenancy_index.joblib`](models/tenancy_index.joblib) | Passages, embeddings and the TF-IDF vectorizer. |
| [`app.py`](app.py) | Streamlit demo, with a prominent disclaimer. |

---

## How to run it locally

From this folder (`projects/05-tenancy-qa-assistant`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_pages.py          # optional if data/ is already present
python src/chunk_and_index.py
python src/evaluate.py
streamlit run app.py
```

The first index build downloads the MiniLM model into your local
Hugging Face cache (about 90 MB). No API key is required.

---

## Key code, with plain-English notes

### 1. Every saved page keeps its source and fetch time

```python
rec = {
    "page_id": slug,
    "url": url,
    "title": title or slug,
    "fetched_at_dublin": fetched_at.strftime("%Y-%m-%d %H:%M %Z"),
    "http_status": status,
    "n_chars": len(text),
    "text": text,
}
```

That record is what the app cites. If the live site changes later,
the snapshot still tells you *which* wording was indexed and *when*.

### 2. Retrieval is a cosine search, not a generated answer

```python
q = model.encode([query], normalize_embeddings=True)
scores = index["minilm_embeddings"] @ q[0]
# ... return the top-k passages, unchanged, with url and fetch date
```

Normalised embeddings mean the dot product *is* cosine similarity.
The passage text is not rewritten, so a citation can be checked
against the official page.

### 3. A hit is decided at page level

```python
def first_hit_rank(hits, expected):
    for hit in hits:
        if hit["page_id"] in expected:
            return int(hit["rank"])
    return None
```

`expected` is the set of pages that contain the answer in this
snapshot. MRR is the mean of `1/rank` for that first hit (or zero
if it never appears). That is why the README treats a perfect
hit@1 as optimistic: it does not score whether the *sentence* is
complete or legally applied.

---

## What this walkthrough is not

It is not a tutorial on Irish tenancy law, and it is not a licence
to republish the stored pages. Follow the source links, and read the
critical evaluation in the README before quoting the retrieval
scores.
