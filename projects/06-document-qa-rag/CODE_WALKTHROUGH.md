# Code walkthrough — document Q&A (RAG)

## The pipeline at a glance

```
1. python src/generate_client_docs.py
        ↓ writes data/client_docs/*.md|.txt|.pdf  (SYNTHETIC)

2. python src/evaluate.py
        ↓ chunks the docs, builds 4 retrievers + re-ranker
        ↓ scores retrieval and answering on eval/questions.json
        ↓ writes reports/metrics.json, *.csv, *.png and models/abstain_threshold.json

3. pytest -q
4. streamlit run app.py
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/generate_client_docs.py`](src/generate_client_docs.py) | The fictional company's eight documents; renders two as PDF with reportlab. |
| [`src/rag.py`](src/rag.py) | Everything reusable: ingestion, chunking, BM25, TF-IDF, dense, hybrid RRF, cross-encoder re-ranking, extractive answer, optional LLM call. |
| [`src/evaluate.py`](src/evaluate.py) | The experiment: retrieval metrics, dev-tuned abstention threshold, test results, plots. |
| [`eval/questions.json`](eval/questions.json) | 44 questions with gold document/section, key facts and dev/test split. |
| [`app.py`](app.py) | Streamlit UI with file upload and an adjustable abstention slider. |
| [`tests/test_smoke.py`](tests/test_smoke.py) | Ingestion, section parsing, BM25, lexical index, end-to-end answer, metrics present. |

## Key ideas in the code

**Headings travel with the text.** Every chunk starts with its section
heading (`"Mileage. Use of a private car…"`). Short sections become one
chunk; this helps all retrievers and lets the answer cite the section.

**PDF headings are guessed.** PDF text has no `#` markers, so
`split_sections` treats a short line (<60 characters) without a final
full stop as a heading. Simple, and it worked for both PDFs here; a
client's PDFs may need a layout-aware parser.

**BM25 from scratch.** `class BM25` is the standard Okapi formula
(k1 = 1.5, b = 0.75) with a small stop-word list — about 20 lines, so the
project needs no extra search library.

**Reciprocal-rank fusion.** `rank_hybrid` adds `1 / (60 + rank)` from
each retriever. It needs no score normalisation, which is why it is
popular — but see the README for why it did not help on its own here.

**Re-rank, then answer.** The cross-encoder reads the question and each
of the top 20 passages *together*, which is slower (~100 ms per query on
CPU) but much more accurate than comparing separate embeddings. The same
model then picks the best sentence inside the top passages.

**Abstain on low confidence.** If the best re-rank score is below the
threshold saved in `models/abstain_threshold.json`, the answer is "not
found". The threshold comes only from dev questions.

**LLM hook.** `llm_answer` posts the top passages to
`$RAG_LLM_BASE_URL/chat/completions`, so it works with Ollama, vLLM, LM
Studio, Azure OpenAI or OpenAI without code changes.
