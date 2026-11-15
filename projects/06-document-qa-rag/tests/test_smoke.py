"""Smoke tests (run `pytest -q` from the project folder). Lexical parts run
without models; the full-pipeline test needs the cached Hugging Face models."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag import BM25, Index, chunk_documents, extractive_answer, split_sections  # noqa: E402


def test_ingest_all_formats():
    chunks = chunk_documents()
    assert {c.doc for c in chunks} >= {"expenses_policy", "service_level_agreement", "health_and_safety_procedures"}
    assert any(c.section == "Mileage" for c in chunks)  # PDF heading detection worked


def test_markdown_sections():
    secs = split_sections("# T\n## A\nbody one.\n## B\nbody two.", is_pdf=False)
    assert [s[0] for s in secs] == ["A", "B"]


def test_bm25_ranks_exact_match_first():
    b = BM25(["the car park closes at eight", "annual leave is 23 days", "passwords need 14 characters"])
    assert b.scores("annual leave days").argmax() == 1


def test_lexical_index_without_models():
    idx = Index(chunk_documents(), load_models=False)
    top = idx.chunks[idx.rank_bm25("mileage rate per kilometre")[0]]
    assert top.section == "Mileage"


def test_full_pipeline_answers_and_cites():
    idx = Index(chunk_documents())
    out = extractive_answer(idx, "How many days annual leave do I get?", threshold=-100)
    assert "23 days" in out["answer"] and out["citations"]


def test_metrics_recorded():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert m["retrieval_all_answerable"]["hybrid_rerank"]["mrr@20"] > m["retrieval_all_answerable"]["bm25"]["mrr@20"]
