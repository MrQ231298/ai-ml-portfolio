"""
Evaluate retrieval quality on the hand-built question set.

Metrics (page-level, as the gold labels are source pages):

- hit@k: fraction of questions for which at least one expected page
  appears in the top-k passages, for k in {1, 3, 5}
- MRR: mean reciprocal rank of the first passage whose page is expected

Both the preferred MiniLM backend (if present) and the TF-IDF baseline
are scored. No numbers are invented — only this run is reported.

Usage
-----
    python src/evaluate.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieve import available_backends, load_index, project_paths, search

DUBLIN_TZ = ZoneInfo("Europe/Dublin")
KS = (1, 3, 5)


def first_hit_rank(hits: list[dict], expected: set[str]) -> int | None:
    for hit in hits:
        if hit["page_id"] in expected:
            return int(hit["rank"])
    return None


def summarise(ranks: list[int | None], n: int) -> dict:
    hits = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    hits["mrr"] = mrr
    hits["n_questions"] = n
    hits["n_answered"] = sum(r is not None for r in ranks)
    return hits


def evaluate_backend(index, questions, backend: str, k_max: int = 5) -> tuple[dict, list[dict]]:
    rows = []
    ranks = []
    for item in questions:
        expected = set(item["expected_page_ids"])
        hits = search(index, item["question"], k=k_max, backend=backend)
        rank = first_hit_rank(hits, expected)
        ranks.append(rank)
        top = hits[0] if hits else None
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "backend": backend,
                "first_hit_rank": rank,
                "hit@1": int(rank is not None and rank <= 1),
                "hit@3": int(rank is not None and rank <= 3),
                "hit@5": int(rank is not None and rank <= 5),
                "top_page_id": top["page_id"] if top else None,
                "top_title": top["title"] if top else None,
                "top_score": top["score"] if top else None,
                "top_url": top["url"] if top else None,
                "retrieved_page_ids": " | ".join(dict.fromkeys(h["page_id"] for h in hits)),
            }
        )
    return summarise(ranks, len(questions)), rows


def plot_metrics(metrics_by_backend: dict[str, dict], dest: Path) -> None:
    backends = list(metrics_by_backend)
    labels = ["hit@1", "hit@3", "hit@5", "MRR"]
    keys = ["hit@1", "hit@3", "hit@5", "mrr"]
    x = range(len(labels))
    width = 0.35 if len(backends) > 1 else 0.5
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, backend in enumerate(backends):
        vals = [metrics_by_backend[backend][k] for k in keys]
        offset = (i - (len(backends) - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], vals, width, label=backend)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Tenancy Q&A retrieval — page-level hit@k and MRR")
    ax.legend()
    ax.axhline(1.0, color="#888", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main() -> None:
    paths = project_paths()
    paths["reports"].mkdir(parents=True, exist_ok=True)
    questions = json.loads(paths["eval"].read_text(encoding="utf-8"))["questions"]
    index = load_index()
    backends = available_backends(index)
    # Preferred backend first in the report.
    preferred = index.get("preferred_backend")
    if preferred in backends:
        backends = [preferred] + [b for b in backends if b != preferred]

    metrics_by_backend = {}
    all_rows = []
    for backend in backends:
        summary, rows = evaluate_backend(index, questions, backend)
        metrics_by_backend[backend] = summary
        all_rows.extend(rows)
        print(f"\n{backend}:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    run_at = datetime.now(DUBLIN_TZ).strftime("%Y-%m-%d %H:%M %Z")
    payload = {
        "run_at_dublin": run_at,
        "n_questions": len(questions),
        "n_pages": index["n_pages"],
        "n_chunks": index["n_chunks"],
        "preferred_backend": index.get("preferred_backend"),
        "minilm_model_name": index.get("minilm_model_name"),
        "minilm_error": index.get("minilm_error"),
        "metrics": metrics_by_backend,
        "ks": list(KS),
        "note": (
            "Page-level retrieval metrics on a hand-built set of questions. "
            "A hit means at least one gold source page appeared in the top-k "
            "passages. This is not a measure of legal correctness."
        ),
    }
    metrics_path = paths["reports"] / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    df = pd.DataFrame(all_rows)
    csv_path = paths["reports"] / "eval_results.csv"
    df.to_csv(csv_path, index=False)
    plot_path = paths["reports"] / "retrieval_metrics.png"
    plot_metrics(metrics_by_backend, plot_path)
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
