"""
Evaluate retrieval and answering on the SYNTHETIC Brightwater document pack.

1. Retrieval (no tuning involved, so all 36 answerable questions are used):
   section-level hit@1/3/5 and MRR for BM25, TF-IDF, dense MiniLM,
   hybrid RRF and hybrid + cross-encoder re-ranking.
2. Answering: the abstention threshold on the best cross-encoder score is
   chosen on the *dev* questions only, then applied unchanged to the
   *test* questions. Reported: answer accuracy on answerable questions
   (all expected key facts present in the extractive answer), abstention
   precision/recall on unanswerable ones, and overall accuracy.

    python src/evaluate.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import ROOT, Index, chunk_documents, extractive_answer  # noqa: E402

REPORTS, MODELS = ROOT / "reports", ROOT / "models"


def main() -> None:
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    qs = json.loads((ROOT / "eval" / "questions.json").read_text())["questions"]
    chunks = chunk_documents()
    t0 = time.time()
    idx = Index(chunks)
    build_s = time.time() - t0
    print(f"chunks={len(chunks)} docs={len({c.doc for c in chunks})} index_build_s={build_s:.1f}")

    def gold_rank(order, q) -> int | None:
        for r, i in enumerate(order[:20], start=1):
            c = chunks[i]
            if c.doc == q["gold_doc"] and c.section == q["gold_section"]:
                return r
        return None

    answerable = [q for q in qs if q["gold_doc"]]
    retr_rows, retr = [], {}
    for name in ("bm25", "tfidf", "dense", "hybrid", "hybrid_rerank"):
        ranks, lat = [], []
        for q in answerable:
            t = time.time()
            order = idx.rank_rerank(q["question"])[0] if name == "hybrid_rerank" else getattr(idx, f"rank_{name}")(q["question"])
            lat.append(time.time() - t)
            r = gold_rank(order, q)
            ranks.append(r)
            retr_rows.append({"retriever": name, "id": q["id"], "split": q["split"], "gold_rank": r,
                              "top1": f"{chunks[order[0]].doc} § {chunks[order[0]].section}"})
        rr = np.array([1 / r if r else 0 for r in ranks])
        retr[name] = {
            "hit@1": round(float(np.mean([r == 1 for r in ranks])), 4),
            "hit@3": round(float(np.mean([bool(r) and r <= 3 for r in ranks])), 4),
            "hit@5": round(float(np.mean([bool(r) and r <= 5 for r in ranks])), 4),
            "mrr@20": round(float(rr.mean()), 4),
            "median_latency_ms": round(float(np.median(lat) * 1000), 1),
        }
        print(name, retr[name])
    pd.DataFrame(retr_rows).to_csv(REPORTS / "retrieval_ranks.csv", index=False)

    # ---- answering: tune abstention threshold on dev, report on test ----
    runs = {}
    for q in qs:
        out = extractive_answer(idx, q["question"], threshold=-1e9)   # never abstain; threshold applied below
        ans = out["answer"] or ""
        runs[q["id"]] = {"best": out["best_score"], "answer": ans, "citations": out["citations"],
                         "correct_if_answered": bool(q["gold_doc"]) and all(k.lower() in ans.lower() for k in q["answer_keywords"])}

    def evaluate(split: str, thr: float) -> dict:
        sub = [q for q in qs if q["split"] == split]
        ok, ans_ok, n_ans, abst_tp, abst_fp, n_unans = 0, 0, 0, 0, 0, 0
        for q in sub:
            r = runs[q["id"]]
            abstain = r["best"] < thr
            if q["gold_doc"]:
                n_ans += 1
                good = (not abstain) and r["correct_if_answered"]
                ans_ok += good; ok += good; abst_fp += abstain
            else:
                n_unans += 1
                abst_tp += abstain; ok += abstain
        n_abst = abst_tp + abst_fp
        return {"n": len(sub), "answerable": n_ans, "unanswerable": n_unans,
                "answer_accuracy_on_answerable": round(ans_ok / n_ans, 4),
                "abstention_recall_on_unanswerable": round(abst_tp / n_unans, 4),
                "abstention_precision": round(abst_tp / n_abst, 4) if n_abst else None,
                "overall_accuracy": round(ok / len(sub), 4)}

    # Candidate thresholds: midpoints between consecutive *dev* scores (never
    # test scores). Ties on dev accuracy are broken towards the LOWEST
    # threshold, i.e. answer more often: every extractive answer shows its
    # source passage, so a weak answer can be checked by the user, whereas a
    # needless "not found" gives them nothing.
    dev_s = sorted(runs[q["id"]]["best"] for q in qs if q["split"] == "dev")
    grid = [-1e9] + [round((a + b) / 2, 3) for a, b in zip(dev_s, dev_s[1:])]
    best_acc = max(evaluate("dev", t)["overall_accuracy"] for t in grid)
    best_thr = min(t for t in grid if evaluate("dev", t)["overall_accuracy"] == best_acc)
    print("chosen threshold", best_thr)
    res = {"dev": evaluate("dev", best_thr), "test": evaluate("test", best_thr),
           "test_without_abstention": evaluate("test", -1e9)}
    for k, v in res.items():
        print(k, v)
    (MODELS / "abstain_threshold.json").write_text(json.dumps({"threshold": best_thr, "chosen_on": "dev questions"}, indent=2))

    rows = []
    for q in qs:
        r = runs[q["id"]]
        rows.append({"id": q["id"], "split": q["split"], "question": q["question"], "answerable": bool(q["gold_doc"]),
                     "best_rerank_score": round(r["best"], 3), "abstained": r["best"] < best_thr,
                     "answer": "" if r["best"] < best_thr else r["answer"], "citations": "; ".join(r["citations"]),
                     "keywords_found": r["correct_if_answered"]})
    pd.DataFrame(rows).to_csv(REPORTS / "answer_eval.csv", index=False)

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(retr)
    for j, m in enumerate(["hit@1", "hit@3", "mrr@20"]):
        ax.bar(np.arange(len(names)) + j * 0.27, [retr[n][m] for n in names], 0.27, label=m)
    ax.set_xticks(np.arange(len(names)) + 0.27); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.legend(); ax.set_title(f"Section-level retrieval on {len(answerable)} answerable questions")
    fig.tight_layout(); fig.savefig(REPORTS / "retrieval_metrics.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for lab, col, cond in (("answerable", "#2b6cb0", True), ("not in documents", "#dd6b20", False)):
        ax.scatter([r["best_rerank_score"] for r in rows if r["answerable"] == cond],
                   [r["split"] for r in rows if r["answerable"] == cond], label=lab, color=col, alpha=0.7)
    ax.axvline(best_thr, color="k", ls="--", label=f"abstain below {best_thr:.2f} (chosen on dev)")
    ax.set_xlabel("best cross-encoder score"); ax.legend(fontsize=8); ax.set_title("Abstention threshold")
    fig.tight_layout(); fig.savefig(REPORTS / "abstention_threshold.png", dpi=130); plt.close(fig)

    metrics = {
        "run_at_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "corpus": {"documents": len({c.doc for c in chunks}), "chunks": len(chunks), "status": "SYNTHETIC_EXAMPLE_DATA",
                   "formats": sorted({p.suffix for p in (ROOT / "data" / "client_docs").iterdir()})},
        "questions": {"total": len(qs), "answerable": len(answerable), "unanswerable": len(qs) - len(answerable),
                      "dev": sum(q["split"] == "dev" for q in qs), "test": sum(q["split"] == "test" for q in qs)},
        "models": {"embedder": "sentence-transformers/all-MiniLM-L6-v2", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                   "generator": "extractive (no LLM); optional OpenAI-compatible hook not evaluated"},
        "index_build_seconds_cpu": round(build_s, 1),
        "retrieval_all_answerable": retr,
        "abstention_threshold": best_thr,
        "answering": res,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", REPORTS / "metrics.json")


if __name__ == "__main__":
    main()
