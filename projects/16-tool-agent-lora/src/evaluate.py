"""Evaluation harness: run every planner on both test sets and write reports/.

Planners (all on CPU, greedy decoding):
  no_tools       base model answers directly (closed-book baseline, lenient text scoring)
  base_zeroshot  base model, same system prompt as the fine-tuned model
  base_fewshot   base model + 4 worked examples in the prompt
  lora           base model + LoRA adapter, zero-shot prompt
  gold_plan      the gold plan itself (sanity check of the harness; must be 100%)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from agent import ADAPTER_DIR, LocalLM, answer_without_tools, run_agent, task_success, text_success
from executor import run_plan
from tools import ToolError, sql_query

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(split):
    return [json.loads(l) for l in open(ROOT / "data" / f"tasks_{split}.jsonl", encoding="utf-8")]


def sql_exec_match(task, calls):
    """Execution accuracy of the first SQL call (results equal as multisets), or None if gold has no SQL."""
    gold_sql = [c for c in task["plan"] if c["tool"] == "sql"]
    if not gold_sql:
        return None
    pred_sql = [c for c in (calls or []) if c.get("tool") == "sql" and isinstance((c.get("args") or {}).get("query"), str)]
    if not pred_sql:
        return False
    try:
        return sorted(map(str, sql_query(pred_sql[0]["args"]["query"]))) == sorted(map(str, sql_query(gold_sql[0]["args"]["query"])))
    except ToolError:
        return False


def score_plan_row(task, out, planner, split):
    calls = out["first"]["calls"]
    gold_tools = [c["tool"] for c in task["plan"]]
    pred_tools = [c.get("tool") for c in calls] if calls else None
    return {"planner": planner, "split": split, "id": task["id"], "family": task["family"],
            "question": task["question"], "valid_json": calls is not None,
            "tool_seq_correct": pred_tools == gold_tools,
            "first_tool_correct": bool(pred_tools) and pred_tools[0] == gold_tools[0],
            "sql_exec_match": sql_exec_match(task, calls),
            "success_first_try": task_success(task, out["first"]["final"]),
            "repair_used": out["repaired"] is not None,
            "success": task_success(task, out["final"]),
            "seconds": round(out["seconds"], 3), "raw": out["first"]["raw"][:400],
            "error": out["error"]}


def bootstrap_ci(values, n=2000, seed=0):
    rng = random.Random(seed)
    v = list(values)
    means = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def summarise(df):
    out = {}
    for (planner, split), g in df.groupby(["planner", "split"]):
        sq = g["sql_exec_match"].dropna()
        out.setdefault(planner, {})[split] = {
            "n": int(len(g)),
            "task_success": round(g["success"].mean(), 4),
            "task_success_ci95": bootstrap_ci(g["success"].astype(float)),
            "task_success_first_try": round(g["success_first_try"].mean(), 4),
            "valid_json_rate": round(g["valid_json"].mean(), 4) if planner != "no_tools" else None,
            "tool_sequence_accuracy": round(g["tool_seq_correct"].mean(), 4) if planner != "no_tools" else None,
            "first_tool_accuracy": round(g["first_tool_correct"].mean(), 4) if planner != "no_tools" else None,
            "sql_execution_accuracy": round(sq.astype(float).mean(), 4) if planner != "no_tools" and len(sq) else None,
            "repair_rate": round(g["repair_used"].mean(), 4),
            "mean_seconds_per_task": round(g["seconds"].mean(), 3),
            "by_family_success": {k: round(v, 3) for k, v in g.groupby("family")["success"].mean().items()},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="tasks per split (0 = all)")
    ap.add_argument("--per-family", type=int, default=0, help="first N tasks per family per split (0 = all)")
    ap.add_argument("--suffix", default=None, help="output file suffix (default: '' or '_smoke')")
    ap.add_argument("--planners", default="gold_plan,no_tools,base_zeroshot,base_fewshot,lora")
    a = ap.parse_args()
    splits = {s: load(s) for s in ("test_id", "test_ood")}
    if a.limit:
        splits = {k: v[:a.limit] for k, v in splits.items()}
    if a.per_family:
        def take(tasks):
            seen, out = defaultdict(int), []
            for t in tasks:
                if seen[t["family"]] < a.per_family:
                    seen[t["family"]] += 1
                    out.append(t)
            return out
        splits = {k: take(v) for k, v in splits.items()}
    planners = a.planners.split(",")
    rows = []
    base = LocalLM() if any(p in planners for p in ("no_tools", "base_zeroshot", "base_fewshot")) else None
    timings = {}
    for planner in planners:
        t0 = time.time()
        if planner == "lora":
            lm = LocalLM(adapter=ADAPTER_DIR)
        else:
            lm = base
        for split, tasks in splits.items():
            for t in tasks:
                if planner == "gold_plan":
                    res = run_plan(t["plan"])
                    out = {"first": {"calls": t["plan"], "final": res["final"], "raw": ""}, "repaired": None,
                           "final": res["final"], "error": res["error"], "seconds": 0.0}
                    rows.append(score_plan_row(t, out, planner, split))
                elif planner == "no_tools":
                    o = answer_without_tools(lm, t["question"])
                    ok = text_success(t, o["raw"])
                    rows.append({"planner": planner, "split": split, "id": t["id"], "family": t["family"],
                                 "question": t["question"], "valid_json": False, "tool_seq_correct": False,
                                 "first_tool_correct": False, "sql_exec_match": None, "success_first_try": ok,
                                 "repair_used": False, "success": ok, "seconds": round(o["seconds"], 3),
                                 "raw": o["raw"][:400], "error": None})
                else:
                    out = run_agent(lm, t["question"], few_shot=(planner == "base_fewshot"))
                    rows.append(score_plan_row(t, out, planner, split))
            print(planner, split, "done", round(time.time() - t0), "s", flush=True)
        timings[planner] = round(time.time() - t0, 1)
        REPORTS.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(REPORTS / f"eval_rows_partial{a.suffix or ''}.csv", index=False)  # survive interruptions
    df = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    suffix = a.suffix if a.suffix is not None else ("" if not a.limit else "_smoke")
    df.to_csv(REPORTS / f"eval_rows{suffix}.csv", index=False)
    summary = summarise(df)
    ft = json.loads((REPORTS / "finetune_log.json").read_text()) if (REPORTS / "finetune_log.json").exists() else None
    metrics = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": "cpu (8 threads)",
               "sampling": f"first {a.per_family} tasks per family per split" if a.per_family else "all tasks",
               "test_sizes": {k: len(v) for k, v in splits.items()}, "planners": summary,
               "planner_wall_seconds": timings, "finetune": ft and {k: v for k, v in ft.items()}}
    (REPORTS / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=2, default=str))
    for p, s in summary.items():
        print(p, {k: (v["task_success"], v["tool_sequence_accuracy"]) for k, v in s.items()})


if __name__ == "__main__":
    main()
