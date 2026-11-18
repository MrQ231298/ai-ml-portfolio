"""Plan parsing, execution with references, and scoring rules."""

import json
from pathlib import Path

from agent import parse_plan, task_success, text_success
from executor import run_plan

DATA = Path(__file__).resolve().parents[1] / "data"


def test_parse_plan_takes_first_valid_object_and_rejects_junk():
    text = 'Sure! {"calls": [{"tool": "calculator", "args": {"expression": "2 * 3"}}]} trailing'
    assert parse_plan(text) == [{"tool": "calculator", "args": {"expression": "2 * 3"}}]
    assert parse_plan("{0}") is None
    assert parse_plan('{"answer": 3}') is None


def test_references_pass_results_between_calls():
    plan = [{"tool": "sql", "args": {"query": "SELECT unit_price FROM products WHERE sku = 'CN-100'"}},
            {"tool": "calculator", "args": {"expression": "{0} * 10 * 1.23"}}]
    assert run_plan(plan)["final"] == round(2.40 * 10 * 1.23, 4)
    assert run_plan([{"tool": "calculator", "args": {"expression": "{1} + 1"}}])["error"]


def test_every_gold_plan_executes_and_scores_as_success():
    for split in ("train", "test_id", "test_ood"):
        for line in open(DATA / f"tasks_{split}.jsonl", encoding="utf-8"):
            t = json.loads(line)
            res = run_plan(t["plan"])
            assert res["error"] is None, t["id"]
            assert task_success(t, res["final"]), t["id"]


def test_splits_do_not_share_questions_and_ood_uses_held_out_phrasings():
    q = {s: [json.loads(l) for l in open(DATA / f"tasks_{s}.jsonl")] for s in ("train", "test_id", "test_ood")}
    train_q = {t["question"] for t in q["train"]}
    assert not train_q & {t["question"] for t in q["test_id"] + q["test_ood"]}
    assert all(t["phrasing"] >= 4 for t in q["test_ood"])


def test_lenient_text_scoring():
    t = {"family": "vat_calc", "gold": 123.0}
    assert text_success(t, "It comes to €123.00 including VAT") and not text_success(t, "about 150")
