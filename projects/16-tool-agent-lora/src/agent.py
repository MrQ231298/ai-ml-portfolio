"""Plan-and-execute tool agent.

The language model reads the question and returns a JSON plan of tool calls.
The executor runs the plan; if a call fails, the error is shown to the model
once and it may return a corrected plan (self-repair).

Planners
- local Hugging Face model (base or base + LoRA adapter), few-shot or zero-shot
- "no_tools": the same base model answering directly (closed-book baseline)
- optional OpenAI-compatible API model: set AGENT_API_BASE, AGENT_API_MODEL
  (and AGENT_API_KEY if the endpoint needs one). Not used in the recorded run.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from business_db import SCHEMA_TEXT
from executor import run_plan

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = ROOT / "models" / "lora_adapter"

SYSTEM_PROMPT = f"""You plan tool calls for Northlight Components, an electronics parts distributor.
Tools:
- sql(query): read-only SQLite. Tables:
{SCHEMA_TEXT}
  All dates are in 2026; filter a month with strftime('%m', o.order_date) = 'MM'. Revenue = SUM(quantity * unit_price) over delivered orders.
- calculator(expression): arithmetic, e.g. 120 * 1.23
- search_docs(query): search company policy documents
Reply with JSON only: {{"calls": [{{"tool": "...", "args": {{...}}}}]}}. Use {{0}} for the result of the first call."""

FEW_SHOT = [
    ("How many trade customers are in Mayo?",
     {"calls": [{"tool": "sql", "args": {"query": "SELECT COUNT(*) FROM customers WHERE segment = 'trade' AND county = 'Mayo'"}}]}),
    ("What is EUR 80 including 23% VAT?", {"calls": [{"tool": "calculator", "args": {"expression": "80 * 1.23"}}]}),
    ("Do you deliver on Saturdays?", {"calls": [{"tool": "search_docs", "args": {"query": "delivery days"}}]}),
    ("What would 4 units of the Hall effect sensor cost including 23% VAT?",
     {"calls": [{"tool": "sql", "args": {"query": "SELECT unit_price FROM products WHERE name = 'Hall effect sensor'"}},
                {"tool": "calculator", "args": {"expression": "{0} * 4 * 1.23"}}]}),
]


def plan_messages(question: str, few_shot: bool) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for q, p in FEW_SHOT:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(p)}]
    msgs.append({"role": "user", "content": question})
    return msgs


def target_text(plan: list[dict]) -> str:
    return json.dumps({"calls": plan})


def parse_plan(text: str):
    """Return the list of calls from the first balanced JSON object, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    calls = obj.get("calls") if isinstance(obj, dict) else None
                    if isinstance(calls, list) and all(isinstance(c, dict) for c in calls):
                        return calls
                    break
        start = text.find("{", start + 1)
    return None


class LocalLM:
    """Greedy generation with a small causal LM on CPU, optionally with a LoRA adapter."""

    def __init__(self, model_name: str = BASE_MODEL, adapter: Path | None = None, threads: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).merge_and_unload()
        self.model.eval()

    def chat(self, messages: list[dict], max_new_tokens: int = 200) -> tuple[str, int]:
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        new = out[0][enc["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), int(enc["input_ids"].shape[1])


class ApiLM:
    """Optional OpenAI-compatible chat endpoint (not used in the recorded run)."""

    def __init__(self):
        self.base = os.environ["AGENT_API_BASE"].rstrip("/")
        self.model = os.environ["AGENT_API_MODEL"]
        self.key = os.environ.get("AGENT_API_KEY", "")

    def chat(self, messages, max_new_tokens=200):
        import urllib.request
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "max_tokens": max_new_tokens}).encode()
        req = urllib.request.Request(f"{self.base}/chat/completions", body,
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"], 0


def run_agent(lm, question: str, few_shot: bool = False, repair: bool = True) -> dict:
    t0 = time.perf_counter()
    msgs = plan_messages(question, few_shot)
    text, _ = lm.chat(msgs)
    calls = parse_plan(text)
    res = run_plan(calls) if calls is not None else {"final": None, "error": "invalid JSON plan", "trace": []}
    first = {"raw": text, "calls": calls, "final": res["final"], "error": res["error"]}
    repaired = None
    if repair and res["error"]:
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"That plan failed: {res['error']}. Reply with a corrected JSON plan."}]
        text2, _ = lm.chat(msgs)
        calls2 = parse_plan(text2)
        res2 = run_plan(calls2) if calls2 is not None else {"final": None, "error": "invalid JSON plan", "trace": []}
        repaired = {"raw": text2, "calls": calls2, "final": res2["final"], "error": res2["error"]}
        res = res2
    return {"first": first, "repaired": repaired, "final": res["final"], "trace": res.get("trace", []),
            "error": res["error"], "seconds": time.perf_counter() - t0}


def answer_without_tools(lm, question: str) -> dict:
    t0 = time.perf_counter()
    text, _ = lm.chat([{"role": "system", "content": "You are a helpful assistant for Northlight Components, "
                        "an Irish electronics parts distributor. Answer briefly."},
                       {"role": "user", "content": question}], max_new_tokens=80)
    return {"raw": text, "seconds": time.perf_counter() - t0}


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def close(a, b) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(0.01, 1e-3 * abs(b))


def task_success(task: dict, final) -> bool:
    """Compare an executed plan's final answer with the gold answer."""
    gold = task["gold"]
    if task["family"] == "doc_lookup":
        return isinstance(final, dict) and final.get("doc") == gold["doc"] and gold["key"].lower() in final.get("text", "").lower()
    if isinstance(gold, list):
        return isinstance(final, list) and len(final) >= 1 and str(final[0]) == str(gold[0])
    return close(final, gold)


def text_success(task: dict, text: str) -> bool:
    """Lenient score for free-text answers: any number in the text within tolerance, or the key phrase / name present."""
    gold = task["gold"]
    t = (text or "").lower()
    if task["family"] == "doc_lookup":
        return gold["key"].lower() in t
    if isinstance(gold, list):
        return str(gold[0]).lower() in t
    return any(close(n, gold) for n in numbers_in(text))
