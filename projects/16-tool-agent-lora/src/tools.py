"""The three business tools the agent can call: read-only SQL, a safe calculator
and BM25 search over the policy documents."""

from __future__ import annotations

import ast
import math
import operator
import re
import sqlite3
from collections import Counter
from pathlib import Path

from business_db import DB_PATH, DOCS_DIR

MAX_ROWS = 20


class ToolError(Exception):
    """Raised when a tool call is invalid; the message is shown to the agent."""


# ---------------------------------------------------------------- SQL
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


def sql_query(query: str, db_path: Path = DB_PATH) -> list[tuple]:
    """Run one read-only SELECT. Writes and multiple statements are refused."""
    q = (query or "").strip().rstrip(";")
    if not q.lower().startswith(("select", "with")):
        raise ToolError("only SELECT queries are allowed")
    if ";" in q or _FORBIDDEN.search(q):
        raise ToolError("query contains a forbidden statement")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(q).fetchmany(MAX_ROWS)
    except sqlite3.Error as e:
        raise ToolError(f"SQL error: {e}") from None
    finally:
        con.close()
    return [tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in rows]


# ---------------------------------------------------------------- calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos, ast.Mod: operator.mod}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt}


def calculator(expression: str) -> float:
    """Evaluate arithmetic without eval(): numbers, + - * / ** %, round/abs/min/max/sqrt."""
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", (expression or "").replace("€", "")).strip()
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ToolError(f"cannot parse expression: {expression!r}") from None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 10:
                raise ToolError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ToolError(f"unsupported expression element: {ast.dump(n)[:40]}")

    try:
        return round(float(ev(tree)), 4)
    except ZeroDivisionError:
        raise ToolError("division by zero") from None


# ---------------------------------------------------------------- document search
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or is are be by with what how do does can i we our my per "
            "which who when where much many long there it at as from".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class DocIndex:
    """BM25 (k1 = 1.5, b = 0.75) over the '##' sections of the Markdown documents."""

    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.passages = []  # (doc_id, heading, text)
        for p in sorted(docs_dir.glob("*.md")):
            heading, buf = None, []
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("## "):
                    if heading:
                        self.passages.append((p.stem, heading, " ".join(buf).strip()))
                    heading, buf = line[3:].strip(), []
                elif heading and line.strip():
                    buf.append(line.strip())
            if heading:
                self.passages.append((p.stem, heading, " ".join(buf).strip()))
        self.docs = [_tokens(f"{h} {h} {t}") for _, h, t in self.passages]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 1) -> list[dict]:
        q = _tokens(query or "")
        if not q:
            raise ToolError("empty search query")
        scores = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            s = sum(self.idf.get(t, 0) * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * len(d) / self.avgdl))
                    for t in q if t in tf)
            scores.append((s, i))
        scores.sort(reverse=True)
        out = []
        for s, i in scores[:k]:
            doc, head, text = self.passages[i]
            out.append({"doc": doc, "section": head, "text": text, "score": round(s, 3)})
        return out


_INDEX: DocIndex | None = None


def search_docs(query: str) -> dict:
    global _INDEX
    if _INDEX is None:
        _INDEX = DocIndex()
    return _INDEX.search(query, k=1)[0]


TOOLS = {"sql": ("query", sql_query), "calculator": ("expression", calculator), "search_docs": ("query", search_docs)}
