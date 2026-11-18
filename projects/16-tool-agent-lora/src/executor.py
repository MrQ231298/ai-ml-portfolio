"""Execute a tool plan. Calls may reference earlier results as {0}, {1}, ..."""

from __future__ import annotations

import re

from tools import TOOLS, ToolError


def scalar(result):
    """First cell of a SQL result, the number from the calculator, or the passage dict."""
    if isinstance(result, list):
        if not result or not result[0]:
            raise ToolError("query returned no rows")
        return result[0][0]
    return result


def _resolve(text: str, outputs: list):
    def rep(m):
        i = int(m.group(1))
        if i >= len(outputs):
            raise ToolError(f"reference {{{i}}} has no earlier result")
        v = scalar(outputs[i])
        if v is None:
            raise ToolError(f"result {{{i}}} is empty")
        return str(v)
    return re.sub(r"\{(\d+)\}", rep, text)


def run_plan(plan: list[dict]) -> dict:
    """Run each call in order. Returns outputs, final answer and the first error (if any)."""
    outputs, trace = [], []
    for call in plan:
        name = call.get("tool")
        if name not in TOOLS:
            return {"outputs": outputs, "trace": trace, "final": None, "error": f"unknown tool {name!r}"}
        arg_name, fn = TOOLS[name]
        arg = (call.get("args") or {}).get(arg_name)
        if not isinstance(arg, str):
            return {"outputs": outputs, "trace": trace, "final": None, "error": f"{name} needs a '{arg_name}' string"}
        try:
            res = fn(_resolve(arg, outputs))
        except ToolError as e:
            return {"outputs": outputs, "trace": trace, "final": None, "error": f"{name}: {e}"}
        outputs.append(res)
        trace.append({"tool": name, "arg": arg, "result": res})
    if not outputs:
        return {"outputs": [], "trace": [], "final": None, "error": "empty plan"}
    last = outputs[-1]
    if isinstance(last, list):  # SQL: keep the first row (one value, or name + value)
        final = (list(last[0]) if len(last[0]) > 1 else last[0][0]) if last and last[0] else None
    else:
        final = last
    return {"outputs": outputs, "trace": trace, "final": final, "error": None}
