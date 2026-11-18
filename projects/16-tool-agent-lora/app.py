"""Streamlit demo: ask the fine-tuned agent a business question and watch the plan run."""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from agent import ADAPTER_DIR, LocalLM, run_agent  # noqa: E402

st.set_page_config(page_title="Tool-using agent (LoRA)", layout="wide")
st.title("Tool-using business agent: small open model + LoRA")
st.caption("SYNTHETIC company data (Northlight Components, fictional). Runs on CPU; no API key. "
           "Do not enter personal data.")


@st.cache_resource
def load(adapter: bool):
    return LocalLM(adapter=ADAPTER_DIR if adapter else None)


col1, col2 = st.columns([2, 1])
with col2:
    use_lora = st.radio("Planner", ["LoRA fine-tuned", "Base model, few-shot"], index=0) == "LoRA fine-tuned"
    examples = ["How many orders did customers in Cork place in March?",
                "What would 25 units of the M12 circular connector cost including 23% VAT?",
                "What was the revenue from sensors in July?",
                "What is the late payment interest rate?",
                "Take 15% off EUR 940."]
    pick = st.selectbox("Examples", examples)
    m = json.loads((ROOT / "reports" / "metrics.json").read_text()) if (ROOT / "reports" / "metrics.json").exists() else None
    if m:
        st.subheader("Recorded test results")
        for p in ("no_tools", "base_fewshot", "lora"):
            s = m["planners"].get(p, {})
            if s:
                st.write(f"**{p}**: in-distribution {s['test_id']['task_success']:.0%}, "
                         f"held-out phrasings {s['test_ood']['task_success']:.0%}")
with col1:
    q = st.text_input("Question", pick)
    if st.button("Run agent"):
        with st.spinner("Loading model and planning (CPU)..."):
            out = run_agent(load(use_lora), q, few_shot=not use_lora)
        st.markdown("**Model output (plan)**")
        st.code(out["first"]["raw"], language="json")
        if out["repaired"]:
            st.warning(f"First plan failed ({out['first']['error']}); the model was asked to repair it.")
            st.code(out["repaired"]["raw"], language="json")
        st.markdown("**Tool calls**")
        for step in out["trace"]:
            st.write(f"`{step['tool']}` ← `{step['arg']}`")
            st.json(step["result"], expanded=False)
        if out["error"]:
            st.error(out["error"])
        else:
            st.success(f"Answer: {out['final']}")
        st.caption(f"{out['seconds']:.1f} s on CPU")
