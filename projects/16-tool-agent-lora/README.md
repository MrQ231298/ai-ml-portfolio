# Tool-using LLM agent with a LoRA fine-tune (small open model, CPU)

A **plan-and-execute agent** that answers business questions by calling
tools: **read-only SQL** over a company database, a **safe calculator**,
and **BM25 search** over policy documents. The model writes a JSON plan,
the executor runs it, and on an error the model gets one chance to repair
the plan. The planner is **SmolLM2-360M-Instruct**, a small open model.
It was adapted with **LoRA** on CPU and compared with the same model used
zero-shot, few-shot, and without tools.

A portfolio piece for applied-LLM / ML-engineering roles, a client-facing
demo, and evidence toward a QQI Level 9 (Master's) RPL file.

**Data status:** **SYNTHETIC.** The database belongs to a fictional
distributor, "Northlight Components Ltd" (240 customers, 15 products,
1,009 orders, 2,028 order lines, January–October 2026), with 6 policy
documents. Everything is generated from a fixed seed
(`src/business_db.py`). No real customer or employer data is used. The
gold answer for every task comes from **executing** its gold plan, so the
answer key cannot drift from the data.

**What was and was not run:** the LoRA fine-tune (2 epochs, CPU) and the
evaluation of five planners **ran**. To fit the time budget, the
evaluation scored a **fixed subset: the first 4 tasks of each of the 10
task families in each test split (40 in-distribution + 40
out-of-distribution tasks)**, not the full 136 + 144. A full-set run was
started and stopped before finishing, so no full-set numbers are
reported. The optional OpenAI-compatible API planner (`AGENT_API_*`
variables) was **not** run. The Streamlit app was not screenshot-tested.

New to the code? Start with the [code walkthrough](CODE_WALKTHROUGH.md).
For a non-technical summary, read the [case study](CASE_STUDY.md).

---

## Problem

Staff ask the same kinds of question all day: "how many orders did Cork
place in March?", "what is 25 of these with VAT?", "how long is the
returns window?". A chat model on its own makes the numbers up. An agent
has to:

- pick the right tool and write correct SQL,
- chain steps (look up a price, then calculate),
- refuse to write to the database,
- keep working when the question is phrased in a new way or names a
  customer area or product it has not seen.

The question here is whether a **360M-parameter model that runs on a
laptop CPU** can do this reliably after a small LoRA fine-tune, and how it
compares with prompting alone.

---

## Approach

| Stage | Detail | Code |
|---|---|---|
| Tools | `sql(query)`: SELECT/WITH only, write keywords and multiple statements refused, 20-row cap. `calculator(expression)`: AST-evaluated arithmetic, no `eval`. `search_docs(query)`: BM25 over policy sections. | `src/tools.py` |
| Executor | Runs calls in order; `{0}`, `{1}` pass earlier results into later calls; returns the first error | `src/executor.py` |
| Tasks | 10 families (order counts, category revenue, top product, price lookup, segment counts, VAT, discount, revenue incl. VAT, order total, policy lookup), 6 phrasings each. **train** 796 tasks (phrasings 0–3); **test_id** 136 (phrasings 0–3, questions never seen in training); **test_ood** 144 (**phrasings 4–5 never trained on**, plus **held-out counties** Kildare and Wexford and **4 held-out products**) | `src/tasks.py` |
| Planners | `no_tools` (model answers directly); `base_zeroshot` (same system prompt as LoRA); `base_fewshot` (+4 worked examples); `lora` (base + adapter, zero-shot prompt); `gold_plan` (harness check, must score 100%) | `src/agent.py`, `src/evaluate.py` |
| Fine-tune | LoRA r = 16, α = 32, dropout 0.05 on q/k/v/o projections; loss on plan tokens only; lr 3e-4, 2 epochs, effective batch 8; 10% of training tasks held out for validation loss | `src/finetune_lora.py` |
| Scoring | **Task success** = executed final answer matches the gold (numbers within 0.1% or 0.01; policy answers must return the right document section). Also valid-JSON rate, exact tool-sequence accuracy and SQL execution accuracy. `no_tools` gets a **lenient** text check (any number in the reply within tolerance). 95% bootstrap CIs. | `src/agent.py`, `src/evaluate.py` |

---

## Results

All numbers are from `reports/metrics.json` (per-task rows in
`reports/eval_rows.csv`), run on 18 November 2026 on CPU, on **40 + 40
tasks** (4 per family per split; the same tasks for every planner).

### Fine-tune (`reports/finetune_log.json`)

| | |
|---|---|
| Trainable parameters | **3,276,800 of 365,097,920 (0.90%)** |
| Training / validation examples | 717 / 79 |
| Optimiser steps | 180 (2 epochs) |
| Wall time | 1,814.7 s on 8 CPU threads (the machine was shared with other jobs) |
| Validation loss | 0.663 before → 0.0180 after epoch 1 → **0.0097** after epoch 2 |
| Adapter size | 13 MB (`models/lora_adapter/`, committed) |

### Agent evaluation

| Planner | Task success, in-distribution (95% CI) | Task success, **out-of-distribution** (95% CI) | Valid JSON (ID / OOD) | Exact tool sequence (ID / OOD) | SQL execution accuracy (ID / OOD) |
|---|---|---|---|---|---|
| No tools (lenient text check) | 0.0% | 0.0% | – | – | – |
| Base, zero-shot | 0.0% | 0.0% | 10% / 0% | 0% / 0% | 0% / 0% |
| Base, few-shot (4 examples) | 15.0% (5–25) | 12.5% (2.5–22.5) | 100% / 100% | 17.5% / 17.5% | 7.1% / 14.3% |
| **Base + LoRA** | **97.5% (92.5–100)** | **85.0% (72.5–95)** | 100% / 100% | 100% / 92.5% | 100% / 89.3% |
| Gold plan (harness check) | 100% | 100% | 100% | 100% | 100% |

Mean time per task for LoRA was 2.3 s in-distribution and 10.4 s
out-of-distribution, on CPU.

What the rows show:

- **Without tools the model invents numbers** ("Sligo customers placed
  100 orders"), so it scored 0% even with lenient scoring.
- **Zero-shot, the base model rarely produced a parseable plan** (10% /
  0% valid JSON). Few-shot fixed the format (100% valid JSON) but not the
  content: it mostly copied the examples' tools and wrote the wrong SQL.
- **LoRA made the 360M model a usable planner**, and the gain held on
  phrasings and entities it never saw (85%).
- **The 6 out-of-distribution failures** (`eval_rows.csv`): two "Look up
  the unit price for …" questions (an unseen phrasing) were sent to
  `search_docs` instead of SQL. One held-out product name, "CAN bus cable
  5 m", was mangled into `'bus cable'` and `* 5`. One revenue question
  ("…in euro?") got an extra VAT multiplication. Two policy questions
  searched with the wrong keywords. In-distribution, the single failure
  was a policy search returning the wrong section. Self-repair was
  triggered once and did not change the outcome.

---

## How to run

```bash
cd projects/16-tool-agent-lora
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # downloads SmolLM2-360M-Instruct (~0.7 GB) on first use

cd src
python business_db.py && python tasks.py  # rebuild the SYNTHETIC DB and task sets (already committed)
python finetune_lora.py                   # ~30 min on 8 CPU threads; writes models/lora_adapter
python evaluate.py --per-family 4         # the recorded evaluation (~25 min on CPU)
python evaluate.py                        # full test sets (longer; not completed for this README)
cd ..
streamlit run app.py
pytest -q
```

**Tests:** 10 fast tests that need no model download: read-only SQL
enforcement, SQL errors returned as tool errors, calculator safety
(code rejected), document search, plan parsing, `{n}` references between
calls, **every gold plan executes and scores as a success**, no question
shared between splits and OOD using only held-out phrasings, the lenient
text scorer, and consistency of `metrics.json` with `eval_rows.csv` and
the fine-tune log.

---

## Client value

- **Private and cheap.** A model this size runs on an ordinary server or
  laptop with no API costs, and company data never leaves the building.
- **Answers are grounded.** Every number comes from an executed SQL query
  or calculation, and the plan is shown, so it can be checked.
- **Safe by construction.** The SQL tool is read-only and the calculator
  cannot run code, whatever the model writes.
- **Measured honestly.** Scores include questions phrased in new ways and
  about unseen areas and products, which is how real users behave.

---

## Critical evaluation

- **Small evaluation sample.** 40 + 40 tasks; the confidence intervals
  are wide (OOD 72.5–95%). The full test sets (136 + 144) were generated
  but not fully scored within the time budget.
- **Template-generated questions.** Even the held-out phrasings come from
  the same generator, so real users will be more varied. 85% on held-out
  templates is an upper bound for free-form questions.
- **Synthetic, simple schema.** Four tables and 10 question types. Real
  databases have many more tables, ambiguous names and messy data.
- **Single-turn questions only.** There is no conversation memory, no
  clarifying questions and no answer write-up.
- **No comparison with a large hosted model.** The API hook exists but
  was not run, so the gap to a frontier model is unmeasured.
- **Fine-tune on one seed**, with very low validation loss on templated
  data, which suggests the model learnt the templates closely. The OOD
  drop (97.5% → 85%) is the evidence of that.
- **CPU latency** of 2–10 s per question is acceptable for a demo, not
  for heavy use.

---

## Next steps

1. Score the full test sets and add a hand-written set of free-form
   questions from real staff phrasing.
2. Add paraphrase augmentation and unseen-schema tasks to training.
3. Compare with a hosted model through the existing API hook, on cost,
   latency and accuracy.
4. Add an answer-writing step with citations to the executed query.
5. Constrained JSON decoding to guarantee valid plans.
