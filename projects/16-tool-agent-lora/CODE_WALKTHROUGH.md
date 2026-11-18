# Code walkthrough

Read in this order. Paths are relative to this project folder.

1. **`src/business_db.py`**: builds the SYNTHETIC SQLite database
   (`data/SYNTHETIC_northlight.db`: customers, products, orders, order
   lines) and six markdown policy documents in `data/docs/`, from seed 16.
   `SCHEMA_TEXT` is the schema description shown to the model.

2. **`src/tools.py`**
   - `sql_query()`: only SELECT/WITH, a keyword block-list for writes, one
     statement, 20 rows. Errors are raised as `ToolError`, so the agent
     sees them.
   - `calculator()`: parses with `ast` and allows only numbers,
     `+ - * / ** %` and round/abs/min/max/sqrt; no `eval`.
   - `search_docs()`: BM25 over document sections and returns the best
     section.
   - `TOOLS` maps tool names to functions.

3. **`src/executor.py`**: `run_plan()` runs each call. `{n}` in an argument
   is replaced by the scalar result of call *n*. It returns outputs, the
   final answer and the first error.

4. **`src/tasks.py`**: 10 task families, each with 6 phrasings, slot
   samplers and a gold-plan builder. Train uses phrasings 0–3 and
   training entities. `test_id` uses unseen question strings, `test_ood`
   uses phrasings 4–5 plus held-out counties and products. Gold answers
   come from executing gold plans. It writes `data/tasks_*.jsonl`.

5. **`src/agent.py`**
   - `SYSTEM_PROMPT` (tools, schema, JSON format) and `FEW_SHOT`.
   - `parse_plan()`: the first valid `{"calls": [...]}` object.
   - `LocalLM`: greedy decoding on CPU, optionally with the LoRA adapter
     merged. `ApiLM` is the unused OpenAI-compatible hook.
   - `run_agent()`: plan, execute, one repair attempt on error.
   - `answer_without_tools()`: the closed-book baseline.
   - `task_success()` and the lenient `text_success()` scorers.

6. **`src/finetune_lora.py`**: tokenises chat prompt + gold plan, masks
   the prompt tokens (loss on the plan only), wraps the model with PEFT
   LoRA and trains with AdamW and warm-up. It logs validation loss per
   epoch to `reports/finetune_log.json` and saves `models/lora_adapter/`.

7. **`src/evaluate.py`**: runs each planner on the test splits
   (`--per-family N` takes the first N tasks per family, as in the
   recorded run). It scores task success, JSON validity, tool sequence
   and SQL execution accuracy with bootstrap CIs, and writes
   `reports/eval_rows.csv` and `reports/metrics.json`.

8. **`app.py`**: Streamlit demo. Pick LoRA or few-shot, ask a question,
   see the plan, each tool result and the answer.

9. **`tests/`**: tool safety, executor, gold-plan integrity, split
   hygiene, scoring, and report consistency. None of them need the model.
