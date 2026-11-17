# Code walkthrough — SQL + Python ETL pipeline

## The pipeline at a glance

```
1. python src/generate_sources.py
        ↓ data/raw/batch_*/ (messy extracts) + data/truth/ (answers)
2. python src/cli.py run-all
        ↓ extract.read_batch      → text DataFrames + file hashes
        ↓ clean.clean_*           → clean rows + rejected rows with reasons
        ↓ load.load_batch         → core.* upserts, quarantine, retries, batch log
        ↓ load.build_marts        → sql/10..40_*.sql
        ↓ validate.run_checks     → etl.dq_result; exit 1 on an error-level failure
3. python src/evaluate.py         → reports/metrics.json, dq_report.md, marts/*.csv
4. pytest -q
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/generate_sources.py`](src/generate_sources.py) | Builds clean truth, corrupts a copy, writes two batches and the answers. |
| [`src/etl/extract.py`](src/etl/extract.py) | Reads a batch as text; SHA-256 per file; synthetic-flag guard. |
| [`src/etl/clean.py`](src/etl/clean.py) | Parsing and standardisation rules; `(clean, rejected, stats)` per table. |
| [`src/etl/load.py`](src/etl/load.py) | DuckDB schema, upserts, referential splits, quarantine, retry, mart build. |
| [`src/etl/validate.py`](src/etl/validate.py) | 15 SQL data-quality checks with severities. |
| [`src/cli.py`](src/cli.py) | `run`, `run-all`, `status`, `export`, `reset`. |
| [`src/evaluate.py`](src/evaluate.py) | Reconciliation, detection scoring, SCD1 and idempotency experiments. |
| [`sql/00_schema.sql`](sql/00_schema.sql) | Core and control tables with primary keys. |
| [`sql/10_mart_sales_line.sql`](sql/10_mart_sales_line.sql) … `40_*.sql` | Reporting marts. |
| [`tests/`](tests/) | 14 unit tests of cleaning rules, 4 integration tests of the whole pipeline. |

## Key ideas in the code

**Text first, types second.** `read_batch` loads every column as a
string. Typing happens in one place (`clean.py`), so a bad value becomes
a reject reason rather than a crash or a silent `NaN`.

**Explicit date formats.** `parse_date` tries a fixed list of formats
and is never month-first, because `04/03/2026` is 4 March in Ireland.

**Rejected rows keep their payload.** The quarantine stores the cleaned
row as JSON. `_retry` re-reads those rows when a later batch delivers
the missing customer, order or product, inserts them, and stamps
`resolved_batch`.

**Idempotency comes from keys, not luck.** Every core table has a
primary key and is written with `INSERT OR REPLACE`; marts are rebuilt
from core with `CREATE OR REPLACE TABLE`. Re-running cannot double
count.

**Money in DECIMAL.** `fct_sales_line` multiplies `DECIMAL` columns and
rounds half-up to the cent; the generator's expected totals use Python's
`Decimal` the same way, which is why reconciliation is exact.
