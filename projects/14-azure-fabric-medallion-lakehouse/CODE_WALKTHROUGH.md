# Code walkthrough — local medallion lakehouse

## The pipeline at a glance

```
cd src
1. python fetch_and_land.py      UCI zip → SHA-256 check → landing/online_retail_ii/YYYY-MM.csv
2. python run_pipeline.py        bronze.ingest → silver.build → gold.build → dq.run
3. python evaluate.py            full vs incremental, replay, correction, time travel → reports/
4. pytest -q                     (from the project folder) fixture-based tests
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`conf/lakehouse.json`](conf/lakehouse.json) | Source, checksum, local paths and their OneLake / ADLS equivalents, non-product codes, DQ limits. |
| [`src/common.py`](src/common.py) | Paths, SHA-256, Delta read / write helpers (delta-rs), DuckDB connection. |
| [`src/fetch_and_land.py`](src/fetch_and_land.py) | Download, verify, split into monthly files as text. |
| [`src/bronze.py`](src/bronze.py) | Hash-based incremental ingest; predicate overwrite for corrected files; control table. |
| [`src/silver.py`](src/silver.py) | Typing, de-duplication, `line_type` classification, month-partition rebuilds. |
| [`src/gold.py`](src/gold.py) | Star schema and aggregates (DuckDB SQL → Delta). |
| [`src/dq.py`](src/dq.py) | Ten expectations across the three layers, written to a DQ Delta table. |
| [`src/run_pipeline.py`](src/run_pipeline.py) | Orchestrator / CLI. |
| [`src/evaluate.py`](src/evaluate.py) | The equivalence, idempotency and correction experiments; business summary. |
| [`infra/main.bicep`](infra/main.bicep) | Azure landing infrastructure (compiled to `main.json`, not deployed). |
| [`adf/pl_medallion_online_retail.json`](adf/pl_medallion_online_retail.json) | ADF / Fabric pipeline definition (not deployed). |
| [`fabric/notebooks/`](fabric/notebooks/) | PySpark ports of bronze, silver, gold and DQ (not executed). |
| [`docs/AZURE_FABRIC_MAPPING.md`](docs/AZURE_FABRIC_MAPPING.md) | Component-by-component mapping and deployment steps. |

## Key ideas in the code

**Bronze never interprets.** Every column is read as text; the only
additions are lineage columns. If silver logic changes, silver can be
rebuilt from bronze without asking the source again.

**Hashes drive incrementality.** A file is ingested if its SHA-256 is
new. A known name with a new hash is a correction: its month partition
is overwritten with a predicate, in bronze and then in silver.

**Deterministic rules everywhere.** Duplicates keep the first row by
file position; product descriptions use "most frequent, then
alphabetical"; customers use "latest country, then alphabetical". Without
explicit tie-breaks, incremental and full runs disagreed.

**Checksums prove equivalence.** `evaluate.checksum` hashes every
business column of every silver and gold table in a fixed order, so
"identical" means identical.
