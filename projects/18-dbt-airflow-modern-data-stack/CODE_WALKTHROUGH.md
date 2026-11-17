# Code walkthrough

Read in this order. Paths are relative to this project folder.

1. **`ingest/load_month.py`**
   - `fetch(month)` downloads the TLC Parquet file into `downloads/`.
     The first time, it records the file's SHA-256 in
     `data_manifest.json`. After that, it refuses to load if the checksum
     has changed.
   - `load(month, path)` creates `raw.yellow_trips` if needed. In **one
     transaction** it deletes that month's rows and inserts the file with
     three metadata columns. That is why reruns are idempotent, and
     `rows_replaced` shows it.

2. **`nyc_taxi/` (the dbt project)**
   - `dbt_project.yml` sets the layers' schemas and materialisations, and
     declares `vars.month`. `profiles.yml` points dbt at the DuckDB file,
     which can be overridden with `DBT_DUCKDB_PATH`.
   - `models/staging/` renames and types the raw columns.
     `stg_yellow_trips` builds `trip_key` as an md5 of identifying fields.
     When `month` is set, it reads only that month.
   - `models/intermediate/int_trips_flagged.sql` holds the seven quality
     rules in one `CASE`, in priority order, with `row_number()` over
     `trip_key` for duplicates. `int_trips_enriched.sql` keeps the valid
     rows and joins zone names and payment descriptions.
   - `models/marts/fct_trips.sql` is
     `materialized='incremental', incremental_strategy='delete+insert',
     unique_key='source_month'`. A monthly run replaces that month only,
     and `--full-refresh` rebuilds everything. `dq_rejections_by_month`
     works the same way.
   - `tests/generic/*.sql` hold the three custom generic tests.
     `tests/assert_*.sql` are the reconciliation tests (rows and money).
   - The `.yml` files hold the descriptions (they feed `dbt docs`), the
     tests and the exposure.

3. **`airflow/dags/nyc_taxi_monthly.py`**: an Airflow 3 DAG.
   - `MONTH` is a Jinja template on `logical_date`, so a backfill passes
     each month to the loader and to `dbt build --vars`.
   - `dbt_build` has `retries=0`, because a failed data test needs a
     person, not a retry. The loader has two retries.
   - `quality_gate` reads dbt's `run_results.json`, counts the statuses,
     writes `reports/airflow_runs/gate_<month>.json`, and raises an error
     if anything failed.

4. **`scripts/verify.py`**
   - `fingerprint()` is a row count plus an MD5 over the sorted key and
     amount for three tables.
   - It re-runs January through Airflow and compares the fingerprints. It
     rebuilds a copy with `--full-refresh` and compares again. Then it
     plants two bad rows in another copy and records which dbt tests
     failed.

5. **`tests/`**
   - `test_pipeline.py` builds a tiny warehouse from 8 hand-made trips,
     one per rule, plus two valid trips. It checks that the loader is
     idempotent and that each rule rejects exactly its row.
   - `test_dag.py` loads the DAG with Airflow's `DagBag`. It checks the
     task order, catchup and the no-retry setting on `dbt_build`.
