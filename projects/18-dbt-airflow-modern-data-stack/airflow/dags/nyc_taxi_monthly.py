"""Monthly ELT for NYC yellow-taxi trips: extract + load -> dbt seed -> dbt build -> quality gate.

Each run handles the month of its logical date. All steps are idempotent, so
retries, re-runs and backfills (catchup) are safe. dbt lives in its own virtual
environment (DBT_BIN), separate from Airflow's, to avoid dependency clashes.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag, task

PROJECT = Path(os.environ.get("NYC_TAXI_PROJECT", Path(__file__).resolve().parents[2]))
DBT_DIR = PROJECT / "nyc_taxi"
DBT_BIN = os.environ.get("DBT_BIN", "dbt")
PYTHON_BIN = os.environ.get("ELT_PYTHON", sys.executable)
MONTH = "{{ logical_date.strftime('%Y-%m') }}"


@dag(
    dag_id="nyc_taxi_monthly",
    schedule="@monthly",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 2, 1),
    catchup=True,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["dbt", "duckdb", "portfolio"],
    doc_md=__doc__,
)
def nyc_taxi_monthly():
    extract_load = BashOperator(
        task_id="extract_load",
        bash_command=f"{PYTHON_BIN} {PROJECT}/ingest/load_month.py --month {MONTH}",
        execution_timeout=timedelta(minutes=30),
    )
    dbt_seed = BashOperator(
        task_id="dbt_seed",
        bash_command=f"cd {DBT_DIR} && {DBT_BIN} seed --profiles-dir . --project-dir .",
    )
    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command=(f"cd {DBT_DIR} && {DBT_BIN} build --profiles-dir . --project-dir . "
                      f"--vars '{{\"month\": \"{MONTH}\"}}' --exclude resource_type:seed"
                      f" && cp target/run_results.json {PROJECT}/reports/airflow_runs/run_results_{MONTH}.json"),
        retries=0,  # a failing data test should stop the run for a human, not be retried
    )

    @task
    def quality_gate(month: str) -> dict:
        res = json.loads((PROJECT / "reports" / "airflow_runs" / f"run_results_{month}.json").read_text())
        statuses = {}
        for r in res["results"]:
            kind = r["unique_id"].split(".")[0]
            statuses.setdefault(kind, {}).setdefault(r["status"], 0)
            statuses[kind][r["status"]] += 1
        bad = [r["unique_id"] for r in res["results"] if r["status"] in ("fail", "error")]
        summary = {"month": month, "statuses": statuses, "failed": bad,
                   "dbt_elapsed_s": round(res["elapsed_time"], 2)}
        (PROJECT / "reports" / "airflow_runs" / f"gate_{month}.json").write_text(json.dumps(summary, indent=2))
        if bad:
            raise ValueError(f"quality gate failed: {bad}")
        return summary

    (PROJECT / "reports" / "airflow_runs").mkdir(parents=True, exist_ok=True)
    extract_load >> dbt_seed >> dbt_build >> quality_gate(MONTH)


nyc_taxi_monthly()
