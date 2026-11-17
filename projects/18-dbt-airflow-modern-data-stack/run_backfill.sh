#!/usr/bin/env bash
# Run the Airflow DAG for January and February 2024 with `airflow dags test`
# (executes every task in-process against a local SQLite metadata DB).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export AIRFLOW_HOME="$HERE/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$HERE/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export DBT_BIN="${DBT_BIN:-$HERE/.venv-dbt/bin/dbt}"
export ELT_PYTHON="${ELT_PYTHON:-$HERE/.venv-dbt/bin/python}"
AIRFLOW="${AIRFLOW_BIN:-$HERE/.venv-airflow/bin/airflow}"
"$AIRFLOW" db migrate
"$AIRFLOW" dags reserialize
for d in 2024-01-01 2024-02-01; do
  "$AIRFLOW" dags test nyc_taxi_monthly "$d" > "$HERE/reports/airflow_dags_test_${d:0:7}.log" 2>&1
  grep -o "DagRun Finished.*state=[a-z]*" "$HERE/reports/airflow_dags_test_${d:0:7}.log" | sed 's/run_id.*state=/state=/'
done
