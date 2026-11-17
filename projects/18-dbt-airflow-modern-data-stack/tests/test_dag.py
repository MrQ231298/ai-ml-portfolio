"""DAG structure. Run with the Airflow environment's Python (skipped if Airflow is absent)."""

from pathlib import Path

import pytest

pytest.importorskip("airflow.sdk")

DAGS = Path(__file__).resolve().parents[1] / "airflow" / "dags"


def test_dag_imports_and_has_the_expected_chain():
    from airflow.dag_processing.dagbag import DagBag
    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    dag = bag.get_dag("nyc_taxi_monthly")
    assert [t.task_id for t in dag.topological_sort()] == ["extract_load", "dbt_seed", "dbt_build", "quality_gate"]
    assert dag.catchup and dag.max_active_runs == 1
    assert dag.get_task("dbt_build").retries == 0  # data-test failures go to a human, not a retry
