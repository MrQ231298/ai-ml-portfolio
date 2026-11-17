"""Tests for the KPI definitions: run `pytest -q` from the project folder
after `python src/generate_data.py` and `python src/analyse.py`."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import kpi_lib as K  # noqa: E402
from kpi_sql import kpi_query  # noqa: E402


@pytest.fixture(scope="module")
def star():
    return K.load_star()


def test_star_schema_keys_are_valid(star):
    f = star["fact_production"]
    assert not f.duplicated(["date_key", "shift_key", "line_key"]).any()  # declared grain
    for fk, dim in [("date_key", "dim_date"), ("line_key", "dim_line"), ("shift_key", "dim_shift"), ("product_key", "dim_product")]:
        assert set(f[fk]) <= set(star[dim][fk])
    assert set(star["fact_downtime"]["reason_key"]) <= set(star["dim_downtime_reason"]["reason_key"])
    assert set(star["fact_defects"]["defect_key"]) <= set(star["dim_defect"]["defect_key"])


def test_counts_and_minutes_are_consistent(star):
    f = star["fact_production"]
    assert (f["first_pass_good_count"] <= f["good_count"]).all()
    assert (f["good_count"] <= f["total_count"]).all()
    assert (f["total_count"] - f["good_count"] == f["scrap_count"]).all()
    assert ((f["run_minutes"] + f["unplanned_downtime_minutes"] - f["planned_production_minutes"]).abs() < 0.11).all()
    ev = star["fact_downtime"].groupby(["date_key", "shift_key", "line_key"])["duration_minutes"].sum()
    joined = f.set_index(["date_key", "shift_key", "line_key"])["unplanned_downtime_minutes"]
    diff = (joined - ev.reindex(joined.index, fill_value=0)).abs()
    assert diff.max() < 0.5  # events are stored to 0.1 min


def test_oee_toy_example():
    # one shift: 450 planned, 400 run, ideal 30 s, 720 made, 700 first-pass good
    p = pd.DataFrame({"planned_production_minutes": [450.0], "run_minutes": [400.0], "unplanned_downtime_minutes": [50.0],
                      "total_count": [720], "first_pass_good_count": [700], "good_count": [710], "scrap_count": [10],
                      "rework_count": [10], "target_units": [765], "ideal_cycle_time_s": [30]})
    p["ideal_minutes_total"] = p["ideal_cycle_time_s"] * p["total_count"] / 60
    p["ideal_minutes_fpg"] = p["ideal_cycle_time_s"] * p["first_pass_good_count"] / 60
    k = K.kpis(p)
    assert k["availability"] == pytest.approx(400 / 450)
    assert k["performance"] == pytest.approx(360 / 400)
    assert k["quality"] == pytest.approx(700 / 720)
    assert k["oee"] == pytest.approx(350 / 450)
    assert k["oee"] == pytest.approx(k["oee_direct"])


def test_python_matches_sql(star):
    p = K.production_view(star)
    py, sq = K.kpis(p), kpi_query().iloc[0]
    for m in ["availability", "performance", "quality", "output_vs_target"]:
        assert py[m] == pytest.approx(sq[m], abs=1e-12)
    assert py["oee_direct"] == pytest.approx(sq["oee"], abs=1e-12)


def test_pareto_sums_to_one(star):
    par = K.downtime_pareto(star)
    assert par["cumulative_share"].iloc[-1] == pytest.approx(1.0)
    assert par["minutes"].is_monotonic_decreasing


def test_reported_detection_finds_planted_issues():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    d = m["detection"]
    assert d["P1_solder_spike"]["recall_weeks"] == 1.0
    assert d["P2_line_reason_scan"]["found_planted"]
    assert d["P4_night_shift"]["ci95"][1] < 0
