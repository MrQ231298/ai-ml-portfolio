"""Pipeline tests on a 11-row fixture landing zone (fast, no download needed)."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_pipeline  # noqa: E402
from common import path, read  # noqa: E402
from evaluate import checksum  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "landing"


@pytest.fixture
def lake(tmp_path):
    r = run_pipeline.run(tmp_path, landing=FIX, quiet=True)
    return tmp_path, r


def test_layers_and_classification(lake):
    base, r = lake
    assert r["bronze"]["rows"] == 11 and r["silver"]["exact_duplicates_removed"] == 1
    s = read(path("silver", base)).to_pandas().set_index("invoice_no")
    assert s.loc["500002", "line_type"] == "non_product"
    assert s.loc["C500003", "line_type"] == "cancellation"
    assert s.loc["500005", "line_type"] == "stock_adjustment"
    assert s.loc["500006", "line_type"] == "zero_price"
    assert s.loc["A500007", "line_type"] == "bad_debt_adjustment"
    assert s.loc["500004", "customer_id"] == "UNKNOWN" and bool(s.loc["500004", "is_guest"])
    assert s.loc["500101", "stock_code"] == "85123A"   # case conformed
    assert r["failed"] == 0


def test_gold_reconciles_and_dims_cover_fact(lake):
    base, _ = lake
    fact = read(path("gold", base) / "fact_sales").to_pandas()
    agg = read(path("gold", base) / "agg_monthly_country").to_pandas()
    assert float(fact.line_amount_gbp.sum()) == pytest.approx(float(agg.net_sales_gbp.sum()))
    assert set(fact.line_type) == {"sale", "cancellation"}
    prod = read(path("gold", base) / "dim_product").to_pandas().set_index("stock_code")
    # most common description wins; trailing space removed by trim
    assert prod.loc["85123A", "description"] == "WHITE HANGING HEART T-LIGHT HOLDER"


def test_replay_is_idempotent(lake):
    base, _ = lake
    before = checksum(base)
    r = run_pipeline.run(base, landing=FIX, quiet=True)
    assert r["bronze"]["files_ingested"] == 0 and checksum(base) == before


def test_corrected_file_replaces_only_its_month(lake, tmp_path_factory):
    base, _ = lake
    land = tmp_path_factory.mktemp("land")
    shutil.copytree(FIX, land, dirs_exist_ok=True)
    df = pd.read_csv(land / "2010-02.csv", dtype=str)
    df.loc[0, "Quantity"] = "24"
    df.to_csv(land / "2010-02.csv", index=False)
    r = run_pipeline.run(base, landing=land, quiet=True)
    assert r["bronze"]["files_replaced"] == 1 and r["silver"]["months_rebuilt"] == 1
    s = read(path("silver", base)).to_pandas()
    assert len(s) == 10  # 11 raw - 1 duplicate, no extra rows after the correction
    assert int(s.loc[s.invoice_no == "500101", "quantity"].iloc[0]) == 24


def test_full_run_report_if_present():
    p = ROOT / "reports" / "metrics.json"
    if not p.exists():
        pytest.skip("run src/evaluate.py first")
    m = json.loads(p.read_text())
    assert m["incremental"]["identical_to_full_run"] and m["replay"]["identical_after_replay"]
    assert m["run_locally_not_on_azure"] is True
