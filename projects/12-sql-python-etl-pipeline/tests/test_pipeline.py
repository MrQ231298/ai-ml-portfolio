"""Integration tests: run the real pipeline into temporary DuckDB files."""

import json
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cli  # noqa: E402
from evaluate import checksum  # noqa: E402

BATCHES = sorted(p for p in (ROOT / "data" / "raw").iterdir() if p.is_dir())


def test_full_run_reconciles_to_truth(tmp_path):
    db = tmp_path / "w.duckdb"
    assert cli.run_batches(BATCHES, db, quiet=True) == 0
    truth = json.loads((ROOT / "data" / "truth" / "expected_totals.json").read_text())
    con = duckdb.connect(str(db), read_only=True)
    net = float(con.execute("SELECT SUM(net_revenue_eur) FROM mart.monthly_revenue").fetchone()[0])
    assert round(net, 2) == truth["net_revenue_eur"]
    assert con.execute("SELECT COUNT(*) FROM core.customer").fetchone()[0] == truth["unique_customers"]


def test_rerun_and_force_are_idempotent(tmp_path):
    db = tmp_path / "w.duckdb"
    cli.run_batches(BATCHES, db, quiet=True)
    first = checksum(db)
    cli.run_batches(BATCHES, db, quiet=True)
    cli.run_batches(BATCHES, db, force=True, quiet=True)
    assert checksum(db) == first


def test_missing_fx_rate_fails_the_run(tmp_path):
    bad = tmp_path / "raw"
    for b in BATCHES:
        shutil.copytree(b, bad / b.name)
    fx = pd.read_csv(bad / "batch_2026_10" / "fx_rates.csv")
    fx[fx["month"] != "2026-09"].to_csv(bad / "batch_2026_10" / "fx_rates.csv", index=False)
    fx = pd.read_csv(bad / "batch_2026_06" / "fx_rates.csv")
    fx[fx["month"] != "2026-09"].to_csv(bad / "batch_2026_06" / "fx_rates.csv", index=False)
    assert cli.run_batches(sorted(bad.iterdir()), tmp_path / "w.duckdb", quiet=True) == 1


def test_untagged_batch_is_refused(tmp_path):
    shutil.copytree(BATCHES[0], tmp_path / "b")
    m = json.loads((tmp_path / "b" / "manifest.json").read_text())
    m["data_flag"] = "LIVE"
    (tmp_path / "b" / "manifest.json").write_text(json.dumps(m))
    try:
        cli.run_batches([tmp_path / "b"], tmp_path / "w.duckdb", quiet=True)
        assert False, "should refuse"
    except ValueError:
        pass
