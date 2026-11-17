"""
Command-line entry point.

    python src/cli.py run data/raw/batch_2026_06       # load one batch
    python src/cli.py run-all                          # every batch in data/raw, in name order
    python src/cli.py run-all --force                  # reprocess even if files are unchanged
    python src/cli.py status                           # row counts, open quarantine, last DQ results
    python src/cli.py export                           # marts -> reports/marts/*.csv
    python src/cli.py reset                            # delete the warehouse file

Exit code 1 if any `error` data-quality check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etl.extract import read_batch  # noqa: E402
from etl.load import already_loaded, build_marts, connect  # noqa: E402
from etl.validate import run_checks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_DB = ROOT / "warehouse" / "retail.duckdb"
MARTS = ["fct_sales_line", "monthly_revenue", "customer_summary", "product_performance"]


def run_batches(folders, db=DEFAULT_DB, force=False, quiet=False) -> int:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = connect(db)
    run_id = con.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM etl.batch_log").fetchone()[0]
    loaded_any, last_extract = False, None
    for f in folders:
        b = read_batch(f)
        last_extract = str(b.extract_date.date())
        if already_loaded(con, b) and not force:
            if not quiet:
                print(f"[skip] {b.batch_id}: identical files already loaded (use --force to reprocess)")
            continue
        from etl.load import load_batch
        s = load_batch(con, b, run_id)
        loaded_any = True
        if not quiet:
            print(f"[load] {b.batch_id}: " + ", ".join(f"{t}.{m}={v}" for (t, m), v in sorted(s.items()) if m in ("rows_in", "upserted")))
    if last_extract is None:
        last_extract = con.execute("SELECT MAX(CAST(order_ts AS DATE)) FROM core.sales_order").fetchone()[0]
    build_marts(con, str(last_extract))
    res = run_checks(con, run_id)
    failed = [r for r in res if not r["passed"] and r["severity"] == "error"]
    if not quiet:
        for r in res:
            print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['severity']:5s} {r['check_name']:28s} observed={r['observed']:.4g} rule {r['detail']}")
        print("loaded" if loaded_any else "nothing new", "| DQ:", "FAILED" if failed else "all error-level checks passed")
    con.close()
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Retail ETL pipeline (SYNTHETIC demo data)")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("batch"); r.add_argument("--force", action="store_true")
    ra = sub.add_parser("run-all"); ra.add_argument("--force", action="store_true")
    sub.add_parser("status"); sub.add_parser("export"); sub.add_parser("reset")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run_batches([Path(a.batch)], a.db, a.force)
    if a.cmd == "run-all":
        return run_batches(sorted(p for p in RAW.iterdir() if p.is_dir()), a.db, a.force)
    if a.cmd == "reset":
        Path(a.db).unlink(missing_ok=True); print("warehouse removed"); return 0
    con = connect(a.db)
    if a.cmd == "status":
        for t in ["core.customer", "core.product", "core.sales_order", "core.order_line"]:
            print(f"{t:22s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:7d}")
        print(con.execute("""SELECT table_name, reason, COUNT(*) AS n, COUNT(resolved_batch) AS resolved
                             FROM etl.quarantine GROUP BY ALL ORDER BY 1, 2""").df().to_string(index=False))
    if a.cmd == "export":
        out = ROOT / "reports" / "marts"; out.mkdir(parents=True, exist_ok=True)
        for m in MARTS[1:]:
            con.execute(f"SELECT * FROM mart.{m}").df().to_csv(out / f"{m}.csv", index=False)
        print(f"exported {len(MARTS) - 1} marts to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
