"""
End-to-end evaluation of the pipeline against the generator's ground
truth (data/truth/ is read here only, never by the pipeline).

    python src/evaluate.py   -> reports/metrics.json, reports/dq_report.md, reports/marts/*.csv

Checks: revenue reconciliation, defect detection per type, SCD type 1
updates, and idempotency (re-run, forced re-run, one-batch-at-a-time,
and out-of-order loading).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli  # noqa: E402

ROOT = cli.ROOT
REPORTS = ROOT / "reports"
TRUTH = ROOT / "data" / "truth"
TMP = ROOT / "warehouse" / "eval"
TABLES = {"core.customer": "customer_id", "core.product": "product_id", "core.sales_order": "order_id",
          "core.order_line": "order_id, line_no", "mart.monthly_revenue": "order_month, channel",
          "mart.customer_summary": "customer_id", "mart.product_performance": "product_id"}


def checksum(db: Path) -> dict[str, str]:
    con = duckdb.connect(str(db), read_only=True)
    out = {}
    for t, k in TABLES.items():
        cols = [c[0] for c in con.execute(f"DESCRIBE {t}").fetchall() if c[0] != "last_batch_id"]
        out[t] = con.execute(f"SELECT md5(string_agg(CAST(ROW({', '.join(cols)}) AS VARCHAR), '|' ORDER BY {k})) FROM {t}").fetchone()[0]
    con.close()
    return out


def batches():
    return sorted(p for p in cli.RAW.iterdir() if p.is_dir())


def main() -> None:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    truth = json.loads((TRUTH / "expected_totals.json").read_text())
    injected = pd.read_csv(TRUTH / "injected_defects.csv")

    # --- run 1: clean warehouse, all batches
    db = TMP / "run.duckdb"
    t0 = time.perf_counter()
    rc = cli.run_batches(batches(), db, quiet=True)
    runtime = time.perf_counter() - t0
    ck1 = checksum(db)
    # --- run 2: same files again (should skip), run 3: forced reprocess
    rc2 = cli.run_batches(batches(), db, quiet=True); ck2 = checksum(db)
    rc3 = cli.run_batches(batches(), db, force=True, quiet=True); ck3 = checksum(db)
    # --- one batch per invocation into a fresh warehouse
    db_b = TMP / "stepwise.duckdb"
    for b in batches():
        cli.run_batches([b], db_b, quiet=True)
    ck4 = checksum(db_b)
    # --- out-of-order: newest batch first
    db_r = TMP / "reversed.duckdb"
    for b in reversed(batches()):
        cli.run_batches([b], db_r, quiet=True)
    ck5 = checksum(db_r)

    con = duckdb.connect(str(db), read_only=True)
    net = float(con.execute("SELECT SUM(net_revenue_eur) FROM mart.monthly_revenue").fetchone()[0])
    lines = con.execute("SELECT COUNT(*) FROM mart.fct_sales_line WHERE status = 'completed'").fetchone()[0]
    orders = con.execute("SELECT COUNT(DISTINCT order_id) FROM mart.fct_sales_line WHERE status = 'completed'").fetchone()[0]
    by_month = con.execute("SELECT order_month, SUM(net_revenue_eur) FROM mart.monthly_revenue GROUP BY 1 ORDER BY 1").fetchall()
    month_diff = max(abs(float(v) - truth["net_revenue_by_month"][m]) for m, v in by_month)
    q = con.execute("SELECT table_name, record_key, reason, resolved_batch FROM etl.quarantine").df()
    open_q = q[q.resolved_batch.isna()]
    null_email = set(r[0] for r in con.execute("SELECT customer_id FROM core.customer WHERE email IS NULL").fetchall())
    stats = con.execute("SELECT batch_id, table_name, metric, value FROM etl.run_stats WHERE run_id = 1").df()
    dq = con.execute("SELECT check_name, severity, passed, observed, detail FROM etl.dq_result WHERE run_id = 1").df()
    sligo_expected = set(pd.read_csv(cli.RAW / "batch_2026_10" / "customers.csv").query("county == 'Sligo'").customer_id)
    sligo_core = set(r[0] for r in con.execute("SELECT customer_id FROM core.customer WHERE county = 'Sligo'").fetchall())
    missing_price = set(injected.loc[injected.defect_type == "missing_price", "record_key"])
    cascade = set(r[0] for r in con.execute(
        "SELECT record_key FROM etl.quarantine WHERE reason = 'unknown_product' AND json_extract_string(payload, '$.product_id') IN "
        f"({', '.join(repr(x) for x in missing_price)})").fetchall())
    con.close()
    con_r = duckdb.connect(str(db_r), read_only=True)
    sligo_rev = set(r[0] for r in con_r.execute("SELECT customer_id FROM core.customer WHERE county = 'Sligo'").fetchall())
    con_r.close()

    def pr(found: set, planted: set) -> dict:
        tp = len(found & planted)
        return {"injected": len(planted), "flagged": len(found), "true_positive": tp,
                "recall": tp / len(planted) if planted else None, "precision": tp / len(found) if found else None}

    inj = lambda kind: set(injected.loc[injected.defect_type == kind, "record_key"])  # noqa: E731
    oq = lambda t, r: set(open_q.loc[(open_q.table_name == t) & (open_q.reason == r), "record_key"])  # noqa: E731
    orphan_prod_flagged = oq("order_lines", "unknown_product")
    detection = {
        "invalid_email": pr(null_email, inj("invalid_email")),
        "orphan_customer": pr(oq("orders", "unknown_customer"), inj("orphan_customer")),
        "future_date": pr(oq("orders", "future_date"), inj("future_date")),
        "null_or_zero_quantity": pr(oq("order_lines", "null_or_zero_quantity"), inj("null_or_zero_quantity")),
        "orphan_product": pr(orphan_prod_flagged - cascade, inj("orphan_product")),
        "missing_price": pr(oq("products", "missing_or_invalid_price"), missing_price),
        "lines_blocked_by_missing_price": len(cascade & orphan_prod_flagged),
        "late_arriving_orders_resolved_by_retry": int(((q.reason == "unknown_customer") & q.resolved_batch.notna()).sum()),
        "late_arriving_lines_resolved_by_retry": int(((q.reason == "unknown_order") & q.resolved_batch.notna()).sum()),
        "exact_duplicates_removed": {t: int(stats.loc[(stats.table_name == t) & (stats.metric == "exact_duplicates_removed"), "value"].sum())
                                     for t in ["customers", "order_lines", "orders", "products"]},
        "injected_exact_duplicates": {"customers": int(((injected.table == "customers") & (injected.defect_type == "exact_duplicate")).sum()),
                                      "order_lines": int(((injected.table == "order_lines") & (injected.defect_type == "exact_duplicate")).sum())},
    }
    metrics = {
        "data_flag": "SYNTHETIC_EXAMPLE_DATA",
        "runtime_seconds_two_batches": round(runtime, 2),
        "raw_rows": {b.name: {f: int(len(pd.read_csv(b / f))) for f in ["customers.csv", "orders.csv", "order_lines.csv"]} for b in batches()},
        "reconciliation": {"expected_net_revenue_eur": truth["net_revenue_eur"], "pipeline_net_revenue_eur": round(net, 2),
                           "difference_eur": round(net - truth["net_revenue_eur"], 2),
                           "max_monthly_difference_eur": round(month_diff, 2),
                           "expected_completed_lines": truth["completed_order_lines"], "pipeline_completed_lines": int(lines),
                           "expected_completed_orders": truth["completed_orders"], "pipeline_completed_orders": int(orders)},
        "detection": detection,
        "scd1_county_update": {"customers_moved_in_batch_2": len(sligo_expected), "in_order_load_correct": sligo_core == sligo_expected,
                               "reversed_order_load_correct": sligo_rev == sligo_expected,
                               "reversed_order_customers_with_stale_county": len(sligo_expected - sligo_rev)},
        "idempotency": {"rerun_same_files_identical": ck1 == ck2, "forced_reprocess_identical": ck1 == ck3,
                        "one_batch_per_run_identical": ck1 == ck4, "reversed_order_identical": ck1 == ck5,
                        "reversed_order_tables_differing": [t for t in TABLES if ck1[t] != ck5[t]],
                        "exit_codes": [rc, rc2, rc3]},
        "dq_checks": dq.to_dict("records"),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    # data-quality report for humans
    md = ["# Data-quality report (SYNTHETIC demo data)", "", f"Generated by `python src/evaluate.py`. Runtime for two batches: {runtime:.1f} s.", "",
          "| Check | Severity | Result | Observed | Rule |", "| --- | --- | --- | ---: | --- |"]
    for r in dq.itertuples():
        md.append(f"| `{r.check_name}` | {r.severity} | {'PASS' if r.passed else 'FAIL'} | {r.observed:.4g} | `{r.detail}` |")
    md += ["", "## Quarantine (open = still blocked after the latest batch)", "", "| Table | Reason | Rows | Resolved by retry | Open |", "| --- | --- | ---: | ---: | ---: |"]
    for (t, r), g in q.groupby(["table_name", "reason"]):
        md.append(f"| {t} | {r} | {len(g)} | {g.resolved_batch.notna().sum()} | {g.resolved_batch.isna().sum()} |")
    (REPORTS / "dq_report.md").write_text("\n".join(md) + "\n")
    cli.main(["--db", str(db), "export"])
    print(json.dumps({k: metrics[k] for k in ["runtime_seconds_two_batches", "reconciliation", "detection", "scd1_county_update", "idempotency"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
