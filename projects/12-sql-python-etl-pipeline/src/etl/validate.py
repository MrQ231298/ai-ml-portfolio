"""
Data-quality checks run after every load (SQL against the warehouse).
`error` checks stop the run with a non-zero exit code; `warn` checks are
reported. Results are stored in etl.dq_result and printed by the CLI.
"""

from __future__ import annotations

CHECKS = [
    # name, severity, SQL returning one number, rule applied to that number
    ("pk_unique_customer", "error", "SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM core.customer", "== 0"),
    ("pk_unique_order_line", "error", "SELECT COUNT(*) - COUNT(DISTINCT (order_id, line_no)) FROM core.order_line", "== 0"),
    ("fk_order_customer", "error", "SELECT COUNT(*) FROM core.sales_order o ANTI JOIN core.customer c USING (customer_id)", "== 0"),
    ("fk_line_order", "error", "SELECT COUNT(*) FROM core.order_line l ANTI JOIN core.sales_order o USING (order_id)", "== 0"),
    ("fk_line_product", "error", "SELECT COUNT(*) FROM core.order_line l ANTI JOIN core.product p USING (product_id)", "== 0"),
    ("no_zero_quantity", "error", "SELECT COUNT(*) FROM core.order_line WHERE quantity = 0", "== 0"),
    ("accepted_status", "error", "SELECT COUNT(*) FROM core.sales_order WHERE status NOT IN ('completed', 'cancelled')", "== 0"),
    ("accepted_country", "error", "SELECT COUNT(*) FROM core.customer WHERE country_code NOT IN ('IE', 'GB', 'NI')", "== 0"),
    ("fx_available_for_gbp", "error",
     "SELECT COUNT(*) FROM mart.fct_sales_line WHERE currency = 'GBP' AND fx_eur_per_unit IS NULL", "== 0"),
    ("mart_reconciles_to_fact", "error",
     """SELECT ABS((SELECT SUM(net_revenue_eur) FROM mart.monthly_revenue)
              - (SELECT SUM(net_eur) FROM mart.fct_sales_line WHERE status = 'completed'))""", "< 0.005"),
    ("customer_mart_reconciles", "error",
     """SELECT ABS((SELECT SUM(net_revenue_eur) FROM mart.customer_summary)
              - (SELECT SUM(net_eur) FROM mart.fct_sales_line WHERE status = 'completed'))""", "< 0.005"),
    ("email_null_share", "warn", "SELECT AVG(CASE WHEN email IS NULL THEN 1.0 ELSE 0 END) FROM core.customer", "< 0.05"),
    ("return_line_share", "warn", "SELECT AVG(CASE WHEN is_return THEN 1.0 ELSE 0 END) FROM core.order_line", "< 0.02"),
    ("open_quarantine_line_share", "warn",
     """SELECT (SELECT COUNT(*) FROM etl.quarantine WHERE resolved_batch IS NULL AND table_name = 'order_lines')::DOUBLE
              / (SELECT COUNT(*) FROM core.order_line)""", "< 0.02"),
    ("discount_range", "error", "SELECT COUNT(*) FROM core.order_line WHERE discount_pct < 0 OR discount_pct > 0.9", "== 0"),
]


def run_checks(con, run_id: int) -> list[dict]:
    out = []
    for name, sev, sql, rule in CHECKS:
        v = con.execute(sql).fetchone()[0]
        v = float(v) if v is not None else 0.0
        passed = bool(eval(f"{v!r} {rule}"))  # rule strings are constants defined above
        out.append({"check_name": name, "severity": sev, "passed": passed, "observed": v, "detail": rule})
        con.execute("INSERT OR REPLACE INTO etl.dq_result VALUES (?, ?, ?, ?, ?, ?)", [run_id, name, sev, passed, v, rule])
    return out
