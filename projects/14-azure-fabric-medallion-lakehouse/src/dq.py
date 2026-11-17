"""Data-quality expectations per layer (stored as a Delta table, like a Fabric DQ log)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from common import CONF, duck, path, read, write


def run(base: Path, run_id: str) -> list[dict]:
    con = duck()
    con.register("bronze", read(path("bronze", base)))
    con.register("silver", read(path("silver", base)))
    con.register("control", read(path("control", base)))   # read via the Delta log, never the raw parquet files
    for t in ["fact_sales", "dim_product", "dim_customer", "agg_monthly_country"]:
        con.register(t, read(path("gold", base) / t))
    lim = CONF["dq"]
    checks = [
        ("bronze", "bronze_rows_match_control", "error",
         "SELECT COUNT(*) FROM bronze", "SELECT SUM(rows) FROM control", "eq_other"),
        ("silver", "silver_line_key_unique", "error", "SELECT COUNT(*) - COUNT(DISTINCT line_key) FROM silver", None, "== 0"),
        ("silver", "silver_no_null_core_fields", "error",
         "SELECT COUNT(*) FROM silver WHERE invoice_no IS NULL OR stock_code IS NULL OR invoice_ts IS NULL OR quantity IS NULL", None, "== 0"),
        ("silver", "silver_guest_share", "warn", "SELECT AVG(is_guest::INT) FROM silver", None, f"<= {lim['max_unknown_customer_share_silver']}"),
        ("silver", "silver_zero_price_share", "warn", "SELECT AVG((line_type = 'zero_price')::INT) FROM silver", None,
         f"<= {lim['max_non_positive_price_share_silver']}"),
        ("silver", "silver_line_types_known", "error",
         "SELECT COUNT(*) FROM silver WHERE line_type NOT IN ('sale','cancellation','non_product','stock_adjustment','zero_price','bad_debt_adjustment')", None, "== 0"),
        ("gold", "gold_fact_products_in_dim", "error",
         "SELECT COUNT(*) FROM fact_sales f ANTI JOIN dim_product p USING (stock_code)", None, "== 0"),
        ("gold", "gold_fact_customers_in_dim", "warn",
         "SELECT COUNT(DISTINCT customer_id) FROM fact_sales f WHERE NOT is_guest AND customer_id NOT IN (SELECT customer_id FROM dim_customer)", None, "== 0"),
        ("gold", "gold_agg_reconciles_to_fact", "error",
         "SELECT ABS((SELECT SUM(net_sales_gbp) FROM agg_monthly_country) - (SELECT SUM(line_amount_gbp) FROM fact_sales))", None, "< 0.005"),
        ("gold", "gold_sales_positive", "error", "SELECT COUNT(*) FROM fact_sales WHERE line_type = 'sale' AND line_amount_gbp <= 0", None, "== 0"),
    ]
    res = []
    for layer, name, sev, sql, other, rule in checks:
        v = float(con.execute(sql).fetchall()[0][0] or 0)
        if rule == "eq_other":
            o = float(con.execute(other).fetchall()[0][0] or 0)
            ok, detail = v == o, f"== control rows ({o:.0f})"
        else:
            ok, detail = bool(eval(f"{v!r} {rule}")), rule   # rules are constants above
        res.append({"run_id": run_id, "layer": layer, "check": name, "severity": sev, "passed": ok, "observed": v, "rule": detail})
    write(path("dq", base), pa.Table.from_pandas(pd.DataFrame(res), preserve_index=False), mode="append")
    return res
