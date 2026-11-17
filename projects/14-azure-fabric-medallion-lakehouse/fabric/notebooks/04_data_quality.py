# Fabric notebook - DATA QUALITY gate
# STATUS: NOT EXECUTED. Same checks as src/dq.py; raises to fail the pipeline run.
checks = {
    "silver_line_key_unique": "SELECT count(*) - count(DISTINCT line_key) FROM silver_sales_line",
    "gold_fact_products_in_dim": "SELECT count(*) FROM gold_fact_sales f LEFT ANTI JOIN gold_dim_product p ON f.stock_code = p.stock_code",
    "gold_agg_reconciles_to_fact": "SELECT abs((SELECT sum(net_sales_gbp) FROM gold_agg_monthly_country) - (SELECT sum(line_amount_gbp) FROM gold_fact_sales))",
}
failed = {k: v for k, v in ((k, spark.sql(q).first()[0]) for k, q in checks.items()) if v and float(v) > 0.005}
if failed:
    raise Exception(f"DQ failed: {failed}")
