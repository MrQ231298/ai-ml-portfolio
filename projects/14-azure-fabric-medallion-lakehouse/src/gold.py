"""
GOLD - star schema and aggregates for Power BI (Direct Lake in Fabric).
Rebuilt in full from silver each run (cheap at this size; an
incremental MERGE pattern is described in docs/AZURE_FABRIC_MAPPING.md).
"""

from __future__ import annotations

from pathlib import Path

from common import duck, path, read, write

GOLD = {
    "dim_product": """
        SELECT stock_code,
               (array_agg(description ORDER BY n DESC, description))[1] AS description,  -- most common wins; ties alphabetical (deterministic)
               COUNT(DISTINCT description) AS description_variants
        FROM (SELECT stock_code, description, COUNT(*) n FROM silver
              WHERE line_type IN ('sale', 'cancellation') AND description IS NOT NULL GROUP BY 1, 2)
        GROUP BY stock_code ORDER BY stock_code""",
    "dim_customer": """
        SELECT customer_id,
               (array_agg(country ORDER BY invoice_ts DESC, country))[1] AS country,     -- latest country seen; deterministic ties
               MIN(CASE WHEN line_type = 'sale' THEN invoice_date END) AS first_purchase_date,  -- NULL: cancellations only
               COUNT(DISTINCT country) AS countries_seen
        FROM silver WHERE NOT is_guest AND line_type IN ('sale', 'cancellation') GROUP BY customer_id ORDER BY customer_id""",
    "dim_date": """
        SELECT d::DATE AS date, year(d) AS year, month(d) AS month, strftime(d, '%Y-%m') AS year_month,
               isodow(d) AS iso_weekday, strftime(d, '%A') AS weekday
        FROM range(DATE '2009-12-01', DATE '2012-01-01', INTERVAL 1 DAY) t(d)""",
    "fact_sales": """
        SELECT line_key, invoice_no, invoice_date, invoice_ts, stock_code, customer_id, is_guest, country,
               line_type, quantity, unit_price_gbp, line_amount_gbp
        FROM silver WHERE line_type IN ('sale', 'cancellation')""",
    "agg_monthly_country": """
        SELECT strftime(invoice_date, '%Y-%m') AS year_month, country,
               COUNT(DISTINCT CASE WHEN line_type = 'sale' THEN invoice_no END) AS invoices,
               COUNT(DISTINCT CASE WHEN line_type = 'sale' AND NOT is_guest THEN customer_id END) AS customers,
               SUM(CASE WHEN line_type = 'sale' THEN line_amount_gbp ELSE 0 END) AS gross_sales_gbp,
               SUM(CASE WHEN line_type = 'cancellation' THEN line_amount_gbp ELSE 0 END) AS cancellations_gbp,
               SUM(line_amount_gbp) AS net_sales_gbp
        FROM fact_sales GROUP BY ALL ORDER BY 1, 2""",
    "agg_product_monthly": """
        SELECT strftime(f.invoice_date, '%Y-%m') AS year_month, f.stock_code, p.description,
               SUM(CASE WHEN line_type = 'sale' THEN quantity ELSE 0 END) AS units_sold,
               SUM(line_amount_gbp) AS net_sales_gbp
        FROM fact_sales f LEFT JOIN dim_product p USING (stock_code) GROUP BY ALL ORDER BY 1, 5 DESC""",
}


def build(base: Path) -> dict:
    con = duck()
    con.register("silver", read(path("silver", base)))
    out = {}
    for name, sql in GOLD.items():
        tbl = con.sql(sql).to_arrow_table()
        con.register(name, tbl)
        write(path("gold", base) / name, tbl, mode="overwrite")
        out[name] = int(tbl.num_rows)
    return out
