# Fabric notebook (Spark SQL) - GOLD star schema for a Direct Lake semantic model
# STATUS: NOT EXECUTED. Port of src/gold.py; queries are the same apart from
# DuckDB's array_agg(... ORDER BY) which becomes max_by / sort_array in Spark.
spark.sql("""
CREATE OR REPLACE TABLE gold_fact_sales USING DELTA AS
SELECT line_key, invoice_no, invoice_date, invoice_ts, stock_code, customer_id, is_guest, country,
       line_type, quantity, unit_price_gbp, line_amount_gbp
FROM silver_sales_line WHERE line_type IN ('sale', 'cancellation')""")
spark.sql("""
CREATE OR REPLACE TABLE gold_dim_product USING DELTA AS
SELECT stock_code, max_by(description, struct(n, description)) AS description, count(DISTINCT description) AS description_variants
FROM (SELECT stock_code, description, count(*) n FROM silver_sales_line
      WHERE line_type IN ('sale','cancellation') AND description IS NOT NULL GROUP BY 1, 2) GROUP BY stock_code""")
spark.sql("""
CREATE OR REPLACE TABLE gold_dim_customer USING DELTA AS
SELECT customer_id, max_by(country, struct(invoice_ts, country)) AS country,
       min(CASE WHEN line_type = 'sale' THEN invoice_date END) AS first_purchase_date, count(DISTINCT country) AS countries_seen
FROM silver_sales_line WHERE NOT is_guest AND line_type IN ('sale','cancellation') GROUP BY customer_id""")
spark.sql("""
CREATE OR REPLACE TABLE gold_agg_monthly_country USING DELTA AS
SELECT date_format(invoice_date, 'yyyy-MM') AS year_month, country,
       count(DISTINCT CASE WHEN line_type = 'sale' THEN invoice_no END) AS invoices,
       count(DISTINCT CASE WHEN line_type = 'sale' AND NOT is_guest THEN customer_id END) AS customers,
       sum(CASE WHEN line_type = 'sale' THEN line_amount_gbp ELSE 0 END) AS gross_sales_gbp,
       sum(CASE WHEN line_type = 'cancellation' THEN line_amount_gbp ELSE 0 END) AS cancellations_gbp,
       sum(line_amount_gbp) AS net_sales_gbp
FROM gold_fact_sales GROUP BY 1, 2""")
# Then: Lakehouse -> New semantic model -> pick the gold_* tables (Direct Lake), relate fact to dims,
# and reuse the measure pattern from project 11 (sums divided by sums).
