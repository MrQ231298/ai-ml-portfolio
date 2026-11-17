# Fabric notebook (PySpark / Spark SQL) - SILVER transform
# STATUS: NOT EXECUTED. Spark SQL port of src/silver.py (DuckDB SQL); the
# logic is identical, only function names differ (TRY_CAST -> try_cast etc.).
non_product = ["POST", "DOT", "M", "D", "C2", "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS", "ADJUST", "ADJUST2",
               "S", "B", "TEST001", "TEST002", "SP1002", "GIFT_0001_10", "GIFT_0001_20", "GIFT_0001_30", "GIFT_0001_40", "GIFT_0001_50"]
batch_id = spark.conf.get("spark.app.id")  # in Fabric: a parameter cell set by the pipeline
changed_months = [r._source_month for r in spark.sql(
    f"SELECT DISTINCT _source_month FROM bronze_online_retail_ii_raw WHERE _batch_id = '{batch_id}'").collect()]
months_sql = ", ".join(f"'{m}'" for m in changed_months)
np_sql = ", ".join(f"'{c}'" for c in non_product)

silver = spark.sql(f"""
WITH typed AS (
  SELECT trim(invoice) AS invoice_no, upper(trim(stock_code)) AS stock_code, nullif(trim(description), '') AS description,
         try_cast(quantity AS INT) AS quantity, try_cast(invoice_date AS TIMESTAMP) AS invoice_ts,
         try_cast(price AS DECIMAL(12,2)) AS unit_price_gbp,
         CAST(try_cast(try_cast(customer_id AS DOUBLE) AS BIGINT) AS STRING) AS customer_id, trim(country) AS country,
         _source_file, _source_month, _batch_id,
         monotonically_increasing_id() AS _row_number
  FROM bronze_online_retail_ii_raw WHERE _source_month IN ({months_sql})),
dedup AS (
  SELECT *, row_number() OVER (PARTITION BY invoice_no, stock_code, description, quantity, invoice_ts,
                                            unit_price_gbp, customer_id, country ORDER BY _row_number) AS dup_rank FROM typed)
SELECT md5(concat_ws('|', invoice_no, stock_code, coalesce(description,''), quantity, invoice_ts, unit_price_gbp,
                     coalesce(customer_id,''), country)) AS line_key,
       invoice_no, stock_code, description, quantity, invoice_ts, CAST(invoice_ts AS DATE) AS invoice_date, unit_price_gbp,
       CAST(quantity * unit_price_gbp AS DECIMAL(14,2)) AS line_amount_gbp,
       coalesce(customer_id, 'UNKNOWN') AS customer_id, customer_id IS NULL AS is_guest, country,
       CASE WHEN invoice_no LIKE 'A%' THEN 'bad_debt_adjustment'
            WHEN stock_code IN ({np_sql}) THEN 'non_product'
            WHEN invoice_no LIKE 'C%' THEN 'cancellation'
            WHEN quantity < 0 THEN 'stock_adjustment'
            WHEN unit_price_gbp <= 0 THEN 'zero_price' ELSE 'sale' END AS line_type,
       _source_file, _source_month, _batch_id
FROM dedup WHERE dup_rank = 1 AND quantity IS NOT NULL AND invoice_ts IS NOT NULL AND unit_price_gbp IS NOT NULL
""")
(silver.write.format("delta").mode("overwrite").option("replaceWhere", f"_source_month IN ({months_sql})")
       .partitionBy("_source_month").saveAsTable("silver_sales_line"))
