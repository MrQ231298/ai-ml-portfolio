"""
SILVER - typed, de-duplicated, conformed sales lines.
Equivalent to a Fabric Spark notebook writing a Lakehouse Delta table.
Only months whose bronze partition changed are rebuilt (delete those
months, append the new rows), so re-runs are idempotent and a corrected
file replaces its old rows.
"""

from __future__ import annotations

from pathlib import Path

from common import CONF, duck, exists, path, read, write

SILVER_SQL = """
WITH typed AS (
    SELECT
        trim(invoice)                                      AS invoice_no,
        upper(trim(stock_code))                            AS stock_code,
        nullif(trim(description), '')                      AS description,
        TRY_CAST(quantity AS INTEGER)                      AS quantity,
        TRY_CAST(invoice_date AS TIMESTAMP)                AS invoice_ts,
        TRY_CAST(price AS DECIMAL(12, 2))                  AS unit_price_gbp,
        CAST(TRY_CAST(TRY_CAST(customer_id AS DOUBLE) AS BIGINT) AS VARCHAR) AS customer_id,
        trim(country)                                      AS country,
        _source_file, _source_month, _batch_id, _row_number
    FROM bronze
),
dedup AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY invoice_no, stock_code, description, quantity, invoice_ts, unit_price_gbp, customer_id, country
        ORDER BY _row_number) AS dup_rank
    FROM typed
)
SELECT
    md5(concat_ws('|', invoice_no, stock_code, coalesce(description, ''), quantity, invoice_ts,
                  unit_price_gbp, coalesce(customer_id, ''), country))    AS line_key,
    invoice_no, stock_code, description, quantity, invoice_ts,
    CAST(invoice_ts AS DATE)                                              AS invoice_date,
    unit_price_gbp,
    CAST(quantity * unit_price_gbp AS DECIMAL(14, 2))                     AS line_amount_gbp,
    coalesce(customer_id, 'UNKNOWN')                                      AS customer_id,
    customer_id IS NULL                                                   AS is_guest,
    country,
    CASE
        WHEN invoice_no LIKE 'A%'                     THEN 'bad_debt_adjustment'
        WHEN stock_code IN ({non_product})            THEN 'non_product'
        WHEN invoice_no LIKE 'C%'                     THEN 'cancellation'
        WHEN quantity < 0                             THEN 'stock_adjustment'
        WHEN unit_price_gbp <= 0                      THEN 'zero_price'
        ELSE 'sale'
    END                                                                   AS line_type,
    _source_file, _source_month, _batch_id
FROM dedup
WHERE dup_rank = 1
  AND quantity IS NOT NULL AND invoice_ts IS NOT NULL AND unit_price_gbp IS NOT NULL
"""


def build(base: Path, months: list[str]) -> dict:
    if not months:
        return {"months_rebuilt": 0, "rows_in": 0, "rows_out": 0}
    con = duck()
    bronze = read(path("bronze", base))
    con.register("bronze_all", bronze)
    month_list = ", ".join(f"'{m}'" for m in months)
    con.execute(f"CREATE TEMP TABLE bronze AS SELECT * FROM bronze_all WHERE _source_month IN ({month_list})")
    non_product = ", ".join(f"'{c.upper()}'" for c in CONF["non_product_stock_codes"])
    rows_in = con.execute("SELECT COUNT(*) FROM bronze").fetchall()[0][0]
    exact_dups = con.execute("SELECT COUNT(*) - COUNT(DISTINCT (invoice, stock_code, description, quantity, invoice_date, price, customer_id, country)) FROM bronze").fetchall()[0][0]
    untypeable = con.execute("""SELECT COUNT(*) FROM bronze WHERE TRY_CAST(quantity AS INTEGER) IS NULL
                                OR TRY_CAST(invoice_date AS TIMESTAMP) IS NULL OR TRY_CAST(price AS DECIMAL(12,2)) IS NULL""").fetchall()[0][0]
    out = con.sql(SILVER_SQL.format(non_product=non_product)).to_arrow_table()
    p = path("silver", base)
    if exists(p):
        write(p, out, mode="overwrite", partition_by=["_source_month"], predicate=f"_source_month IN ({month_list})")
    else:
        write(p, out, mode="overwrite", partition_by=["_source_month"])
    return {"months_rebuilt": len(months), "rows_in": int(rows_in), "rows_out": int(out.num_rows),
            "exact_duplicates_removed": int(exact_dups), "untypeable_rows_dropped": int(untypeable)}
