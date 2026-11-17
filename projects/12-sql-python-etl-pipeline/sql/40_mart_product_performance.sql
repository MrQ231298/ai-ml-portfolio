-- Product league table with margin and return rate.
CREATE OR REPLACE TABLE mart.product_performance AS
SELECT
    p.product_id,
    p.product_name,
    p.category,
    SUM(CASE WHEN NOT s.is_return THEN s.quantity ELSE 0 END)          AS units_sold,
    SUM(CASE WHEN s.is_return THEN -s.quantity ELSE 0 END)             AS units_returned,
    COALESCE(SUM(s.net_eur), 0)                                        AS net_revenue_eur,
    COALESCE(SUM(s.net_eur) - SUM(s.cost_eur), 0)                      AS gross_margin_eur,
    ROUND(SUM(CASE WHEN s.is_return THEN -s.quantity ELSE 0 END)::DOUBLE
          / NULLIF(SUM(CASE WHEN NOT s.is_return THEN s.quantity ELSE 0 END), 0), 4) AS return_rate
FROM core.product p
LEFT JOIN mart.fct_sales_line s ON s.product_id = p.product_id AND s.status = 'completed'
GROUP BY ALL
ORDER BY net_revenue_eur DESC;
