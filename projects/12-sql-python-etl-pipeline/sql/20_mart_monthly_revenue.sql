-- Monthly KPIs by channel (completed orders only).
CREATE OR REPLACE TABLE mart.monthly_revenue AS
SELECT
    order_month,
    channel,
    COUNT(DISTINCT order_id)                                AS orders,
    COUNT(*)                                                AS order_lines,
    SUM(net_eur)                                            AS net_revenue_eur,
    SUM(CASE WHEN is_return THEN -net_eur ELSE 0 END)       AS returns_eur,
    SUM(net_eur) - SUM(cost_eur)                            AS gross_margin_eur,
    ROUND(SUM(net_eur) / COUNT(DISTINCT order_id), 2)       AS avg_order_value_eur
FROM mart.fct_sales_line
WHERE status = 'completed'
GROUP BY ALL
ORDER BY order_month, channel;
