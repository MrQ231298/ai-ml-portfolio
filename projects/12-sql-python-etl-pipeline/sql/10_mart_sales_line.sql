-- Reporting fact: one row per order line, money in EUR, completed orders flagged.
CREATE OR REPLACE TABLE mart.fct_sales_line AS
SELECT
    l.order_id,
    l.line_no,
    o.customer_id,
    l.product_id,
    CAST(o.order_ts AS DATE)                                     AS order_date,
    strftime(o.order_ts, '%Y-%m')                                AS order_month,
    o.channel,
    o.status,
    o.currency,
    l.quantity,
    l.is_return,
    CASE WHEN o.currency = 'EUR' THEN 1.0 ELSE fx.eur_per_unit END AS fx_eur_per_unit,
    ROUND(CAST(l.quantity * l.unit_price * (1 - l.discount_pct)
               * CASE WHEN o.currency = 'EUR' THEN 1.0 ELSE fx.eur_per_unit END AS DECIMAL(18, 6)), 2) AS net_eur,
    ROUND(l.quantity * p.unit_cost_eur, 2)                        AS cost_eur
FROM core.order_line l
JOIN core.sales_order o USING (order_id)
JOIN core.product p USING (product_id)
LEFT JOIN core.fx_rate fx ON fx.month = strftime(o.order_ts, '%Y-%m') AND fx.currency = o.currency;
