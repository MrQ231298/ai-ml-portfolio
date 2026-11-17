-- One row per customer: lifetime value and recency, for CRM / segmentation.
CREATE OR REPLACE TABLE mart.customer_summary AS
SELECT
    c.customer_id,
    c.country_code,
    c.county,
    c.signup_date,
    c.email IS NOT NULL                                       AS contactable_by_email,
    MIN(s.order_date)                                         AS first_order_date,
    MAX(s.order_date)                                         AS last_order_date,
    COUNT(DISTINCT s.order_id)                                AS completed_orders,
    COALESCE(SUM(s.net_eur), 0)                               AS net_revenue_eur,
    DATE_DIFF('day', MAX(s.order_date), DATE '{extract_date}') AS recency_days
FROM core.customer c
LEFT JOIN mart.fct_sales_line s ON s.customer_id = c.customer_id AND s.status = 'completed'
GROUP BY ALL;
