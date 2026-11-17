-- Data-quality report: how many raw rows each rule removed, per file month.
-- Incremental by month, like fct_trips, so a monthly run keeps earlier months.
{{ config(materialized='incremental', incremental_strategy='delete+insert', unique_key='source_month') }}

select
    source_month,
    coalesce(dq_reason, 'valid') as outcome,
    count(*) as row_count,
    round(100.0 * count(*) / sum(count(*)) over (partition by source_month), 3) as pct_of_month
from {{ ref('int_trips_flagged') }}
group by 1, 2
