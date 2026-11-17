-- Trip fact table. Incremental by file month: a monthly run deletes and
-- re-inserts only that month (delete+insert on source_month), so reruns and
-- Airflow retries are idempotent. A --full-refresh rebuilds every month.
{{ config(materialized='incremental', incremental_strategy='delete+insert', unique_key='source_month') }}

select * from {{ ref('int_trips_enriched') }}
{% if is_incremental() and var('month') %}
where source_month = '{{ var("month") }}'
{% endif %}
