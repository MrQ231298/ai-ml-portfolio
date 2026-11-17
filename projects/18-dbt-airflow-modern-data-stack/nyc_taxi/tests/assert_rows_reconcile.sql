-- Every raw row is either in fct_trips or counted as rejected: nothing is lost or invented.
with raw_counts as (
    select source_month, count(*) as raw_rows from {{ source('raw', 'yellow_trips') }} group by 1
), outcome as (
    select source_month,
           sum(case when outcome = 'valid' then row_count else 0 end) as valid_rows,
           sum(case when outcome <> 'valid' then row_count else 0 end) as rejected_rows
    from {{ ref('dq_rejections_by_month') }} group by 1
), fact as (
    select source_month, count(*) as fact_rows from {{ ref('fct_trips') }} group by 1
)
select r.source_month, r.raw_rows, o.valid_rows, o.rejected_rows, f.fact_rows
from raw_counts r
left join outcome o using (source_month)
left join fact f using (source_month)
where r.raw_rows <> o.valid_rows + o.rejected_rows or o.valid_rows <> coalesce(f.fact_rows, 0)
