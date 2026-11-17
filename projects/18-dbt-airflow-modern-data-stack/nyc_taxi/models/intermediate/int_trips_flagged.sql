-- Apply the data-quality rules. Every staged row gets exactly one dq_reason
-- (first rule that fails, in the order below) or NULL if it passes.
with base as (
    select
        *,
        date_diff('second', pickup_at, dropoff_at) / 60.0 as duration_min,
        row_number() over (partition by trip_key order by _loaded_at) as dup_rank
    from {{ ref('stg_yellow_trips') }}
)
select
    *,
    case
        when dup_rank > 1 then 'duplicate_record'
        when strftime(pickup_at, '%Y-%m') <> source_month then 'pickup_outside_file_month'
        when duration_min <= 0 or duration_min > 360 then 'duration_out_of_range'
        when trip_distance_miles <= 0 or trip_distance_miles > 200 then 'distance_out_of_range'
        when fare_amount < 0 or total_amount < 0 then 'negative_amount'
        when total_amount > 1000 then 'implausible_total'
        when passenger_count = 0 then 'zero_passengers'
        else null
    end as dq_reason
from base
