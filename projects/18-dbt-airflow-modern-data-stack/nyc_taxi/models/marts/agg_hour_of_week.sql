-- Average trips per hour of the week, for staffing and pricing questions.
with hourly as (
    select date_trunc('hour', pickup_at) as hour_start, count(*) as trips
    from {{ ref('fct_trips') }}
    group by 1
)
select
    isodow(hour_start) as iso_weekday,
    hour(hour_start)   as hour_of_day,
    count(*)           as hours_observed,
    round(avg(trips), 1) as avg_trips,
    max(trips)         as max_trips
from hourly
group by 1, 2
