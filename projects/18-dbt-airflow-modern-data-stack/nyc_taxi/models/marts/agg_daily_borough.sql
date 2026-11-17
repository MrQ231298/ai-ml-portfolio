-- Daily KPIs by pickup borough (rebuilt from the fact table each run).
select
    cast(pickup_at as date)                                  as trip_date,
    pickup_borough,
    count(*)                                                 as trips,
    sum(total_amount)                                        as revenue,
    round(avg(fare_amount), 2)                               as avg_fare,
    round(avg(duration_min), 2)                              as avg_duration_min,
    round(avg(trip_distance_miles), 2)                       as avg_distance_miles,
    round(100 * sum(case when payment_type = 1 then tip_amount end)
              / nullif(sum(case when payment_type = 1 then fare_amount end), 0), 2) as card_tip_pct_of_fare
from {{ ref('fct_trips') }}
group by 1, 2
