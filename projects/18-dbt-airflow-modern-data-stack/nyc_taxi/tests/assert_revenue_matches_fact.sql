-- The daily aggregate must add back to the fact table to the cent.
select * from (
    select abs((select sum(revenue) from {{ ref('agg_daily_borough') }})
             - (select sum(total_amount) from {{ ref('fct_trips') }})) as diff
) where diff > 0.005
