-- One row per raw TLC record: renamed, typed, with a deterministic surrogate key.
-- Nothing is filtered here; quality rules live in intermediate/int_trips_flagged.
select
    md5(concat_ws('|', VendorID, tpep_pickup_datetime, tpep_dropoff_datetime, PULocationID, DOLocationID,
                  trip_distance, passenger_count, fare_amount, total_amount, payment_type)) as trip_key,
    cast(VendorID as integer)              as vendor_id,
    tpep_pickup_datetime                   as pickup_at,
    tpep_dropoff_datetime                  as dropoff_at,
    cast(passenger_count as integer)       as passenger_count,
    cast(trip_distance as double)          as trip_distance_miles,
    cast(RatecodeID as integer)            as rate_code_id,
    cast(PULocationID as integer)          as pickup_location_id,
    cast(DOLocationID as integer)          as dropoff_location_id,
    cast(payment_type as integer)          as payment_type,
    cast(fare_amount as decimal(10, 2))    as fare_amount,
    cast(tip_amount as decimal(10, 2))     as tip_amount,
    cast(tolls_amount as decimal(10, 2))   as tolls_amount,
    cast(total_amount as decimal(10, 2))   as total_amount,
    source_month,
    _loaded_at
from {{ source('raw', 'yellow_trips') }}
{% if var('month') %}
where source_month = '{{ var("month") }}'
{% endif %}
