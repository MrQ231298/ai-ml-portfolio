-- Valid trips only, enriched with zone names and payment descriptions.
select
    t.trip_key, t.vendor_id, t.pickup_at, t.dropoff_at, t.duration_min,
    t.passenger_count, t.trip_distance_miles, t.rate_code_id,
    t.pickup_location_id, pz.borough as pickup_borough, pz.zone as pickup_zone,
    t.dropoff_location_id, dz.borough as dropoff_borough, dz.zone as dropoff_zone,
    t.payment_type, coalesce(p.payment_description, 'Not recorded') as payment_description,
    t.fare_amount, t.tip_amount, t.tolls_amount, t.total_amount,
    t.source_month
from {{ ref('int_trips_flagged') }} t
left join {{ ref('stg_taxi_zones') }} pz on t.pickup_location_id = pz.location_id
left join {{ ref('stg_taxi_zones') }} dz on t.dropoff_location_id = dz.location_id
left join {{ ref('stg_payment_types') }} p on t.payment_type = p.payment_type
where t.dq_reason is null
