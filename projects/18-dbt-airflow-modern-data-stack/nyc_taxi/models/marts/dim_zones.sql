select location_id, borough, zone, service_zone,
       zone in ('JFK Airport', 'LaGuardia Airport', 'Newark Airport') as is_airport
from {{ ref('stg_taxi_zones') }}
