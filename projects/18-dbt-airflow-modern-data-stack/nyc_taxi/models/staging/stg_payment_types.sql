select payment_type, payment_description from {{ ref('payment_types') }}
