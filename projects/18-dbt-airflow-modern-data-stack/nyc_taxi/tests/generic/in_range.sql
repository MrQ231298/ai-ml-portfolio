{% test in_range(model, column_name, min_value, max_value, inclusive_min=true) %}
select * from {{ model }}
where {{ column_name }} {{ '<' if inclusive_min else '<=' }} {{ min_value }} or {{ column_name }} > {{ max_value }}
{% endtest %}
