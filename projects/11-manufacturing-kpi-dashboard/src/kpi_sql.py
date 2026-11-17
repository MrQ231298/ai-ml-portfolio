"""
Independent SQL implementation of the headline measures (DuckDB over the
same CSVs). Used to cross-check `kpi_lib.py`; the queries are also a
template for a SQL Server / Fabric warehouse view.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
STAR = ROOT / "data" / "star_schema"

KPI_SQL = """
WITH p AS (
    SELECT f.*, t.target_units, l.line, s.shift
    FROM read_csv_auto('{star}/fact_production.csv') f
    JOIN read_csv_auto('{star}/fact_target.csv') t USING (date_key, shift_key, line_key)
    JOIN read_csv_auto('{star}/dim_line.csv') l USING (line_key)
    JOIN read_csv_auto('{star}/dim_shift.csv') s USING (shift_key)
)
SELECT {group_cols}
    SUM(run_minutes) / SUM(planned_production_minutes)                         AS availability,
    SUM(ideal_cycle_time_s * total_count / 60.0) / SUM(run_minutes)            AS performance,
    SUM(first_pass_good_count)::DOUBLE / SUM(total_count)                      AS quality,
    SUM(ideal_cycle_time_s * first_pass_good_count / 60.0)
        / SUM(planned_production_minutes)                                      AS oee,
    SUM(good_count)::DOUBLE / SUM(target_units)                                AS output_vs_target
FROM p
{group_by}
ORDER BY 1
"""


def kpi_query(by: str | None = None):
    sql = KPI_SQL.format(star=STAR.as_posix(), group_cols=f"{by}," if by else "",
                         group_by=f"GROUP BY {by}" if by else "")
    return duckdb.sql(sql).df()


if __name__ == "__main__":
    print(kpi_query())
    print(kpi_query("line"))
