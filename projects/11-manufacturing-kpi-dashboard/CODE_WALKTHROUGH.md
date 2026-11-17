# Code walkthrough — manufacturing KPI dashboard

## The pipeline at a glance

```
1. python src/generate_data.py
        ↓ data/star_schema/*.csv (4 facts, 6 dims) + planted_issues.json
2. python src/analyse.py
        ↓ uses src/kpi_lib.py (pandas) and src/kpi_sql.py (DuckDB SQL)
        ↓ reports/metrics.json, *.csv, *.png
3. pytest -q
4. streamlit run app.py              (Streamlit dashboard)
   powerbi/POWERBI_BUILD_GUIDE.md    (same model in Power BI Desktop)
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/generate_data.py`](src/generate_data.py) | Seeded generator: shifts, downtime events, speed losses, defects, targets, planted issues. |
| [`src/kpi_lib.py`](src/kpi_lib.py) | The KPI definitions in pandas: `kpis`, `kpis_by`, `rty`, `downtime_pareto`, `oee_losses_minutes`. |
| [`src/kpi_sql.py`](src/kpi_sql.py) | The same measures as a DuckDB SQL query over the CSVs. |
| [`src/analyse.py`](src/analyse.py) | Cross-check, control charts, rate scan, bootstrap, plots, `metrics.json`. |
| [`app.py`](app.py) | Streamlit + Altair dashboard. |
| [`powerbi/measures.dax`](powerbi/measures.dax) | DAX measures in display folders. |
| [`powerbi/POWERBI_BUILD_GUIDE.md`](powerbi/POWERBI_BUILD_GUIDE.md) | Load, model, measures, pages, expected values. |
| [`powerbi/theme.json`](powerbi/theme.json) | Optional Power BI colour theme. |
| [`tests/test_kpis.py`](tests/test_kpis.py) | Keys, count consistency, a hand-worked OEE example, pandas = SQL, Pareto, detection. |

## Key ideas in the code

**Generate events, then derive the shift record.** Downtime events are
drawn first (Poisson count, log-normal duration per reason); run time is
planned time minus their sum. That way the event table and the shift
table agree, as they should in a real MES.

**Weighted ratios everywhere.** `kpis()` sums minutes and counts over
whatever slice it is given and only then divides. The same function
serves the plant card, a line, a week or one shift.

**Phase I / phase II.** `weekly_pchart` and `weekly_downtime_chart`
compute limits from July–August only and then flag September–October.
Limits are never re-estimated on the period being monitored.

**Scoring against ground truth.** `main()` in `analyse.py` compares the
signals with `planted_issues.json` and counts everything else as a false
alarm.
