# Power BI build guide — Manufacturing KPI dashboard

Step-by-step instructions to rebuild the dashboard in **Power BI
Desktop** (Windows, free) from the files in this project. Allow about
60–90 minutes the first time.

> **Honest status.** The data model, the DAX and the page layout were
> designed and the measures were checked in Python and SQL, but this
> `.pbix` was **not** built or opened on the author's Linux development
> machine, because Power BI Desktop only runs on Windows. Build it once
> on Windows, compare your card values with the
> [expected values](#9-check-your-numbers) below, and save the
> resulting `.pbix` and a screenshot into this folder.

All data is **SYNTHETIC** (a fictional automotive electronics plant).

---

## 1. What you need

- Power BI Desktop (Microsoft Store or microsoft.com/power-bi), any 2025–2026 release.
- The 10 CSVs in [`../data/star_schema/`](../data/star_schema/).
- The measures in [`measures.dax`](measures.dax).
- Optional: the colour theme [`theme.json`](theme.json).

## 2. The model (star schema)

```
                 dim_date (date_key)        dim_shift (shift_key)
                        \                       /
 dim_product ---- fact_production ---- dim_line (line_key)
 (product_key)          |                  |
                  fact_target         fact_downtime --- dim_downtime_reason
                  (date, shift, line)     |
                                     fact_defects --- dim_defect
```

| Table | Grain | Rows |
| --- | --- | ---: |
| `fact_production` | one line × one shift × one day (one product per shift) | 1,128 |
| `fact_target` | same grain as production | 1,128 |
| `fact_downtime` | one unplanned stop event | 2,348 |
| `fact_defects` | line × shift × day × defect type | 2,697 |
| `dim_date` | one calendar day, 1 Jul – 31 Oct 2026 | 123 |
| `dim_line`, `dim_shift`, `dim_product`, `dim_downtime_reason`, `dim_defect` | lookup tables | 4 / 3 / 3 / 8 / 7 |

## 3. Load the data

1. **Home → Get data → Text/CSV**, pick `fact_production.csv`, click
   **Transform Data** (not Load).
2. Repeat for the other nine CSVs (or use **Get data → Folder** on
   `star_schema` and combine — single files are simpler for a first build).
3. In Power Query check the types:
   - `*_key` columns: **Whole number**
   - minutes columns and `duration_minutes`: **Decimal number**
   - counts: **Whole number**
   - `dim_date[date]`: **Date**; `dim_date[week_start]`: **Date**
4. Remove the `data_flag` column from the fact tables *only if* you add
   a visible "SYNTHETIC DATA" text box to every page (step 7). Otherwise
   keep it.
5. **Close & Apply**.

## 4. Relationships (Model view)

Create single-direction, many-to-one relationships, all **active**:

| From (many) | To (one) |
| --- | --- |
| `fact_production[date_key]` | `dim_date[date_key]` |
| `fact_production[shift_key]` | `dim_shift[shift_key]` |
| `fact_production[line_key]` | `dim_line[line_key]` |
| `fact_production[product_key]` | `dim_product[product_key]` |
| `fact_target[date_key]`, `[shift_key]`, `[line_key]` | the same three dims |
| `fact_downtime[date_key]`, `[shift_key]`, `[line_key]`, `[product_key]` | the four dims |
| `fact_downtime[reason_key]` | `dim_downtime_reason[reason_key]` |
| `fact_defects[date_key]`, `[shift_key]`, `[line_key]`, `[product_key]` | the four dims |
| `fact_defects[defect_key]` | `dim_defect[defect_key]` |

Do **not** relate fact tables to each other. Then:

- Select `dim_date` → **Table tools → Mark as date table** → column `date`.
- Hide every `*_key` column in the fact tables (right-click → Hide in report view).
- Sort `dim_date[month]` by `dim_date[month_num]` and `dim_shift[shift]` by `dim_shift[shift_key]`.

## 5. Measures

1. **Home → Enter data**, name the table `_Measures`, load it.
2. For each block in [`measures.dax`](measures.dax): **New measure**,
   paste the line(s) for one measure, press Enter.
3. Set formats: every measure ending in `%` → **Percentage, 1 decimal**;
   minutes and counts → **Whole number, thousands separator**.
4. Put measures in display folders (Model view → Properties → Display
   folder) using the folder names in the comments (`1 Base`, `2 OEE`, …).
5. Delete the dummy column of `_Measures`; it becomes a measure table.

## 6. Report pages

Canvas 16:9. Slicers on every page (sync them via **View → Sync slicers**):
`dim_date[date]` (between), `dim_line[line]`, `dim_shift[shift]`,
`dim_product[product_code]`.

### Page 1 — Plant overview

| Visual | Fields |
| --- | --- |
| 5 cards | `OEE %`, `Availability %`, `Performance %`, `Quality %`, `Output vs Target %` |
| Line chart | X `dim_date[week_start]`; Y `OEE %`; Legend `dim_line[line]`; add constant line 85% |
| Clustered bar | Y `dim_line[line]`; X `Good Count` and `Target Units` |
| Matrix | Rows `dim_line[line]`; Columns `dim_shift[shift]`; Values `OEE %`, conditional background colour from field `OEE Colour` |

### Page 2 — Downtime

| Visual | Fields |
| --- | --- |
| Line-and-clustered-column (Pareto) | X `dim_downtime_reason[reason]` sorted by `Downtime Min (events)` desc; column `Downtime Hours`; line `Downtime Cumulative %` (secondary axis 0–100%) |
| Cards | `Downtime Hours`, `Downtime Events`, `MTTR Min`, `Events per Shift` |
| Matrix | Rows `dim_downtime_reason[reason]`; Columns `dim_line[line]`; Values `Events per Shift` with a colour scale |
| Stacked bar | Y `dim_line[line]`; X `Availability Loss Min`, `Performance Loss Min`, `Quality Loss Min`, `Fully Productive Min` |

### Page 3 — Quality

| Visual | Fields |
| --- | --- |
| Cards | `FPY %`, `RTY %`, `Yield After Rework %`, `Scrap Rate %` |
| Line chart | X `dim_date[week_start]`; Y `Defects per 1k Units`; Legend `dim_defect[defect]`; filter `dim_line[line]` = SMT-1 to see the September solder-bridge spike |
| Treemap | Group `dim_defect[defect]`; Values `Defects` |

### Page 4 — Shift comparison

Clustered column: X `dim_shift[shift]`; Y `Availability %`,
`Performance %`, `Quality %`. The Night shift's lower performance
should be obvious.

## 7. Finishing

- Add a text box on every page: **"SYNTHETIC example data — fictional plant."**
- Apply `theme.json` (**View → Themes → Browse for themes**).
- Title format: `Plant KPI dashboard — {page}`.
- Save as `powerbi/manufacturing_kpi.pbix` and export a PNG per page.

## 8. Common mistakes this guide avoids

- **Averaging OEE.** Never put `AVERAGE` of a per-shift OEE column on a
  card. All measures here are sums divided by sums.
- **Bidirectional filters.** Not needed; they create ambiguous paths
  between the four fact tables.
- **Using the downtime fact for availability.** Availability uses
  `fact_production[run_minutes]` so it matches the shift record;
  the event table is for the Pareto. The two agree within rounding
  (events are stored to 0.1 min).

## 9. Check your numbers

With **no slicers applied** the cards must match these values from
`reports/metrics.json` (produced by `python src/analyse.py`, 17 November
2026, Europe/Dublin):

| Measure | Expected |
| --- | ---: |
| Availability % | 89.4% |
| Performance % | 89.8% |
| Quality % / FPY % | 98.7% |
| OEE % | 79.2% |
| OEE % (time based) | 79.3% |
| Output vs Target % | 93.5% |
| RTY % | 96.7% |
| Yield After Rework % | 99.7% |
| Downtime Hours (events) | 896.7 |
| Top downtime reason | Changeover overrun, 30.7% of minutes |

Filter **Line = SMT-2**: OEE % should read **74.3%** (the lowest line).
If a value differs, the usual cause is a missing relationship or a
column loaded as text.
