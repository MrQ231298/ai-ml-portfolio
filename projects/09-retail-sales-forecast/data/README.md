# Data folder — CSO Retail Sales Index (real public statistics)

**Real official statistics**, not synthetic and not company sales.

| Item | Detail |
| --- | --- |
| Producer | Central Statistics Office (CSO), Ireland |
| Tables | **RSM05** Retail Sales Index (base 2015 = 100), Jan 2015 – Jun 2024; **RSM08** Retail Sales Index (base 2021 = 100), Jan 2021 – Sep 2026 |
| API | `https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/{RSM05,RSM08}/CSV/1.0/en` |
| Measure used | *Retail Sales Index Value Unadjusted* (not seasonally adjusted) |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0). Please credit the CSO if you reuse the figures. |
| Fetched | 15 November 2026, 16:22 (Europe/Dublin) by `src/fetch_cso_retail.py` |

## Files

| File | Content |
| --- | --- |
| `RSM05_raw.csv`, `RSM08_raw.csv` | The raw CSO downloads, unchanged |
| `irish_retail_sales_index_monthly.csv` | 13 sector series × 141 months (Jan 2015 – Sep 2026), chain-linked to the RSM08 base |
| `source_metadata.json` | Fetch time, licence and the chain-link diagnostics per sector |

## Sectors

13 non-overlapping NACE groups (aggregates such as "All retail
businesses" were dropped so series do not double count): motor trades;
supermarkets / non-specialised food (4711); department stores; automotive
fuel; specialised food & drink; electrical goods; hardware, paints &
glass; furniture & lighting; clothing, footwear & textiles; books, news
& stationery; other retail; pharma, medical & cosmetics; bars.

## Chain-linking (and what went wrong first)

RSM05 and RSM08 overlap for 42 months (Jan 2021 – Jun 2024). The first
attempt rescaled RSM05 with **one** mean ratio per sector. The ratio
turned out to have a strong repeating calendar-month pattern (for bars
it is about 1.3 every June and about 4 every March) — single-ratio
coefficient of variation up to 36.8%. Using one ratio would have bent
the pre-2021 seasonal shape.

The final splice uses **one ratio per calendar month** (mean over the
overlapping years). After removing that monthly factor, the remaining
variation of the ratio is 0.9–4.7% for 10 of 13 sectors, but
7.4% (books), 10.9% (bars) and 11.1% (other retail) for three sectors,
whose pre-2021 history should be treated with caution. All diagnostics
are in `source_metadata.json`.

Index values are not euro amounts: they show the change in sales
relative to the base year for each sector.
