# Data folder — CSO / RTB average monthly rents (RIA02)

This folder holds **real, publicly published official statistics**, not
synthetic rents and not a confidential microdata file of individual
tenancies.

## Source and licence

- **Table:** RIA02 — RTB Average Monthly Rent Report
- **Producer:** Residential Tenancies Board (RTB)
- **Host:** Central Statistics Office (CSO) PxStat
- **API:** https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/RIA02/JSON-stat/2.0/en
- **Landing page:** https://data.gov.ie/dataset/ria02-rtb-average-monthly-rent-report
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Licence URL:** https://creativecommons.org/licenses/by/4.0/
- **Cube last updated (as reported by the API):** 2026-05-14T11:00:00.000Z
- **Fetched (Europe/Dublin):** 2026-11-14 20:32 GMT

You may reuse these figures under CC BY 4.0 provided you credit the RTB
and the CSO and link to the licence. This project's own code is under
the repository MIT licence; that does not replace the data licence.

## What the CSV contains

File: `ria02_average_monthly_rent.csv`

Each row is a published **average monthly rent (Euro)** for one
combination of year, location, property type and bedroom count.

Filters applied in `src/fetch_cso_ria02.py`:

- dropped "All bedrooms", "1 to 2 bed" and "1 to 3 bed" (overlapping bands)
- dropped "All property types"
- dropped **county-wide totals** (for example "Galway" as a whole),
  because they nest the towns and neighbourhoods already in the file

Location levels retained: cities, Dublin postal districts, and local
areas (towns and neighbourhoods). Years in this extract: 2008–2025.
Rows: 21,350.

These are official *averages*, not asking prices for a named property.
