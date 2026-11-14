"""
Download and tidy the CSO / RTB Average Monthly Rent table (RIA02).

Source
------
Residential Tenancies Board (RTB) statistics hosted by the Central
Statistics Office (CSO) on PxStat, table RIA02 — "RTB Average Monthly
Rent Report". Accessed through the public PxStat REST API.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/

This script does not invent any rent figures. It only downloads the
official cube, drops aggregate categories (so we do not train on
"All bedrooms" / "All property types" / county-wide totals), and writes
a tidy CSV plus a metadata file that records the URL, fetch time and
licence.

Usage
-----
    python src/fetch_cso_ria02.py
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

API_URL = (
    "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API."
    "ReadDataset/RIA02/JSON-stat/2.0/en"
)
LANDING_PAGE = "https://data.gov.ie/dataset/ria02-rtb-average-monthly-rent-report"
LICENCE_NAME = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"

DUBLIN_TZ = ZoneInfo("Europe/Dublin")

# 26 traditional counties as they appear in RIA02 (no "State" total).
COUNTIES = {
    "Carlow",
    "Cavan",
    "Clare",
    "Cork",
    "Donegal",
    "Dublin",
    "Galway",
    "Kerry",
    "Kildare",
    "Kilkenny",
    "Laois",
    "Leitrim",
    "Limerick",
    "Longford",
    "Louth",
    "Mayo",
    "Meath",
    "Monaghan",
    "Offaly",
    "Roscommon",
    "Sligo",
    "Tipperary",
    "Waterford",
    "Westmeath",
    "Wexford",
    "Wicklow",
}

CITY_TO_COUNTY = {
    "Cork City": "Cork",
    "Galway City": "Galway",
    "Limerick City": "Limerick",
    "Waterford City": "Waterford",
    "Kilkenny City": "Kilkenny",
}

# Keep only specific bedroom and property-type categories — not totals
# or overlapping bands such as "1 to 3 bed".
BED_KEEP = {
    "01": ("One bed", 1),
    "02": ("Two bed", 2),
    "03": ("Three bed", 3),
    "08": ("Four plus bed", 4),
}
TYPE_KEEP = {
    "01": "Detached house",
    "02": "Semi detached house",
    "03": "Terrace house",
    "04": "Apartment",
    "05": "Other flats",
}


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    return {
        "root": root,
        "data": data,
        "raw": data / "ria02_raw.json",
        "csv": data / "ria02_average_monthly_rent.csv",
        "meta": data / "source_metadata.json",
        "source_md": data / "README.md",
    }


def is_dublin_postal(label: str) -> bool:
    return bool(re.fullmatch(r"Dublin \d+[A-Z]?", label))


def county_of(location: str) -> str:
    """Map a location label to its county. Raise if we cannot classify it."""
    if location in COUNTIES:
        return location
    if location in CITY_TO_COUNTY:
        return CITY_TO_COUNTY[location]
    if is_dublin_postal(location) or location.startswith("Dublin "):
        return "Dublin"
    if location.endswith(" Town"):
        base = location[: -len(" Town")]
        if base in COUNTIES:
            return base
    if "," in location:
        suffix = location.split(",")[-1].strip()
        if suffix in COUNTIES:
            return suffix
        if is_dublin_postal(suffix) or suffix.startswith("Dublin"):
            return "Dublin"
    raise ValueError(f"Cannot derive county for location {location!r}")


def location_level(location: str) -> str:
    if location in COUNTIES:
        return "county"
    if location in CITY_TO_COUNTY:
        return "city"
    if is_dublin_postal(location):
        return "dublin_postal"
    return "local"


def download_cube(url: str, dest: Path) -> bytes:
    """Fetch the JSON-stat cube and cache the raw bytes on disk."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ai-ml-portfolio-rent-predictor/1.0 (educational; CC BY attribution)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
        status = getattr(resp, "status", None)
        if status not in (None, 200):
            raise RuntimeError(f"CSO API returned HTTP {status}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return payload


def _axis_codes_and_labels(dimension: dict, size: int) -> list[tuple[str, str]]:
    cat = dimension["category"]
    index = cat["index"]
    if isinstance(index, dict):
        codes = [None] * size
        for code, pos in index.items():
            codes[int(pos)] = code
    else:
        codes = list(index)
    labels = cat.get("label", {})
    return [(code, labels.get(code, code)) for code in codes]


def cube_to_frame(cube: dict) -> tuple[pd.DataFrame, dict]:
    """Flatten the JSON-stat cube and drop aggregate categories / county totals."""
    dim_ids = cube["id"]
    sizes = cube["size"]
    dimensions = cube["dimension"]
    axes = [
        _axis_codes_and_labels(dimensions[dim_id], size)
        for dim_id, size in zip(dim_ids, sizes)
    ]
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]

    values = cube["value"]
    records: list[dict] = []
    n_raw_non_null = 0
    for i, value in enumerate(values):
        if value is None:
            continue
        n_raw_non_null += 1
        rem = i
        decoded = []
        for stride, axis in zip(strides, axes):
            idx = rem // stride
            rem %= stride
            decoded.append(axis[idx])

        # Order in this cube: STATISTIC, Year, Bedrooms, Property Type, Location
        year = int(decoded[1][0])
        bed_code, _bed_label = decoded[2]
        type_code, _type_label = decoded[3]
        _loc_code, location = decoded[4]

        if bed_code not in BED_KEEP or type_code not in TYPE_KEEP:
            continue
        bed_label, n_bedrooms = BED_KEEP[bed_code]
        level = location_level(location)
        # County-wide totals nest every town in that county — drop them so
        # we do not train on two aggregation levels at once.
        if level == "county":
            continue
        records.append(
            {
                "year": year,
                "location": location,
                "county": county_of(location),
                "location_level": level,
                "property_type": TYPE_KEEP[type_code],
                "bedrooms": bed_label,
                "n_bedrooms": n_bedrooms,
                "avg_monthly_rent_eur": float(value),
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(
        ["year", "county", "location", "property_type", "n_bedrooms"]
    ).reset_index(drop=True)
    extras = {
        "cube_label": cube.get("label"),
        "cube_updated": cube.get("updated"),
        "n_cells_in_cube": len(values),
        "n_non_null_in_cube": n_raw_non_null,
        "n_rows_after_filters": int(len(df)),
    }
    return df, extras


def write_data_readme(path: Path, meta: dict, df: pd.DataFrame) -> None:
    years = f"{int(df['year'].min())}–{int(df['year'].max())}"
    path.write_text(
        f"""# Data folder — CSO / RTB average monthly rents (RIA02)

This folder holds **real, publicly published official statistics**, not
synthetic rents and not a confidential microdata file of individual
tenancies.

## Source and licence

- **Table:** RIA02 — RTB Average Monthly Rent Report
- **Producer:** Residential Tenancies Board (RTB)
- **Host:** Central Statistics Office (CSO) PxStat
- **API:** {API_URL}
- **Landing page:** {LANDING_PAGE}
- **Licence:** {LICENCE_NAME}
- **Licence URL:** {LICENCE_URL}
- **Cube last updated (as reported by the API):** {meta.get("cube_updated")}
- **Fetched (Europe/Dublin):** {meta.get("fetched_at_dublin")}

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
areas (towns and neighbourhoods). Years in this extract: {years}.
Rows: {len(df):,}.

These are official *averages*, not asking prices for a named property.
""",
        encoding="utf-8",
    )


def main() -> None:
    paths = project_paths()
    paths["data"].mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(DUBLIN_TZ)
    print(f"Downloading RIA02 from {API_URL}")
    payload = download_cube(API_URL, paths["raw"])
    sha256 = hashlib.sha256(payload).hexdigest()
    print(f"  saved {paths['raw'].name}  ({len(payload):,} bytes, sha256={sha256[:12]}…)")

    cube = json.loads(payload.decode("utf-8"))
    df, extras = cube_to_frame(cube)
    if df.empty:
        raise RuntimeError("No rows left after filters — check the cube layout.")
    unknown = df[df["county"].isna()]
    if len(unknown):
        raise RuntimeError(f"Unclassified locations: {sorted(unknown['location'].unique())}")

    df.to_csv(paths["csv"], index=False)
    print(
        f"  wrote {paths['csv'].name}: {len(df):,} rows, "
        f"{df['location'].nunique()} locations, years "
        f"{int(df['year'].min())}–{int(df['year'].max())}"
    )
    print("  location levels:", df["location_level"].value_counts().to_dict())

    meta = {
        "table_id": "RIA02",
        "table_label": extras.get("cube_label"),
        "producer": "Residential Tenancies Board (RTB)",
        "host": "Central Statistics Office (CSO)",
        "api_url": API_URL,
        "landing_page": LANDING_PAGE,
        "licence": LICENCE_NAME,
        "licence_url": LICENCE_URL,
        "attribution": (
            "Based on the RTB Average Monthly Rent Report (table RIA02), "
            "hosted by the Central Statistics Office and licensed under "
            "CC BY 4.0."
        ),
        "cube_updated": extras.get("cube_updated"),
        "fetched_at_dublin": fetched_at.strftime("%Y-%m-%d %H:%M %Z"),
        "fetched_at_utc": fetched_at.astimezone(ZoneInfo("UTC")).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "raw_file": paths["raw"].name,
        "raw_sha256": sha256,
        "raw_bytes": len(payload),
        "n_cells_in_cube": extras.get("n_cells_in_cube"),
        "n_non_null_in_cube": extras.get("n_non_null_in_cube"),
        "n_rows_after_filters": extras.get("n_rows_after_filters"),
        "filters": {
            "dropped_bedroom_aggregates": True,
            "dropped_property_type_aggregates": True,
            "dropped_county_totals": True,
        },
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "n_locations": int(df["location"].nunique()),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_data_readme(paths["source_md"], meta, df)
    print(f"  metadata: {paths['meta']}")


if __name__ == "__main__":
    main()
