"""
Fetch Irish Retail Sales Index tables from the CSO PxStat API and build
one chain-linked monthly series per retail sector.

Tables (CSO, CC BY 4.0):
- RSM05  Retail Sales Index, base 2015=100, Jan 2015 - Jun 2024
- RSM08  Retail Sales Index, base 2021=100, Jan 2021 - latest

Measure: "Retail Sales Index Value Unadjusted" (not seasonally adjusted,
because the forecasting models must learn the seasonality themselves).

Chain-linking: for each sector the older RSM05 series is rescaled onto
the RSM08 base. A first attempt used a single mean ratio RSM08/RSM05
over the overlap (Jan 2021 - Jun 2024), but the ratio turned out to
follow a strong, *repeating* calendar-month pattern (for bars it ranges
from about 1.3 in June to about 4 in March, every year). A single ratio
would therefore distort the seasonal shape of the pre-2021 data. The
splice instead uses one ratio per calendar month (mean over the 3-4
overlapping years). Both diagnostics are saved: the coefficient of
variation (CV) of the raw ratio, and the CV of the ratio after removing
the calendar-month factor. From Jan 2021 onward RSM08 values are used
unchanged.

    python src/fetch_cso_retail.py
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/{t}/CSV/1.0/en"
MEASURE = "Retail Sales Index Value Unadjusted"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Non-overlapping sector groups (aggregates such as "All retail businesses"
# are excluded so the series do not double count each other).
SECTORS = {
    "Motor trades (45)": "Motor trades",
    "Retail sale in non-specialised stores with food, beverages or tobacco predominating (4711)": "Supermarkets / non-specialised food",
    "Department stores (4719)": "Department stores",
    "Retail sale of automotive fuel (4730)": "Automotive fuel",
    "Retail sale of food, beverages and tobacco in specialised stores (4721 to 4729)": "Specialised food & drink",
    "Retail sale of electrical goods (4741 to 4743,4754)": "Electrical goods",
    "Retail sale of hardware, paints and glass (4752)": "Hardware, paints & glass",
    "Retail sale of furniture and lighting (4759)": "Furniture & lighting",
    "Retail sale of textiles, clothing and footwear (4751,4771,4772)": "Clothing, footwear & textiles",
    "Retail sale of books, newspapers and stationery (4761,4762)": "Books, news & stationery",
    "Other retail sales (4753,4763 to 4765,4776 to 4778)": "Other retail",
    "Retail sale of pharmaceutical, medical and cosmetic articles (4773 to 4775)": "Pharma, medical & cosmetics",
    "Bars (5630)": "Bars",
}


def fetch(table: str) -> pd.DataFrame:
    r = requests.get(API.format(t=table), timeout=120)
    r.raise_for_status()
    (DATA / f"{table}_raw.csv").write_bytes(r.content)
    df = pd.read_csv(io.BytesIO(r.content))
    df = df[(df["Statistic Label"] == MEASURE) & (df["NACE Group"].isin(SECTORS))]
    df = df.assign(
        date=pd.to_datetime(df["TLIST(M1)"].astype(str), format="%Y%m"),
        series=df["NACE Group"].map(SECTORS),
    )
    return df[["series", "date", "VALUE"]].rename(columns={"VALUE": "value"})


def main() -> None:
    DATA.mkdir(exist_ok=True)
    old, new = fetch("RSM05"), fetch("RSM08")
    rows, splice = [], {}
    for s in SECTORS.values():
        o = old[old.series == s].set_index("date")["value"]
        n = new[new.series == s].set_index("date")["value"]
        overlap = o.index.intersection(n.index)
        ratio = n[overlap] / o[overlap]
        by_month = ratio.groupby(ratio.index.month).mean()
        resid = ratio / ratio.index.month.map(by_month).to_numpy()
        splice[s] = {
            "overlap_months": int(len(overlap)),
            "single_ratio_cv_pct": round(float(ratio.std() / ratio.mean() * 100), 2),
            "month_ratio_residual_cv_pct": round(float(resid.std() / resid.mean() * 100), 2),
            "month_ratios": {int(k): round(float(v), 4) for k, v in by_month.items()},
        }
        early = o[o.index < n.index.min()]
        early = early * early.index.month.map(by_month).to_numpy()
        full = pd.concat([early, n]).sort_index()
        rows.append(pd.DataFrame({"series": s, "date": full.index, "value": full.values.round(2)}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA / "irish_retail_sales_index_monthly.csv", index=False)
    meta = {
        "tables": ["RSM05", "RSM08"],
        "measure": MEASURE,
        "publisher": "Central Statistics Office (CSO), Ireland",
        "licence": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "fetched_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "date_range": [str(out.date.min().date()), str(out.date.max().date())],
        "n_series": int(out.series.nunique()),
        "rows": int(len(out)),
        "chain_link_check": splice,
    }
    (DATA / "source_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
