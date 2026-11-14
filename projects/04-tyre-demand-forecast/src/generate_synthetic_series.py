"""
Generate a SYNTHETIC weekly tyre-sales series for an Irish trade desk.

IMPORTANT
---------
Every observation is invented. It is not taken from any distributor's
stock system or from official NCT volumes. The filename, the
`data_source` column and the data README are labelled SYNTHETIC.

Seasonal bumps are *inspired by* real-world timing (winter-tyre
changeover in late autumn, a quieter mid-summer, a spring uptick
when cars come out of winter and NCT bookings rise) but the numbers
themselves are made up.

Usage
-----
    python src/generate_synthetic_series.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
# Complete ISO weeks from early 2019 through the end of 2025.
START = "2019-01-07"  # a Monday
END = "2025-12-29"


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    return {
        "root": root,
        "data": data,
        "csv": data / "SYNTHETIC_weekly_tyre_sales.csv",
        "readme": data / "README.md",
        "meta": data / "source_metadata.json",
    }


def week_of_year_bump(woy: np.ndarray, centre: float, width: float, height: float) -> np.ndarray:
    """Gaussian bump on the 1–52 week circle (so week 1 sits next to 52)."""
    d = np.minimum(np.abs(woy - centre), 52 - np.abs(woy - centre))
    return height * np.exp(-0.5 * (d / width) ** 2)


def generate(idx: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(len(idx), dtype=float)
    woy = idx.isocalendar().week.to_numpy().astype(float)

    # Level and a gentle upward trend (~3% a year at weekly scale).
    level = 820.0
    trend = 0.45 * t

    # Winter-tyre changeover: late October to early January.
    winter = week_of_year_bump(woy, 47.0, 4.0, 230.0) + week_of_year_bump(woy, 2.0, 3.0, 90.0)
    # Spring: post-winter replacements and the busy NCT booking period.
    nct_spring = week_of_year_bump(woy, 16.0, 5.0, 150.0)
    # Early autumn: back-to-school / pre-winter check.
    autumn = week_of_year_bump(woy, 36.0, 3.5, 70.0)
    # Mid-summer lull (holidays).
    summer_dip = week_of_year_bump(woy, 31.0, 2.0, -80.0)

    # A short, invented 2020 disruption (not a real company's sales).
    covid = np.zeros_like(t)
    covid_mask = (idx >= "2020-03-16") & (idx <= "2020-06-08")
    covid[covid_mask] = -220.0

    noise = rng.normal(0, 42.0, size=len(idx))
    # A few one-week promo / weather spikes.
    spikes = np.zeros_like(t)
    spike_weeks = rng.choice(np.arange(len(idx)), size=14, replace=False)
    spikes[spike_weeks] = rng.normal(90, 25, size=14)

    y = level + trend + winter + nct_spring + autumn + summer_dip + covid + noise + spikes
    return np.clip(y, 250.0, None).round(0)


def main() -> None:
    paths = project_paths()
    paths["data"].mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = pd.date_range(START, END, freq="W-MON")
    units = generate(idx, rng)
    df = pd.DataFrame(
        {
            "week_start": idx,
            "year": idx.isocalendar().year.astype(int),
            "week_of_year": idx.isocalendar().week.astype(int),
            "units_sold": units.astype(int),
            "data_source": "SYNTHETIC_EXAMPLE_DATA",
        }
    )
    df.to_csv(paths["csv"], index=False)
    meta = {
        "data_source": "SYNTHETIC_EXAMPLE_DATA",
        "n_weeks": int(len(df)),
        "start": str(df["week_start"].min().date()),
        "end": str(df["week_start"].max().date()),
        "mean_units": float(df["units_sold"].mean()),
        "random_seed": RANDOM_SEED,
        "note": "Invented weekly unit sales for a portfolio demo. Not a real depot extract.",
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["readme"].write_text(
        f"""# Data folder — SYNTHETIC example data only

`SYNTHETIC_weekly_tyre_sales.csv` is **entirely invented**.

- It is not extracted from any tyre distributor's sales system.
- It is not official NCT, CSO or supplier data.
- The `data_source` column is `SYNTHETIC_EXAMPLE_DATA` on every row.

{len(df)} weekly observations, {df['week_start'].min().date()} to
{df['week_start'].max().date()} (weeks starting Monday). The series is
built from a trend, a winter-tyre bump, a spring NCT-inspired bump,
a summer lull, a short 2020 dip, and noise — see
`src/generate_synthetic_series.py`.
""",
        encoding="utf-8",
    )
    print(f"Wrote {len(df)} SYNTHETIC weeks to {paths['csv']}")
    print(df.groupby("year")["units_sold"].agg(["count", "mean", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
