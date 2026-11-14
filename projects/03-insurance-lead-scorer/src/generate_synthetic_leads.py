"""
Generate a SYNTHETIC commercial-insurance lead table for conversion scoring.

IMPORTANT
---------
Every row is invented example data. It is not taken from any broker CRM,
insurer, or real business. The filename, the `data_source` column, and
the data README are all labelled SYNTHETIC.

The conversion labels are produced by a known, noisy process so the
modelling scripts have something to learn — they are not observed
outcomes from real quotes.

Usage
-----
    python src/generate_synthetic_leads.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 3200

INDUSTRIES = [
    "construction",
    "hospitality",
    "retail",
    "professional_services",
    "transport_logistics",
    "agriculture",
    "manufacturing",
    "healthcare",
    "it_telecoms",
    "other",
]
# Slightly Ireland-weighted industry mix for a commercial desk.
INDUSTRY_P = [0.16, 0.10, 0.13, 0.14, 0.09, 0.08, 0.08, 0.06, 0.09, 0.07]

REVENUE_BANDS = [
    "under_100k",
    "100k_250k",
    "250k_500k",
    "500k_1m",
    "1m_5m",
    "over_5m",
]
REVENUE_P = [0.22, 0.24, 0.20, 0.16, 0.13, 0.05]

COVERS = [
    "public_liability",
    "employers_liability",
    "commercial_property",
    "motor_fleet",
    "professional_indemnity",
    "combined_package",
    "cyber",
]
COVER_P = [0.18, 0.14, 0.14, 0.12, 0.13, 0.22, 0.07]

SOURCES = [
    "broker_referral",
    "existing_client",
    "web_form",
    "inbound_call",
    "comparison_site",
    "cold_outbound",
]
SOURCE_P = [0.16, 0.10, 0.28, 0.18, 0.14, 0.14]

REGIONS = [
    "dublin",
    "leinster_ex_dublin",
    "munster",
    "connacht",
    "ulster_roi",
]
REGION_P = [0.32, 0.22, 0.22, 0.14, 0.10]


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data",
        "csv": root / "data" / "SYNTHETIC_commercial_insurance_leads.csv",
        "readme": root / "data" / "README.md",
        "meta": root / "data" / "source_metadata.json",
    }


def conversion_logit(row: dict, rng: np.random.Generator) -> float:
    """
    Latent log-odds of conversion. This is the *data-generating* process,
    not the model we will train. It is written out so the critical
    evaluation can say honestly what signal was planted in the file.
    """
    # Base rate around 10–12% before adjustments (logit of 0.12 ≈ -2.0).
    z = -2.05

    source_fx = {
        "broker_referral": 0.95,
        "existing_client": 1.15,
        "inbound_call": 0.35,
        "web_form": 0.10,
        "comparison_site": -0.25,
        "cold_outbound": -1.15,
    }
    z += source_fx[row["source"]]

    # Fast follow-up helps; very slow follow-up hurts (hours).
    # Effect is non-linear: the first day matters most.
    z += 0.85 * np.exp(-row["response_time_hours"] / 12.0) - 0.35

    # Mid-size firms convert a little better than micro or very large.
    emp = row["n_employees"]
    if 8 <= emp <= 80:
        z += 0.35
    elif emp < 3:
        z -= 0.25
    elif emp > 200:
        z -= 0.15

    rev_fx = {
        "under_100k": -0.20,
        "100k_250k": 0.00,
        "250k_500k": 0.15,
        "500k_1m": 0.25,
        "1m_5m": 0.15,
        "over_5m": -0.05,
    }
    z += rev_fx[row["revenue_band"]]

    cover_fx = {
        "combined_package": 0.35,
        "public_liability": 0.10,
        "employers_liability": 0.15,
        "commercial_property": 0.05,
        "professional_indemnity": 0.20,
        "motor_fleet": 0.00,
        "cyber": -0.15,  # newer product, more browsing than buying in this fiction
    }
    z += cover_fx[row["coverage_type"]]

    # Construction + public liability, or professional services + PI,
    # are slightly more "ready to buy" in this invented world.
    if row["industry"] == "construction" and row["coverage_type"] in {
        "public_liability",
        "employers_liability",
        "combined_package",
    }:
        z += 0.25
    if row["industry"] == "professional_services" and row["coverage_type"] == "professional_indemnity":
        z += 0.30

    if row["has_claims_last_3y"]:
        z -= 0.40
    if row["years_trading"] >= 10:
        z += 0.15
    elif row["years_trading"] < 2:
        z -= 0.20

    # Region has only a *small* effect in the generative process.
    # It is included so we can test whether the fitted model exaggerates it.
    region_fx = {
        "dublin": 0.08,
        "leinster_ex_dublin": 0.04,
        "munster": 0.00,
        "connacht": -0.06,
        "ulster_roi": -0.08,
    }
    z += region_fx[row["region"]]

    # Idiosyncratic noise so the label is not a deterministic function.
    z += rng.normal(0, 0.55)
    return float(z)


def main() -> None:
    paths = project_paths()
    paths["data"].mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    industry = rng.choice(INDUSTRIES, size=N_ROWS, p=INDUSTRY_P)
    # Employees: mostly small Irish commercial clients, a long tail.
    n_employees = np.clip(rng.lognormal(mean=2.2, sigma=1.05, size=N_ROWS).astype(int), 1, 400)
    revenue_band = rng.choice(REVENUE_BANDS, size=N_ROWS, p=REVENUE_P)
    coverage_type = rng.choice(COVERS, size=N_ROWS, p=COVER_P)
    source = rng.choice(SOURCES, size=N_ROWS, p=SOURCE_P)
    region = rng.choice(REGIONS, size=N_ROWS, p=REGION_P)
    years_trading = np.clip(rng.gamma(4.0, 2.5, size=N_ROWS).astype(int), 0, 60)
    n_locations = np.clip(rng.poisson(0.4, size=N_ROWS) + 1, 1, 12)
    has_claims = rng.random(N_ROWS) < 0.18
    # Response time: many same-day, a long tail of slow follow-up.
    response_time_hours = np.clip(rng.lognormal(mean=2.0, sigma=1.1, size=N_ROWS), 0.2, 240)

    records = []
    for i in range(N_ROWS):
        row = {
            "lead_id": f"SYN-L-{i + 1:04d}",
            "industry": industry[i],
            "n_employees": int(n_employees[i]),
            "revenue_band": revenue_band[i],
            "coverage_type": coverage_type[i],
            "source": source[i],
            "region": region[i],
            "years_trading": int(years_trading[i]),
            "n_locations": int(n_locations[i]),
            "has_claims_last_3y": bool(has_claims[i]),
            "response_time_hours": round(float(response_time_hours[i]), 1),
            "data_source": "SYNTHETIC_EXAMPLE_DATA",
        }
        z = conversion_logit(row, rng)
        p = 1.0 / (1.0 + np.exp(-z))
        row["converted"] = int(rng.random() < p)
        records.append(row)

    df = pd.DataFrame.from_records(records)
    # Shuffle so the file is not ordered by generation index.
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df.to_csv(paths["csv"], index=False)

    rate = float(df["converted"].mean())
    meta = {
        "data_source": "SYNTHETIC_EXAMPLE_DATA",
        "n_rows": int(len(df)),
        "n_converted": int(df["converted"].sum()),
        "conversion_rate": round(rate, 4),
        "random_seed": RANDOM_SEED,
        "note": (
            "Invented commercial-insurance leads for a portfolio demo. "
            "Not extracted from any real CRM or insurer."
        ),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["readme"].write_text(
        f"""# Data folder — SYNTHETIC example data only

The file `SYNTHETIC_commercial_insurance_leads.csv` is **entirely invented**.

- It is not extracted from any broker, insurer, or business CRM.
- Names, lead IDs and outcomes are fictional.
- The `data_source` column is `SYNTHETIC_EXAMPLE_DATA` on every row.
- Do not mix this file with live prospect data.

{len(df):,} rows. Conversion rate in this draw: {rate:.1%}.
The labels come from a noisy process documented in
`src/generate_synthetic_leads.py` (source, response time, size, cover
type and a few interactions). Region was given only a small effect so
we can later ask whether a fitted model over-uses it.
""",
        encoding="utf-8",
    )
    print(f"Wrote {len(df)} SYNTHETIC leads to {paths['csv']}")
    print(f"  converted: {int(df['converted'].sum())}  ({rate:.1%})")
    print("  by source:")
    print(
        df.groupby("source")["converted"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
