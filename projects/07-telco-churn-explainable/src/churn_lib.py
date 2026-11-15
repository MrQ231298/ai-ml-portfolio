"""
Shared helpers for the churn project: loading, cleaning and the
feature list. Kept separate so the training script, the app and the
tests all use exactly the same preparation steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
BUNDLE_PATH = ROOT / "models" / "churn_model.joblib"

TARGET = "Churn"

# Protected / age-related attributes are NOT used as model inputs.
# They are kept only for the fairness slice audit.
AUDIT_COLS = ["gender", "SeniorCitizen"]

CAT_COLS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLS = CAT_COLS + NUM_COLS


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the IBM Telco CSV and fix the known data-quality issues.

    - `TotalCharges` is stored as text and is blank for 11 customers with
      tenure 0 (they have not been billed yet). Those blanks become 0.0.
    - `Churn` Yes/No becomes 1/0.
    - `SeniorCitizen` 0/1 becomes "No"/"Yes" for readable slice tables.
    """
    df = pd.read_csv(path)
    tc = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = tc.isna()
    # Every blank row should be a brand-new customer; fail loudly otherwise.
    assert (df.loc[blank, "tenure"] == 0).all(), "unexpected blank TotalCharges"
    df["TotalCharges"] = tc.fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    df["SeniorCitizen"] = np.where(df["SeniorCitizen"] == 1, "Yes", "No")
    return df


def net_benefit(y_true, target_mask, monthly_charges, cfg: dict) -> float:
    """Illustrative campaign value of contacting `target_mask` customers.

    For every contacted customer we pay `offer_cost`. A contacted customer
    who *would* have churned is saved with probability `save_rate`, and a
    saved customer is worth `value_months` x their monthly charge.
    Customers who would not have churned bring no extra value (the offer
    is wasted). Assumptions are deliberately simple and are configurable.
    """
    y = np.asarray(y_true)
    m = np.asarray(target_mask, dtype=bool)
    mc = np.asarray(monthly_charges, dtype=float)
    gain = (y[m] * cfg["save_rate"] * cfg["value_months"] * mc[m]).sum()
    cost = m.sum() * cfg["offer_cost"]
    return float(gain - cost)
