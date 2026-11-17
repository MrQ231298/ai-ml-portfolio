"""
Clean: standardise formats, remove exact duplicates, and split each table
into (clean rows, rejected rows with a reason). Pure functions, no I/O,
so every rule is unit-tested in tests/test_clean.py.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
COUNTRY_MAP = {"ie": "IE", "ireland": "IE", "rep. of ireland": "IE", "irl": "IE", "republic of ireland": "IE",
               "gb": "GB", "uk": "GB", "u.k.": "GB", "united kingdom": "GB",
               "northern ireland": "NI", "n. ireland": "NI", "gb-nir": "NI", "ni": "NI"}
STATUS = {"completed", "cancelled"}


def parse_money(s) -> float:
    """'€12.50', 'EUR 12.50', '12,50', '12.50' -> 12.5 ; blanks -> NaN."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or pd.isna(s) or str(s).strip() == "":
        return float("nan")
    t = str(s).strip().replace("€", "").replace("EUR", "").strip()
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    t = t.replace(",", "")
    return round(float(t), 2)


def parse_date(s) -> pd.Timestamp:
    """ISO, Irish day-first (dd/mm/yyyy) or 'Jan 05 2026'. Never month-first."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%b %d %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.NaT


def normalise_country(s) -> str | None:
    return COUNTRY_MAP.get(str(s).strip().lower()) if not pd.isna(s) else None


def normalise_phone(s) -> str | None:
    """Irish mobiles to E.164: '0871234567' / '+353 87 123 4567' -> '+353871234567'."""
    if pd.isna(s):
        return None
    d = re.sub(r"[^\d+]", "", str(s))
    if d.startswith("+"):
        return d
    if d.startswith("0"):
        return "+353" + d[1:]
    return None


def valid_email(s) -> bool:
    return bool(EMAIL_RE.match(str(s).strip())) if not pd.isna(s) else False


def _split(df: pd.DataFrame, bad: pd.Series, reason: pd.Series | str, table: str, key: pd.Series):
    rej = pd.DataFrame({"table_name": table, "record_key": key[bad].values,
                        "reason": (reason[bad].values if isinstance(reason, pd.Series) else reason),
                        "payload": df[bad].to_json(orient="records", lines=True, date_format="iso").splitlines() if bad.any() else []})
    return df[~bad].copy(), rej


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df)


def clean_customers(raw: pd.DataFrame):
    df, n_dup = dedupe(raw)
    out = pd.DataFrame({
        "customer_id": df["customer_id"].str.strip(),
        "first_name": df["first_name"].str.strip().str.title(),
        "last_name": df["last_name"].str.strip(),
        "email": df["email"].str.strip().str.lower(),
        "phone_e164": df["phone"].map(normalise_phone),
        "country_code": df["country"].map(normalise_country),
        "county": df["county"].str.strip(),
        "signup_date": df["signup_date"].map(parse_date),
    })
    out["email_valid"] = out["email"].map(valid_email)
    out.loc[~out["email_valid"], "email"] = None            # keep the customer, drop the bad value
    # same customer_id twice with different content: keep the last row in the file
    n_conflict = int(out.duplicated("customer_id").sum())
    out = out.drop_duplicates("customer_id", keep="last")
    bad = out["customer_id"].isna() | out["country_code"].isna() | out["signup_date"].isna()
    reason = pd.Series(np.select([out["customer_id"].isna(), out["country_code"].isna()],
                                 ["missing_key", "unknown_country"], "bad_signup_date"), index=out.index)
    clean, rej = _split(out, bad, reason, "customers", out["customer_id"])
    return clean, rej, {"exact_duplicates_removed": n_dup, "key_conflicts_resolved": n_conflict,
                        "invalid_emails_nulled": int((~out["email_valid"]).sum())}


def clean_products(raw: pd.DataFrame):
    df, n_dup = dedupe(raw)
    out = pd.DataFrame({
        "product_id": df["product_id"].str.strip(),
        "product_name": df["product_name"].str.strip(),
        "category": df["category"].str.strip().str.title().str.replace(" And ", " & "),
        "list_price_eur": df["list_price"].map(parse_money),
        "unit_cost_eur": pd.to_numeric(df["unit_cost_eur"], errors="coerce"),
    })
    bad = out["list_price_eur"].isna() | (out["list_price_eur"] <= 0)
    clean, rej = _split(out, bad, "missing_or_invalid_price", "products", out["product_id"])
    return clean, rej, {"exact_duplicates_removed": n_dup}


def clean_fx(raw: pd.DataFrame) -> pd.DataFrame:
    fx = raw.copy()
    fx["eur_per_unit"] = pd.to_numeric(fx["eur_per_unit"])
    return fx[["month", "currency", "eur_per_unit"]]


def clean_orders(raw: pd.DataFrame, extract_date: pd.Timestamp):
    df, n_dup = dedupe(raw)
    out = pd.DataFrame({
        "order_id": df["order_id"].str.strip(),
        "customer_id": df["customer_id"].str.strip(),
        "order_ts": df["order_ts"].map(parse_date),
        "channel": df["channel"].str.strip().str.lower(),
        "status": df["status"].str.strip().str.lower(),
        "currency": df["currency"].str.strip().str.upper(),
    })
    reason = pd.Series(np.select(
        [out["order_ts"].isna(), out["order_ts"] > extract_date, ~out["status"].isin(STATUS), ~out["currency"].isin(["EUR", "GBP"])],
        ["bad_timestamp", "future_date", "unknown_status", "unknown_currency"], ""), index=out.index)
    clean, rej = _split(out, reason != "", reason, "orders", out["order_id"])
    return clean, rej, {"exact_duplicates_removed": n_dup}


def clean_lines(raw: pd.DataFrame):
    df, n_dup = dedupe(raw)
    out = pd.DataFrame({
        "order_id": df["order_id"].str.strip(),
        "line_no": pd.to_numeric(df["line_no"], errors="coerce").astype("Int64"),
        "product_id": df["product_id"].str.strip(),
        "quantity": pd.to_numeric(df["quantity"], errors="coerce"),
        "unit_price": df["unit_price"].map(parse_money),
        "discount_pct": pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0.0),
    })
    key = out["order_id"] + "#" + out["line_no"].astype(str)
    reason = pd.Series(np.select(
        [out["quantity"].isna() | (out["quantity"] == 0), out["unit_price"].isna(), ~out["discount_pct"].between(0, 0.9)],
        ["null_or_zero_quantity", "missing_price", "discount_out_of_range"], ""), index=out.index)
    clean, rej = _split(out, reason != "", reason, "order_lines", key)
    clean["quantity"] = clean["quantity"].astype(int)
    clean["is_return"] = clean["quantity"] < 0
    return clean, rej, {"exact_duplicates_removed": n_dup}
