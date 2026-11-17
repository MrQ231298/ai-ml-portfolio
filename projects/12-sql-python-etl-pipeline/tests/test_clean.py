"""Unit tests for the cleaning rules (no database needed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from etl import clean as C  # noqa: E402


@pytest.mark.parametrize("raw,expected", [("€12.50", 12.5), ("EUR 12.50", 12.5), ("12,50", 12.5), ("1,234.50", 1234.5), ("7", 7.0)])
def test_parse_money(raw, expected):
    assert C.parse_money(raw) == expected


def test_parse_money_blank_is_nan():
    assert pd.isna(C.parse_money(None)) and pd.isna(C.parse_money(""))


@pytest.mark.parametrize("raw", ["2026-03-04", "04/03/2026", "Mar 04 2026"])
def test_parse_date_is_day_first_never_month_first(raw):
    assert C.parse_date(raw) == pd.Timestamp("2026-03-04")


def test_parse_timestamps():
    assert C.parse_date("2026-03-04T10:15:00") == pd.Timestamp("2026-03-04 10:15")
    assert C.parse_date("04/03/2026 10:15") == pd.Timestamp("2026-03-04 10:15")
    assert pd.isna(C.parse_date("next tuesday"))


def test_country_and_phone():
    assert C.normalise_country(" Rep. of Ireland") == "IE"
    assert C.normalise_country("u.k.") == "GB"
    assert C.normalise_country("GB-NIR") == "NI"
    assert C.normalise_country("Atlantis") is None
    assert C.normalise_phone("0871234567") == "+353871234567"
    assert C.normalise_phone("+353 87 123 4567") == "+353871234567"


def test_email_validation():
    assert C.valid_email("aoife.murphy1@example.ie")
    assert not C.valid_email("aoife.murphy1 at example.ie")
    assert not C.valid_email("aoife.murphy1")


def test_clean_lines_rejects_and_flags_returns():
    raw = pd.DataFrame({"order_id": ["A", "A", "B", "B", "B"], "line_no": ["1", "2", "1", "2", "2"],
                        "product_id": ["P1"] * 5, "quantity": ["2", "0", "-1", None, None],
                        "unit_price": ["3.00"] * 5, "discount_pct": ["0"] * 5})
    clean, rej, s = C.clean_lines(raw)
    assert s["exact_duplicates_removed"] == 1
    assert list(clean["is_return"]) == [False, True]
    assert sorted(rej["reason"]) == ["null_or_zero_quantity", "null_or_zero_quantity"]


def test_clean_orders_future_date_relative_to_extract():
    raw = pd.DataFrame({"order_id": ["1", "2"], "customer_id": ["C1", "C1"], "order_ts": ["2026-06-30T10:00:00", "2026-07-02T10:00:00"],
                        "channel": ["Web", "web"], "status": ["COMPLETED", "completed"], "currency": ["eur", "EUR"]})
    clean, rej, _ = C.clean_orders(raw, pd.Timestamp("2026-07-01"))
    assert list(clean["order_id"]) == ["1"] and list(rej["reason"]) == ["future_date"]
    assert clean["status"].iloc[0] == "completed" and clean["currency"].iloc[0] == "EUR"
