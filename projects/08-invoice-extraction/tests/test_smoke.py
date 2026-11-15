"""Smoke tests for the invoice extractor (run `pytest -q` from the project folder)."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extract_lib import (RuleExtractor, arithmetic_ok, clean_ocr, norm_value,  # noqa: E402
                         parse_date, pdf_text)


def test_parsers():
    assert parse_date("29/10/2026") == "2026-10-29"
    assert parse_date("24Sep2025") == "2025-09-24"
    assert parse_date("December 13, 2025") == "2025-12-13"
    assert clean_ocr("VAT 2.322.82") == "VAT 2,322.82"
    assert norm_value("vendor_name", "Suir Office  DAC") == norm_value("vendor_name", "SUIROFFICEDAC")


def test_rules_on_development_layout():
    truth = pd.read_csv(ROOT / "data" / "SYNTHETIC_invoice_ground_truth.csv").set_index("file")
    out = RuleExtractor().extract(pdf_text(ROOT / "data" / "invoices_pdf" / "A_000.pdf"))
    g = truth.loc["A_000.pdf"]
    assert out["invoice_number"] == g.invoice_number
    assert abs(out["total_amount"] - g.total_amount) < 0.01
    assert arithmetic_ok(out)


def test_ml_model_loads_and_extracts():
    ml = joblib.load(ROOT / "models" / "candidate_ranker.joblib")
    out = ml.extract(pdf_text(ROOT / "data" / "invoices_pdf" / "B_003.pdf"))
    assert out["total_amount"] is not None


def test_metrics_recorded():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert m["results"]["dev_test/digital/hybrid"]["mean_field_accuracy"] >= 0.9
