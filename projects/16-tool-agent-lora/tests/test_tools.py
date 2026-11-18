"""The tools are safe and correct; no model is loaded in these tests."""

import pytest

from tools import ToolError, calculator, search_docs, sql_query


def test_sql_is_read_only():
    assert sql_query("SELECT COUNT(*) FROM products") == [(15,)]
    for q in ["DELETE FROM orders", "SELECT 1; DROP TABLE orders", "UPDATE products SET unit_price = 0",
              "PRAGMA table_info(orders)"]:
        with pytest.raises(ToolError):
            sql_query(q)


def test_sql_errors_are_returned_as_tool_errors():
    with pytest.raises(ToolError, match="SQL error"):
        sql_query("SELECT nope FROM orders")


def test_calculator_handles_percent_thousands_and_rejects_code():
    assert calculator("1,200 * 1.23") == 1476.0
    assert calculator("200 * 15%") == 30.0
    assert calculator("round(100 / 3, 2)") == 33.33
    for bad in ["__import__('os').system('ls')", "open('x')", "2 ** 1000", "1 / 0"]:
        with pytest.raises(ToolError):
            calculator(bad)


def test_document_search_finds_the_right_section():
    r = search_docs("trade account payment terms")
    assert r["doc"] == "payment_terms" and "30 days" in r["text"]
