"""Dashboard renders against the recorded sink (skipped if the run has not been done)."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "sink.db").exists(), reason="run src/run_experiment.py first")
def test_dashboard_renders_metrics():
    from streamlit.testing.v1 import AppTest
    os.environ["DASH_NO_REFRESH"] = "1"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Messages processed", "Duplicates dropped"]
