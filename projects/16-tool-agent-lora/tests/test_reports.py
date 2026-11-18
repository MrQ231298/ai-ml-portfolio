"""The recorded metrics agree with the per-task rows and the fine-tune log (no model needed)."""

import json
from pathlib import Path

import pandas as pd
import pytest

R = Path(__file__).resolve().parents[1] / "reports"


@pytest.mark.skipif(not (R / "metrics.json").exists(), reason="run src/evaluate.py")
def test_metrics_match_eval_rows_and_finetune_log():
    m = json.loads((R / "metrics.json").read_text())
    rows = pd.read_csv(R / "eval_rows.csv")
    for (planner, split), g in rows.groupby(["planner", "split"]):
        v = m["planners"][planner][split]
        assert v["n"] == len(g) == m["test_sizes"][split]
        assert v["task_success"] == pytest.approx(g["success"].mean(), abs=1e-4)
        assert g.groupby("family").size().eq(4).all()          # 4 tasks per family per split
    assert m["planners"]["gold_plan"]["test_ood"]["task_success"] == 1.0
    ft = json.loads((R / "finetune_log.json").read_text())
    assert m["finetune"] == ft and ft["trainable_params"] < 0.01 * ft["total_params"]
