"""Gather the run outputs into reports/metrics.json (single source for the README)."""

import json

from common import MODELS, REPORTS

out = {
    "training": json.loads((REPORTS / "training_metrics.json").read_text()),
    "monitoring": {k: v for k, v in json.loads((REPORTS / "monitoring_metrics.json").read_text()).items() if k != "months"},
    "serving_benchmark": json.loads((REPORTS / "serving_benchmark.json").read_text()),
    "served_model_card": json.loads((MODELS / "model_card.json").read_text()),
    "registry": json.loads((REPORTS / "mlflow_registry.json").read_text()),
    "dockerless_smoke": (REPORTS / "dockerless_smoke.txt").read_text().strip().splitlines(),
    "docker_build": "not run: no Docker daemon on the build machine (apt and dockerd unavailable)",
    "github_actions": "workflow written (ci/github-actions-ci.yml) but not run on GitHub: the push token lacks the `workflow` scope",
}
(REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
print("ok")
