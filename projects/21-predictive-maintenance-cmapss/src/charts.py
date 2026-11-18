"""Charts for the README from reports/*.csv / *.json (run src/run.py first)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from data import ROOT  # noqa: E402

REP = ROOT / "reports"


def main(fd="FD001"):
    m = json.loads((REP / "metrics.json").read_text())[fd]
    tr = pd.read_csv(REP / f"engine_traces_{fd}.csv")
    test = tr[tr.split == "test"]
    th_rul = m["policies"]["predicted_rul_alert"]["alert_when_predicted_rul_le"]
    th_h = m["policies"]["health_score_alert"]["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for u in sorted(test.unit.unique())[:4]:
        e = test[test.unit == u]
        axes[0].plot(e.cycle, e.rul.clip(upper=125), "k--", lw=0.8)
        axes[0].plot(e.cycle, e.pred_rul, lw=1.2, label=f"engine {u}")
    axes[0].axhline(th_rul, color="red", lw=1, label=f"alert: predicted RUL <= {th_rul:g}")
    axes[0].set(xlabel="cycle", ylabel="RUL (cycles)", title=f"{fd} held-out engines: predicted (solid) vs true capped RUL (dashed)")
    axes[0].legend(fontsize=7)
    for u in sorted(test.unit.unique()):
        e = test[test.unit == u]
        axes[1].plot(-e.rul, e.health_score, lw=0.7, alpha=0.7)
    axes[1].axhline(th_h, color="red", lw=1, label=f"cost-chosen threshold {th_h:.3f}")
    axes[1].set(xlabel="cycles before failure", ylabel=f"health score ({m['anomaly_selected_on_val']})",
                title=f"{fd}: unsupervised health score, all {test.unit.nunique()} held-out engines")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(REP / f"rul_and_health_{fd}.png", dpi=120); plt.close(fig)

    curves = json.loads((REP / f"cost_curves_{fd}.json").read_text())
    pol = m["policies"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    x, y = zip(*curves["predicted_rul"])
    axes[0].plot(x, y); axes[0].axvline(th_rul, color="red", lw=1)
    axes[0].set(xlabel="alert when predicted RUL <= ...", ylabel="mean cost per engine (validation)",
                title="choosing the alert threshold by cost")
    names = ["run_to_failure", "fixed_interval", "health_score_alert", "predicted_rul_alert", "oracle"]
    axes[1].bar(range(5), [pol[n]["mean_cost"] for n in names], color=["grey", "tab:orange", "tab:green", "tab:blue", "black"])
    for i, n in enumerate(names):
        axes[1].text(i, pol[n]["mean_cost"] + 0.15, f"{pol[n]['failures']} fail", ha="center", fontsize=8)
    axes[1].set_xticks(range(5), ["run to\nfailure", "fixed\ninterval", "health\nscore", "predicted\nRUL", "oracle"], fontsize=8)
    axes[1].set(ylabel="mean cost per engine", title=f"{fd}: held-out engines (failure = 10x planned job)")
    off = pd.read_csv(REP / f"official_test_predictions_{fd}.csv")
    axes[2].scatter(off.true_rul, off.pred_rul, s=10); axes[2].plot([0, 150], [0, 150], "k--", lw=0.8)
    axes[2].set(xlabel="true RUL", ylabel="predicted RUL", title=f"{fd} official test set (100 engines)")
    fig.tight_layout(); fig.savefig(REP / f"policies_{fd}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    for fd in ("FD001", "FD003"):
        main(fd)
    print("ok")
