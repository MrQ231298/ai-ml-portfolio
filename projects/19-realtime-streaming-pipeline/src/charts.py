"""Static charts from the recorded run's sink (reports/*.png)."""

import json
import sqlite3

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import REPORTS, ROOT  # noqa: E402

con = sqlite3.connect(ROOT / "data" / "sink.db")
w = pd.read_sql("select * from window_results", con)
w["window_start"] = pd.to_datetime(w.window_start)
late = pd.read_sql("select * from late_events", con)
late["event_time"] = pd.to_datetime(late.event_time)
late["delay_min"] = (pd.to_datetime(late.watermark) - late.event_time).dt.total_seconds() / 60 + 10
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
for b in ["Manhattan", "Queens", "Brooklyn"]:
    g = w[w.borough == b].sort_values("window_start")
    ax[0].plot(g.window_start, g.trips, label=b, lw=1)
ax[0].set_ylabel("trips / 5 min"); ax[0].legend(); ax[0].set_title("Exactly-once window results, 15-16 Jan 2024 (after a mid-stream crash)")
ax[1].hist(late.delay_min.clip(upper=70), bins=40, color="tab:red")
ax[1].set_xlabel("how far behind the newest event the late event was (min)"); ax[1].set_ylabel("late events")
plt.tight_layout(); plt.savefig(REPORTS / "windows_and_late_events.png", dpi=110)
m = json.loads((REPORTS / "metrics.json").read_text())
eo, nv = m["exactly_once_vs_truth"], m["naive_vs_truth"]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(["truth", "exactly-once\n(after crash)", "naive\n(after crash)"], [eo["expected_trips"], eo["sink_trips"], nv["sink_trips"]],
       color=["grey", "tab:green", "tab:red"])
ax.set_ylabel("trips counted in windows"); ax.set_title("Same crash, two designs")
plt.tight_layout(); plt.savefig(REPORTS / "exactly_once_vs_naive.png", dpi=110)
print("ok")
