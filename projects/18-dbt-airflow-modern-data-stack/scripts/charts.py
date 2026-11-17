"""Charts from the marts (reports/*.png)."""

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
con = duckdb.connect(str(ROOT / "warehouse" / "nyc_taxi.duckdb"), read_only=True)
d = con.execute("""select trip_date, pickup_borough, trips from marts.agg_daily_borough
                   where pickup_borough in ('Manhattan', 'Queens', 'Brooklyn') order by 1""").df()
fig, ax = plt.subplots(figsize=(9, 4))
for b, g in d.groupby("pickup_borough"):
    ax.plot(g.trip_date, g.trips, label=b)
ax.set_yscale("log"); ax.set_ylabel("trips per day (log)"); ax.legend(); ax.set_title("Daily yellow-taxi trips by pickup borough, Jan-Feb 2024")
plt.tight_layout(); plt.savefig(ROOT / "reports" / "daily_trips_by_borough.png", dpi=110)
h = con.execute("select iso_weekday, hour_of_day, avg_trips from marts.agg_hour_of_week").df().pivot(index="iso_weekday", columns="hour_of_day", values="avg_trips")
fig, ax = plt.subplots(figsize=(9, 3.2))
im = ax.imshow(h.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); ax.set_xlabel("hour of day")
plt.colorbar(im, label="avg trips / hour"); ax.set_title("Average trips per hour of week (agg_hour_of_week)")
plt.tight_layout(); plt.savefig(ROOT / "reports" / "hour_of_week_heatmap.png", dpi=110)
q = con.execute("select source_month, outcome, row_count from marts.dq_rejections_by_month where outcome <> 'valid'").df()
fig, ax = plt.subplots(figsize=(8, 3.2))
p = q.pivot(index="outcome", columns="source_month", values="row_count").fillna(0).sort_values("2024-01")
p.plot.barh(ax=ax); ax.set_xlabel("rows rejected"); ax.set_title("Rows rejected by each data-quality rule")
plt.tight_layout(); plt.savefig(ROOT / "reports" / "dq_rejections.png", dpi=110)
print("ok")
