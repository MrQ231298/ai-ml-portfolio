"""
Generate a SYNTHETIC star-schema dataset for a fictional automotive
electronics plant (SMT -> final assembly -> end-of-line test).

Nothing here comes from a real plant. The structure reflects how such
plants usually record production, downtime and defects; the numbers are
invented and seeded. A few problems are *planted* on purpose (listed in
PLANTED_ISSUES) so the dashboard and insight checks can be scored
against a known ground truth.

    python src/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "star_schema"
SEED = 20261117
START, END = "2026-07-01", "2026-10-31"
SHIFT_MIN = 480          # 8-hour shift
BREAK_MIN = 30           # planned stop per shift (breaks / handover)

LINES = [
    # line_key, name, area, sequence in the route
    (1, "SMT-1", "Surface-mount", 1),
    (2, "SMT-2", "Surface-mount", 1),
    (3, "ASSY-1", "Final assembly", 2),
    (4, "EOL-1", "End-of-line test", 3),
]
PRODUCTS = [
    # product_key, code, family, ideal cycle time (s) by line_key
    (1, "ECU-A100", "Body control module", {1: 22, 2: 22, 3: 38, 4: 30}),
    (2, "ECU-B200", "Gateway module", {1: 26, 2: 26, 3: 45, 4: 34}),
    (3, "SNS-C300", "Sensor module", {1: 14, 2: 14, 3: 24, 4: 20}),
]
SHIFTS = [(1, "Early", "06:00"), (2, "Late", "14:00"), (3, "Night", "22:00")]
DOWNTIME_REASONS = [
    # key, reason, category, base rate (events / shift), mean minutes
    (1, "Feeder jam / misload", "Equipment", 0.9, 9),
    (2, "Pick-and-place nozzle error", "Equipment", 0.5, 12),
    (3, "Changeover overrun", "Changeover", 0.35, 25),
    (4, "Material shortage", "Material", 0.25, 30),
    (5, "Tester / fixture fault", "Equipment", 0.3, 18),
    (6, "Conveyor fault", "Equipment", 0.2, 15),
    (7, "Quality hold", "Quality", 0.12, 40),
    (8, "Operator unavailable", "People", 0.15, 20),
]
DEFECTS = [
    # key, defect, detected at, disposition share (rework fraction)
    (1, "Solder bridge", "AOI", 0.85),
    (2, "Missing component", "AOI", 0.7),
    (3, "Tombstoning", "AOI", 0.8),
    (4, "Connector mis-seat", "Assembly check", 0.9),
    (5, "Housing damage", "Visual", 0.2),
    (6, "Functional test fail", "EOL test", 0.5),
    (7, "Label / traceability error", "EOL test", 0.95),
]
LINE_DEFECTS = {1: [1, 2, 3], 2: [1, 2, 3], 3: [4, 5], 4: [6, 7]}
LINE_DEFECT_RATE = {1: 0.012, 2: 0.014, 3: 0.008, 4: 0.010}

PLANTED_ISSUES = [
    {"id": "P1", "what": "Solder-bridge spike on SMT-1 (e.g. after a stencil change)",
     "line": "SMT-1", "defect": "Solder bridge", "from": "2026-09-07", "to": "2026-09-20", "multiplier": 4.0},
    {"id": "P2", "what": "Changeover overruns three times as frequent on SMT-2",
     "line": "SMT-2", "reason": "Changeover overrun", "multiplier": 3.0},
    {"id": "P3", "what": "Tester / fixture faults on EOL-1 rise through October (wearing fixture)",
     "line": "EOL-1", "reason": "Tester / fixture fault", "from": "2026-10-01", "to": "2026-10-31", "multiplier": 3.0},
    {"id": "P4", "what": "Night shift runs ~4 percentage points slower (performance) on every line",
     "shift": "Night", "performance_delta": -0.04},
]


def build() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")

    dim_date = pd.DataFrame({"date": dates})
    dim_date["date_key"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["month_num"] = dim_date["date"].dt.month
    dim_date["month"] = dim_date["date"].dt.strftime("%b %Y")
    dim_date["iso_week"] = dim_date["date"].dt.isocalendar().week.astype(int)
    dim_date["week_start"] = (dim_date["date"] - pd.to_timedelta(dim_date["date"].dt.day_of_week, unit="D")).dt.date
    dim_date["weekday"] = dim_date["date"].dt.day_name()
    dim_date["is_working_day"] = dim_date["date"].dt.day_of_week < 6  # Mon-Sat
    dim_date["date"] = dim_date["date"].dt.date

    dim_line = pd.DataFrame(LINES, columns=["line_key", "line", "area", "route_step"])
    dim_product = pd.DataFrame([(k, c, f) for k, c, f, _ in PRODUCTS], columns=["product_key", "product_code", "product_family"])
    dim_shift = pd.DataFrame(SHIFTS, columns=["shift_key", "shift", "start_time"])
    dim_reason = pd.DataFrame([(k, r, c) for k, r, c, _, _ in DOWNTIME_REASONS], columns=["reason_key", "reason", "reason_category"])
    dim_defect = pd.DataFrame([(k, d, s) for k, d, s, _ in DEFECTS], columns=["defect_key", "defect", "detected_at"])
    ideal = {(p[0], ln): ct for p in PRODUCTS for ln, ct in p[3].items()}

    prod_rows, dt_rows, def_rows, tgt_rows = [], [], [], []
    event_id = 0
    for d in dates:
        wd = d.weekday()
        if wd == 6:
            continue  # Sunday closed
        shifts = [1] if wd == 5 else [1, 2, 3]  # Saturday early shift only
        dkey = int(d.strftime("%Y%m%d"))
        for s in shifts:
            # One product per shift, same product flows SMT -> ASSY -> EOL that shift
            product = int(rng.choice([1, 2, 3], p=[0.45, 0.25, 0.30]))
            for line_key, line_name, _, _ in LINES:
                planned_prod_min = SHIFT_MIN - BREAK_MIN
                # --- unplanned downtime events
                unplanned = 0.0
                for rk, reason, _, rate, mean_min in DOWNTIME_REASONS:
                    lam = rate
                    if reason in ("Feeder jam / misload", "Pick-and-place nozzle error") and line_key > 2:
                        lam = 0.0  # SMT-only failure modes
                    if reason == "Tester / fixture fault" and line_key != 4:
                        lam = rate * 0.15
                    if reason == "Changeover overrun" and line_name == "SMT-2":
                        lam *= 3.0  # P2
                    if reason == "Tester / fixture fault" and line_name == "EOL-1" and d >= pd.Timestamp("2026-10-01"):
                        lam *= 3.0  # P3
                    if reason == "Operator unavailable" and s == 3:
                        lam *= 1.5
                    for _ in range(rng.poisson(lam)):
                        dur = float(np.clip(rng.lognormal(np.log(mean_min), 0.6), 2, 150))
                        if unplanned + dur > planned_prod_min * 0.6:
                            continue
                        unplanned += dur
                        event_id += 1
                        dt_rows.append((event_id, dkey, s, line_key, product, rk, round(dur, 1)))
                run_min = planned_prod_min - unplanned
                # --- performance (speed losses / micro-stops)
                perf = rng.normal(0.91, 0.025) + (-0.04 if s == 3 else 0.0)  # P4
                perf = float(np.clip(perf, 0.6, 0.99))
                ct = ideal[(product, line_key)]
                total = int(run_min * 60 / ct * perf)
                # --- quality
                rate = LINE_DEFECT_RATE[line_key]
                defects_this = {}
                for dk in LINE_DEFECTS[line_key]:
                    r = rate / len(LINE_DEFECTS[line_key]) * rng.lognormal(0, 0.35)
                    if dk == 1 and line_name == "SMT-1" and pd.Timestamp("2026-09-07") <= d <= pd.Timestamp("2026-09-20"):
                        r *= 4.0  # P1
                    n = int(rng.binomial(total, min(r, 0.5)))
                    if n:
                        defects_this[dk] = n
                scrap = rework = 0
                for dk, n in defects_this.items():
                    rew_share = [x for x in DEFECTS if x[0] == dk][0][3]
                    n_rew = int(rng.binomial(n, rew_share))
                    rework += n_rew
                    scrap += n - n_rew
                    def_rows.append((dkey, s, line_key, product, dk, n, n_rew, n - n_rew))
                prod_rows.append((dkey, s, line_key, product, SHIFT_MIN, BREAK_MIN, planned_prod_min,
                                  round(unplanned, 1), round(run_min, 1), ct, total,
                                  total - scrap - rework, total - scrap, scrap, rework))
                # target: 85% of theoretical capacity for the product
                tgt_rows.append((dkey, s, line_key, int(planned_prod_min * 60 / ct * 0.85)))

    fact_production = pd.DataFrame(prod_rows, columns=[
        "date_key", "shift_key", "line_key", "product_key", "shift_minutes", "planned_stop_minutes",
        "planned_production_minutes", "unplanned_downtime_minutes", "run_minutes", "ideal_cycle_time_s",
        "total_count", "first_pass_good_count", "good_count", "scrap_count", "rework_count"])
    fact_downtime = pd.DataFrame(dt_rows, columns=[
        "event_id", "date_key", "shift_key", "line_key", "product_key", "reason_key", "duration_minutes"])
    fact_defects = pd.DataFrame(def_rows, columns=[
        "date_key", "shift_key", "line_key", "product_key", "defect_key", "defect_count", "reworked_count", "scrapped_count"])
    fact_target = pd.DataFrame(tgt_rows, columns=["date_key", "shift_key", "line_key", "target_units"])
    for df in (fact_production, fact_downtime, fact_defects, fact_target):
        df["data_flag"] = "SYNTHETIC_EXAMPLE_DATA"
    return {
        "dim_date": dim_date, "dim_line": dim_line, "dim_product": dim_product, "dim_shift": dim_shift,
        "dim_downtime_reason": dim_reason, "dim_defect": dim_defect,
        "fact_production": fact_production, "fact_downtime": fact_downtime,
        "fact_defects": fact_defects, "fact_target": fact_target,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = build()
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:22s} {len(df):6d} rows")
    (OUT / "planted_issues.json").write_text(json.dumps(
        {"data_flag": "SYNTHETIC_EXAMPLE_DATA", "seed": SEED, "issues": PLANTED_ISSUES}, indent=2))


if __name__ == "__main__":
    main()
