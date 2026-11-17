"""
Campaign KPIs, attribution comparison against the simulated causal truth,
and the hold-out lift test.

    python src/analyse.py   -> reports/metrics.json, reports/*.csv, reports/*.png, reports/insights.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attribution as A  # noqa: E402
from simulate import FLAG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
RNG = np.random.default_rng(13)


def load():
    daily = pd.read_csv(DATA / "SYNTHETIC_campaign_daily.csv", parse_dates=["date"])
    tp = pd.read_csv(DATA / "SYNTHETIC_touchpoints.csv.gz", parse_dates=["ts"])
    conv = pd.read_csv(DATA / "SYNTHETIC_conversions.csv.gz", parse_dates=["ts"])
    hold = pd.read_csv(DATA / "SYNTHETIC_holdout_test.csv.gz")
    for d in (daily, tp, conv, hold):
        assert (d["data_flag"] == FLAG).all()
    truth = json.loads((DATA / "truth_incrementality.json").read_text())
    return daily, tp, conv, hold, truth


def kpi_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, as_index=False)[["impressions", "clicks", "spend", "platform_conversions", "platform_revenue"]].sum()
    g["ctr"] = g["clicks"] / g["impressions"]
    g["cpc"] = g["spend"] / g["clicks"]
    g["cvr_platform"] = g["platform_conversions"] / g["clicks"]
    g["cpa_platform"] = g["spend"] / g["platform_conversions"]
    g["roas_platform"] = g["platform_revenue"] / g["spend"]
    return g


def holdout_lift(hold: pd.DataFrame) -> dict:
    e = hold[hold.eligible_for_prospecting]
    x = e[e.group == "exposed"]["converted"]; h = e[e.group == "holdout"]["converted"]
    diff = x.mean() - h.mean()
    se = np.sqrt(x.mean() * (1 - x.mean()) / len(x) + h.mean() * (1 - h.mean()) / len(h))
    z = diff / se
    return {"exposed_n": int(len(x)), "holdout_n": int(len(h)), "exposed_cvr": float(x.mean()), "holdout_cvr": float(h.mean()),
            "absolute_lift": float(diff), "relative_lift": float(diff / h.mean()),
            "lift_ci95": [float(diff - 1.96 * se), float(diff + 1.96 * se)], "p_value": float(2 * stats.norm.sf(abs(z))),
            "estimated_incremental_conversions_exposed": float(diff * len(x)),
            "estimated_incremental_ci95": [float((diff - 1.96 * se) * len(x)), float((diff + 1.96 * se) * len(x))]}


def main() -> None:
    daily, tp, conv, hold, truth = load()
    REPORTS.mkdir(exist_ok=True)
    channels = sorted(daily["channel"].unique())

    # ---------- attribution
    j = A.journeys(tp, conv)
    n_users = int(truth["_config"]["users"])
    untouched = n_users - len(j)
    untouched_conv = int(len(conv) - j.converted.sum())
    base_rate = untouched_conv / untouched
    results = {name: fn(j) for name, fn in A.MODELS.items()}
    results["Shapley (conversion-rate game)"] = A.shapley(j, base_rate)
    true_conv = pd.Series({c: truth[c]["incremental_conversions"] for c in channels})
    true_rev = pd.Series({c: truth[c]["incremental_revenue"] for c in channels})
    comp = pd.DataFrame({name: r["conversions"].reindex(channels).fillna(0) for name, r in results.items()})
    comp["TRUE incremental"] = true_conv
    shares = comp / comp.sum()
    err = {m: float((shares[m] - shares["TRUE incremental"]).abs().sum() / 2) for m in comp.columns if m != "TRUE incremental"}
    rank = {m: float(stats.spearmanr(comp[m], comp["TRUE incremental"]).statistic) for m in comp.columns if m != "TRUE incremental"}
    comp.round(1).to_csv(REPORTS / "attribution_conversions.csv")

    # ---------- spend (affiliate = commission on network-tracked revenue, already in the export)
    spend = daily.groupby("channel")["spend"].sum()
    rev_attr = pd.DataFrame({name: r["revenue"].reindex(channels).fillna(0) for name, r in results.items()})
    roas = pd.DataFrame({
        "spend": spend,
        "roas_platform": daily.groupby("channel")["platform_revenue"].sum() / spend,
        "roas_last_click": rev_attr["Last click"] / spend,
        "roas_markov": rev_attr["Markov removal effect"] / spend,
        "roas_shapley": rev_attr["Shapley (conversion-rate game)"] / spend,
        "roas_TRUE_incremental": true_rev / spend,
        "cpa_last_click": spend / comp["Last click"],
        "cpa_TRUE_incremental": spend / true_conv,
    }).loc[channels]
    roas.round(3).to_csv(REPORTS / "roas_by_channel.csv")

    # ---------- platform KPIs
    by_channel = kpi_table(daily, ["channel"]).set_index("channel")
    by_aud = kpi_table(daily, ["audience"])
    by_camp = kpi_table(daily, ["campaign", "channel", "audience"]).sort_values("spend", ascending=False)
    by_camp.round(4).to_csv(REPORTS / "kpis_by_campaign.csv", index=False)
    by_aud.round(4).to_csv(REPORTS / "kpis_by_audience.csv", index=False)
    total_spend = float(spend.sum())
    total_rev = float(conv["revenue"].sum())
    touched_conv = int(j.converted.sum())
    platform_claimed = int(daily["platform_conversions"].sum())
    lift = holdout_lift(hold)

    metrics = {
        "data_flag": FLAG,
        "period": "2026-01-01 to 2026-10-31",
        "rows": {"campaign_days": int(len(daily)), "touchpoints": int(len(tp)), "conversions": int(len(conv)), "simulated_people": n_users},
        "totals": {"spend": round(total_spend, 2), "revenue_all_orders": round(total_rev, 2), "orders": int(len(conv)),
                   "orders_with_any_touch": touched_conv, "orders_no_touch_organic": untouched_conv,
                   "blended_roas": total_rev / total_spend, "blended_cpa": total_spend / len(conv),
                   "impressions": int(daily.impressions.sum()), "clicks": int(daily.clicks.sum()),
                   "ctr": float(daily.clicks.sum() / daily.impressions.sum())},
        "platform_over_claim": {"sum_platform_conversions": platform_claimed, "orders_with_any_touch": touched_conv,
                                "ratio": platform_claimed / touched_conv},
        "true_incremental_total": truth["_total"]["incremental_all_paid_and_owned"],
        "attribution_share_error": err,
        "attribution_rank_spearman": rank,
        "attribution_conversions": comp.round(1).to_dict(),
        "roas_by_channel": roas.round(3).to_dict("index"),
        "kpis_by_channel": by_channel[["impressions", "clicks", "spend", "ctr", "cpc", "platform_conversions", "roas_platform"]].round(4).to_dict("index"),
        "holdout_test_paid_social_prospecting": lift,
        "holdout_truth": truth["_holdout_truth"],
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ---------- plots
    fig, ax = plt.subplots(figsize=(10, 5))
    show = ["Last click", "First click", "Linear", "Markov removal effect", "Shapley (conversion-rate game)", "TRUE incremental"]
    x = np.arange(len(channels)); w = 0.13
    for i, m in enumerate(show):
        ax.bar(x + (i - 2.5) * w, shares[m] * 100, w, label=m, color=None if m != "TRUE incremental" else "black")
    ax.set_xticks(x); ax.set_xticklabels(channels, rotation=20, ha="right"); ax.set_ylabel("% of credited conversions")
    ax.set_title("Attribution share by channel vs simulated true incrementality (SYNTHETIC)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "attribution_vs_truth.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    r = roas[["roas_last_click", "roas_TRUE_incremental"]].drop(index="Email")
    r.plot.barh(ax=ax, color=["#3b6ea5", "black"])
    ax.set_xlabel("revenue per € spent"); ax.set_title("Last-click ROAS vs true incremental ROAS (Email excluded: near-zero cost)")
    ax.legend(["Last click", "True incremental"]); fig.tight_layout(); fig.savefig(REPORTS / "roas_last_click_vs_true.png", dpi=120); plt.close(fig)

    wk = daily.assign(week=daily["date"].dt.to_period("W").dt.start_time).groupby(["week", "channel"])["spend"].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 4)); wk.plot.area(ax=ax, linewidth=0); ax.set_ylabel("spend (€) per week")
    ax.set_title("Weekly spend by channel (SYNTHETIC)"); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(REPORTS / "weekly_spend.png", dpi=120); plt.close(fig)

    write_insights(metrics, roas, err)
    print(json.dumps({k: metrics[k] for k in ["totals", "platform_over_claim", "attribution_share_error", "attribution_rank_spearman", "holdout_test_paid_social_prospecting", "holdout_truth"]}, indent=1, default=float))
    print(roas.round(2).to_string())
    print(comp.round(0).to_string())


def write_insights(m, roas, err):
    best = min(err, key=err.get); worst = max(err, key=err.get)
    lift = m["holdout_test_paid_social_prospecting"]
    lines = [
        "# Campaign performance insights (SYNTHETIC data)", "",
        "Auto-generated by `python src/analyse.py`. All figures refer to a simulated retailer, January–October 2026.", "",
        "## Headline", "",
        f"- Spend €{m['totals']['spend']:,.0f}; revenue €{m['totals']['revenue_all_orders']:,.0f} from {m['totals']['orders']:,} orders "
        f"(blended ROAS {m['totals']['blended_roas']:.2f}, blended CPA €{m['totals']['blended_cpa']:.2f}).",
        f"- {m['totals']['orders_no_touch_organic']:,} orders ({m['totals']['orders_no_touch_organic'] / m['totals']['orders']:.0%}) had no ad touch at all.",
        f"- Ad platforms together claimed {m['platform_over_claim']['sum_platform_conversions']:,} conversions — "
        f"{m['platform_over_claim']['ratio']:.2f}× the {m['platform_over_claim']['orders_with_any_touch']:,} orders that had any touch.",
        f"- Only {m['true_incremental_total']:,} orders were truly caused by marketing (simulated counterfactual).", "",
        "## Attribution models vs the truth", "",
        "Share of credit given to the wrong channel (0 = perfect, total-variation distance):", "",
        "| Model | Misallocated share | Rank correlation with truth |", "| --- | ---: | ---: |",
    ]
    for k in sorted(err, key=err.get):
        lines.append(f"| {k} | {err[k]:.1%} | {m['attribution_rank_spearman'][k]:.2f} |")
    lines += ["", f"Closest: **{best}**. Furthest: **{worst}**.", "",
              "## ROAS: what last click says vs what is true", "", "| Channel | Spend (€) | Last-click ROAS | True incremental ROAS |", "| --- | ---: | ---: | ---: |"]
    for c, r in roas.iterrows():
        lines.append(f"| {c} | {r['spend']:,.0f} | {r['roas_last_click']:.2f} | {r['roas_TRUE_incremental']:.2f} |")
    lines += ["", "## Hold-out test (paid social prospecting)", "",
              f"10% of people were randomly withheld. Among people eligible for prospecting, conversion was {lift['exposed_cvr']:.2%} exposed vs "
              f"{lift['holdout_cvr']:.2%} held out (lift {lift['absolute_lift'] * 100:.2f} pp, 95% CI {lift['lift_ci95'][0] * 100:.2f} to "
              f"{lift['lift_ci95'][1] * 100:.2f} pp, p = {lift['p_value']:.2g}). Estimated incremental conversions in the exposed group: "
              f"{lift['estimated_incremental_conversions_exposed']:,.0f} (95% CI {lift['estimated_incremental_ci95'][0]:,.0f}–"
              f"{lift['estimated_incremental_ci95'][1]:,.0f}); simulated truth {m['holdout_truth']['incremental_conversions_from_prospecting_in_exposed_arm']:,}.",
              "", "## Caveats", "",
              "Simulated data with a known answer. Real attribution cannot be scored this way; that is why the recommendation is to run hold-out tests on the biggest budget lines."]
    (REPORTS / "insights.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
