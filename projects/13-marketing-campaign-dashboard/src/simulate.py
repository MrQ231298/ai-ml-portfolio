"""
Simulate SYNTHETIC marketing data with a KNOWN causal truth.

A fictional Irish home-goods retailer runs 14 campaigns across six
channels and three audiences, January - October 2026. Each simulated
person has a purchase intent, receives ad touches (high-intent people are
targeted more by retargeting, brand search and affiliates - selection
bias, as in real ad platforms), and converts with a probability that
depends on intent plus the *true* effect of each touch.

Because the simulation draws one random number per person, the same
people can be re-run with any channel switched off. That gives the
**true incremental conversions per channel**, which real attribution
data never reveals. Attribution models are then scored against it.

Outputs (data/):
    SYNTHETIC_campaign_daily.csv     one row per campaign per day (the "ad platform export")
    SYNTHETIC_touchpoints.csv.gz     one row per touch (user, time, channel, campaign)
    SYNTHETIC_conversions.csv.gz      one row per order (user, time, revenue)
    truth_incrementality.json        true incremental conversions / revenue by channel
    SYNTHETIC_holdout_test.csv.gz     a randomised 10% hold-out for paid social prospecting

    python src/simulate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 1313
FLAG = "SYNTHETIC_EXAMPLE_DATA"
N_USERS = 160_000
START = pd.Timestamp("2026-01-01")
DAYS = 304  # to 31 Oct 2026

# channel: true per-touch effect on the log-odds of purchase (diminishing after 3 touches),
# who it targets (exponent on intent: >0 means it chases people who would buy anyway),
# reach (mean touches per targeted person), cost model.
CHANNELS = {
    "Paid Search - Brand":   {"effect": 0.10, "intent_bias": 2.5, "reach": 0.18, "cpc": 0.45, "ctr": 0.12},
    "Paid Search - Generic": {"effect": 0.40, "intent_bias": 1.0, "reach": 0.30, "cpc": 1.10, "ctr": 0.045},
    "Paid Social":           {"effect": 0.35, "intent_bias": 0.0, "reach": 0.55, "cpc": 0.70, "ctr": 0.012},
    "Display Retargeting":   {"effect": 0.05, "intent_bias": 2.0, "reach": 0.35, "cpc": 0.55, "ctr": 0.004},
    "Email":                 {"effect": 0.25, "intent_bias": 1.0, "reach": 0.40, "cpc": 0.0, "ctr": 0.03},
    "Affiliate":             {"effect": 0.02, "intent_bias": 3.0, "reach": 0.10, "cpc": 0.0, "ctr": 0.05},
}
AFFILIATE_COMMISSION = 0.08         # of affiliate-network-tracked revenue
EMAIL_COST_PER_SEND = 0.004         # ESP cost per email delivered (one touch = one opened send; 1 / ctr sends)
CAMPAIGNS = [
    # campaign, channel, audience
    ("BRAND-ALWAYS-ON", "Paid Search - Brand", "All"),
    ("GEN-HOMEWARE", "Paid Search - Generic", "Prospecting"),
    ("GEN-GARDEN-SPRING", "Paid Search - Generic", "Prospecting"),
    ("SOC-PROSPECT-LOOKALIKE", "Paid Social", "Prospecting"),
    ("SOC-PROSPECT-INTEREST", "Paid Social", "Prospecting"),
    ("SOC-RETARGET-CART", "Paid Social", "Retargeting"),
    ("DISP-RETARGET-VIEWERS", "Display Retargeting", "Retargeting"),
    ("DISP-RETARGET-CART", "Display Retargeting", "Retargeting"),
    ("EMAIL-NEWSLETTER", "Email", "Existing customers"),
    ("EMAIL-WINBACK", "Email", "Lapsed customers"),
    ("EMAIL-CART-ABANDON", "Email", "Retargeting"),
    ("AFF-VOUCHER-SITES", "Affiliate", "All"),
    ("AFF-CONTENT-BLOGS", "Affiliate", "Prospecting"),
    ("SOC-SUMMER-SALE", "Paid Social", "Prospecting"),
]
BASE_LOGIT = -5.0
INTENT_COEF = 2.4


def campaign_for(channel, intent, rng):
    opts = [c for c in CAMPAIGNS if c[1] == channel]
    if channel == "Paid Social":   # retargeting campaign for high intent, prospecting otherwise
        hi = intent > 1.0
        names = [c[0] for c in opts]
        p = np.where(hi[:, None], [0.2, 0.2, 0.5, 0.1], [0.4, 0.4, 0.05, 0.15])
        return np.array([rng.choice(names, p=pi / pi.sum()) for pi in p])
    return rng.choice([c[0] for c in opts], size=len(intent))


def simulate(rng):
    intent = rng.gamma(2.0, 0.5, N_USERS)                      # mean 1.0
    decision_day = rng.integers(30, DAYS, N_USERS)
    u = rng.random(N_USERS)                                      # one draw per person -> counterfactuals
    touches = []
    for ch, cfg in CHANNELS.items():
        w = intent ** cfg["intent_bias"]
        lam = cfg["reach"] * w / w.mean()
        n = rng.poisson(np.minimum(lam, 8))
        idx = np.repeat(np.arange(N_USERS), n)
        if idx.size == 0:
            continue
        lag = rng.integers(0, 30, idx.size)
        if ch in ("Display Retargeting", "Affiliate", "Paid Search - Brand"):
            lag = rng.integers(0, 4, idx.size)                   # late in the journey
        if ch == "Paid Social":
            lag = rng.integers(5, 30, idx.size)                  # early in the journey
        day = decision_day[idx] - lag
        camp = campaign_for(ch, intent[idx], rng)
        touches.append(pd.DataFrame({"user_id": idx, "day": day, "channel": ch, "campaign": camp,
                                     "hour": rng.integers(0, 24, idx.size)}))
    tp = pd.concat(touches, ignore_index=True)
    tp = tp[tp["day"] >= 0]
    tp = tp.sort_values(["user_id", "day", "hour"], ignore_index=True)
    counts = tp.groupby(["user_id", "channel"]).size().unstack(fill_value=0).reindex(range(N_USERS), fill_value=0)
    effects = np.array([CHANNELS[c]["effect"] for c in counts.columns])
    dimin = np.minimum(counts.values, 3)

    def converted(mask_off=None):
        e = effects.copy()
        if mask_off is not None:
            e[[list(counts.columns).index(c) for c in mask_off]] = 0.0
        logit = BASE_LOGIT + INTENT_COEF * np.log1p(intent) * 1.6 + dimin @ e
        return u < 1 / (1 + np.exp(-logit))

    conv = converted()
    truth = {}
    aov = np.round(rng.lognormal(np.log(85), 0.45, N_USERS), 2)
    for c in counts.columns:
        cf = converted([c])
        truth[c] = {"incremental_conversions": int(conv.sum() - cf.sum()),
                    "incremental_revenue": float(round(aov[conv & ~cf].sum(), 2))}
    no_ads = converted(list(counts.columns))
    truth["_baseline_no_ads"] = {"conversions": int(no_ads.sum())}
    truth["_total"] = {"conversions": int(conv.sum()), "revenue": float(round(aov[conv].sum(), 2)),
                       "incremental_all_paid_and_owned": int(conv.sum() - no_ads.sum())}
    # keep only touches up to and including the decision day; conversions at decision day 20:00-23:59
    conv_df = pd.DataFrame({"user_id": np.where(conv)[0]})
    conv_df["day"] = decision_day[conv_df.user_id]
    conv_df["hour"] = 23
    conv_df["revenue"] = aov[conv_df.user_id]
    # holdout test: 10% of users randomly withheld from paid social prospecting campaigns
    hold = rng.random(N_USERS) < 0.10
    pros = tp["campaign"].isin(["SOC-PROSPECT-LOOKALIKE", "SOC-PROSPECT-INTEREST", "SOC-SUMMER-SALE"])
    eff_social = CHANNELS["Paid Social"]["effect"]
    pros_counts = tp[pros].groupby("user_id").size().reindex(range(N_USERS), fill_value=0).values
    soc_col = list(counts.columns).index("Paid Social")
    dimin_hold = dimin.copy()
    dimin_hold[:, soc_col] = np.minimum(np.maximum(counts["Paid Social"].values - pros_counts, 0), 3)
    logit_h = BASE_LOGIT + INTENT_COEF * np.log1p(intent) * 1.6 + dimin_hold @ effects
    conv_hold_world = np.where(hold, u < 1 / (1 + np.exp(-logit_h)), conv)
    # people who would have been reached by prospecting (eligible), in each arm
    eligible = pros_counts > 0
    holdout = pd.DataFrame({"user_id": np.arange(N_USERS), "group": np.where(hold, "holdout", "exposed"),
                            "eligible_for_prospecting": eligible, "converted": conv_hold_world})
    true_pros_effect = int((conv & ~(u < 1 / (1 + np.exp(-logit_h))) & eligible & ~hold).sum())
    truth["_holdout_truth"] = {"incremental_conversions_from_prospecting_in_exposed_arm": true_pros_effect}
    return tp, conv_df, truth, holdout, intent


def platform_daily(tp, conv_df, rng):
    """Ad-platform style daily export per campaign: impressions, clicks, spend and the
    platform's own 'conversions' (any conversion within 7 days after a touch on that platform)."""
    rows = []
    t = tp.merge(conv_df[["user_id", "day", "revenue"]].rename(columns={"day": "conv_day"}), on="user_id", how="left")
    t["platform_conv"] = t["conv_day"].notna() & (t["conv_day"] - t["day"]).between(0, 7)
    # one claimed conversion per user per campaign (last touch in window)
    claimed = (t[t.platform_conv].sort_values(["day", "hour"]).groupby(["user_id", "campaign"]).tail(1))
    clicks = tp.groupby(["campaign", "day"]).size().rename("clicks")
    pc = claimed.groupby(["campaign", "day"]).agg(platform_conversions=("user_id", "count"), platform_revenue=("revenue", "sum"))
    d = pd.concat([clicks, pc], axis=1).fillna(0).reset_index()
    ch = {c[0]: c[1] for c in CAMPAIGNS}
    au = {c[0]: c[2] for c in CAMPAIGNS}
    d["channel"] = d["campaign"].map(ch)
    d["audience"] = d["campaign"].map(au)
    d["date"] = (START + pd.to_timedelta(d["day"], unit="D")).dt.date
    ctr = d["channel"].map(lambda c: CHANNELS[c]["ctr"]) * rng.lognormal(0, 0.15, len(d))
    d["impressions"] = np.round(d["clicks"] / ctr).astype(int)
    cpc = d["channel"].map(lambda c: CHANNELS[c]["cpc"]) * rng.lognormal(0, 0.10, len(d))
    d["spend"] = np.round(d["clicks"] * cpc, 2)
    em = d["channel"] == "Email"
    d.loc[em, "spend"] = np.round(d.loc[em, "impressions"] * EMAIL_COST_PER_SEND, 2)
    af = d["channel"] == "Affiliate"   # networks are paid commission on the sales THEY track
    d.loc[af, "spend"] = np.round(d.loc[af, "platform_revenue"] * AFFILIATE_COMMISSION, 2)
    d["clicks"] = d["clicks"].astype(int)
    d["platform_conversions"] = d["platform_conversions"].astype(int)
    d["platform_revenue"] = d["platform_revenue"].round(2)
    return d.drop(columns="day")


def main() -> None:
    rng = np.random.default_rng(SEED)
    tp, conv_df, truth, holdout, _ = simulate(rng)
    daily = platform_daily(tp, conv_df, rng)
    DATA.mkdir(exist_ok=True)
    tp_out = tp.assign(ts=START + pd.to_timedelta(tp["day"], unit="D") + pd.to_timedelta(tp["hour"], unit="h"))
    tp_out[["user_id", "ts", "channel", "campaign"]].assign(data_flag=FLAG).to_csv(DATA / "SYNTHETIC_touchpoints.csv.gz", index=False)
    conv_out = conv_df.assign(ts=START + pd.to_timedelta(conv_df["day"], unit="D") + pd.to_timedelta(conv_df["hour"], unit="h"))
    conv_out[["user_id", "ts", "revenue"]].assign(order_id=lambda d: "ORD-" + d["user_id"].astype(str), data_flag=FLAG).to_csv(
        DATA / "SYNTHETIC_conversions.csv.gz", index=False)
    daily.assign(data_flag=FLAG).to_csv(DATA / "SYNTHETIC_campaign_daily.csv", index=False)
    holdout.assign(data_flag=FLAG).to_csv(DATA / "SYNTHETIC_holdout_test.csv.gz", index=False)
    truth["_config"] = {"seed": SEED, "users": N_USERS, "channels": CHANNELS, "affiliate_commission": AFFILIATE_COMMISSION,
                        "note": "effects are per-touch log-odds lifts; truth = conversions(all) - conversions(channel switched off)"}
    (DATA / "truth_incrementality.json").write_text(json.dumps(truth, indent=1))
    print(json.dumps({k: v for k, v in truth.items() if k != "_config"}, indent=1))
    print(len(tp), "touches;", len(conv_df), "conversions;", len(daily), "campaign-days")


if __name__ == "__main__":
    main()
