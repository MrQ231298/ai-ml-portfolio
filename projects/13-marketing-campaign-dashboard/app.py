"""
Marketing campaign performance dashboard (SYNTHETIC data).

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"


@st.cache_data
def load():
    daily = pd.read_csv(DATA / "SYNTHETIC_campaign_daily.csv", parse_dates=["date"])
    m = json.loads((REPORTS / "metrics.json").read_text())
    attr = pd.read_csv(REPORTS / "attribution_conversions.csv", index_col=0)
    roas = pd.read_csv(REPORTS / "roas_by_channel.csv", index_col=0)
    return daily, m, attr, roas


def kpis(df):
    s = df[["impressions", "clicks", "spend", "platform_conversions", "platform_revenue"]].sum()
    return s, s.clicks / s.impressions, s.spend / max(s.platform_conversions, 1), s.platform_revenue / max(s.spend, 1e-9)


def main() -> None:
    st.set_page_config(page_title="Campaign performance", page_icon="📣", layout="wide")
    st.title("Marketing campaign performance (SYNTHETIC)")
    st.caption("Simulated home-goods retailer, Jan–Oct 2026, 14 campaigns / 6 channels. Platform figures are what ad platforms report; "
               "the Attribution tab compares models with the simulation's true incremental effect.")
    if not (REPORTS / "metrics.json").exists():
        st.error("Run `python src/simulate.py` and `python src/analyse.py` first."); st.stop()
    daily, m, attr, roas = load()

    with st.sidebar:
        st.header("Filters")
        dmin, dmax = daily.date.min().date(), daily.date.max().date()
        rng = st.date_input("Dates", (dmin, dmax), min_value=dmin, max_value=dmax)
        chans = st.multiselect("Channel", sorted(daily.channel.unique()), default=sorted(daily.channel.unique()))
        auds = st.multiselect("Audience", sorted(daily.audience.unique()), default=sorted(daily.audience.unique()))
        st.caption("Affiliate spend = 8% commission on the sales the affiliate network tracks.")
    if len(rng) != 2 or not chans or not auds:
        st.warning("Pick a date range, at least one channel and one audience."); st.stop()
    f = daily[daily.date.between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1])) & daily.channel.isin(chans) & daily.audience.isin(auds)]

    tab1, tab2, tab3 = st.tabs(["Overview", "Attribution vs truth", "Hold-out test"])
    with tab1:
        s, ctr, cpa, roas_p = kpis(f)
        c = st.columns(6)
        c[0].metric("Spend", f"€{s.spend:,.0f}")
        c[1].metric("Impressions", f"{s.impressions / 1e6:,.1f} m")
        c[2].metric("CTR", f"{ctr:.2%}")
        c[3].metric("Platform conversions", f"{s.platform_conversions:,.0f}")
        c[4].metric("Platform CPA", f"€{cpa:,.2f}")
        c[5].metric("Platform ROAS", f"{roas_p:,.2f}")
        a, b = st.columns(2)
        wk = f.assign(week=f.date.dt.to_period("W").dt.start_time).groupby(["week", "channel"], as_index=False)["spend"].sum()
        a.subheader("Weekly spend by channel")
        a.altair_chart(alt.Chart(wk).mark_area().encode(x=alt.X("week:T", title=None), y=alt.Y("spend:Q", title="€"), color="channel:N"), width="stretch")
        ch = f.groupby("channel", as_index=False)[["spend", "clicks", "impressions", "platform_conversions", "platform_revenue"]].sum()
        ch["ROAS (platform)"] = ch.platform_revenue / ch.spend
        b.subheader("Platform ROAS by channel")
        b.altair_chart(alt.Chart(ch).mark_bar().encode(y=alt.Y("channel:N", sort="-x", title=None), x=alt.X("ROAS (platform):Q"),
                                                       tooltip=["channel", alt.Tooltip("spend:Q", format=",.0f"), alt.Tooltip("ROAS (platform):Q", format=".2f")]),
                       width="stretch")
        st.subheader("Campaigns")
        camp = f.groupby(["campaign", "channel", "audience"], as_index=False)[["impressions", "clicks", "spend", "platform_conversions", "platform_revenue"]].sum()
        camp["CTR"] = camp.clicks / camp.impressions
        camp["CPA"] = camp.spend / camp.platform_conversions.where(camp.platform_conversions > 0)
        camp["ROAS"] = camp.platform_revenue / camp.spend.where(camp.spend > 0)
        st.dataframe(camp.sort_values("spend", ascending=False).style.format(
            {"impressions": "{:,.0f}", "clicks": "{:,.0f}", "spend": "€{:,.0f}", "platform_conversions": "{:,.0f}",
             "platform_revenue": "€{:,.0f}", "CTR": "{:.2%}", "CPA": "€{:,.2f}", "ROAS": "{:.2f}"}), hide_index=True, width="stretch")
    with tab2:
        st.markdown(f"Ad platforms claimed **{m['platform_over_claim']['sum_platform_conversions']:,}** conversions, "
                    f"**{m['platform_over_claim']['ratio']:.2f}×** the {m['platform_over_claim']['orders_with_any_touch']:,} orders that had any ad touch. "
                    f"Only **{m['true_incremental_total']:,}** orders were caused by marketing in the simulation.")
        sh = (attr / attr.sum()).reset_index().melt(id_vars="index", var_name="model", value_name="share").rename(columns={"index": "channel"})
        pick = st.multiselect("Models", list(attr.columns), default=["Last click", "Markov removal effect", "Shapley (conversion-rate game)", "First click", "TRUE incremental"])
        sh = sh[sh.model.isin(pick)]
        st.altair_chart(alt.Chart(sh).mark_bar().encode(x=alt.X("share:Q", axis=alt.Axis(format="%"), title="share of credited conversions"),
                                                         y=alt.Y("model:N", title=None), color="model:N", row=alt.Row("channel:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
                                                         ).properties(height=110), width="stretch")
        err = pd.DataFrame({"model": list(m["attribution_share_error"]), "misallocated share": list(m["attribution_share_error"].values()),
                            "rank correlation": [m["attribution_rank_spearman"][k] for k in m["attribution_share_error"]]}).sort_values("misallocated share")
        st.dataframe(err.style.format({"misallocated share": "{:.1%}", "rank correlation": "{:.2f}"}), hide_index=True)
        st.subheader("ROAS: last click vs true incremental")
        st.dataframe(roas[["spend", "roas_platform", "roas_last_click", "roas_markov", "roas_TRUE_incremental"]].style.format("{:,.2f}"))
    with tab3:
        lift = m["holdout_test_paid_social_prospecting"]
        c = st.columns(4)
        c[0].metric("Exposed conversion rate", f"{lift['exposed_cvr']:.2%}")
        c[1].metric("Hold-out conversion rate", f"{lift['holdout_cvr']:.2%}")
        c[2].metric("Lift", f"{lift['absolute_lift'] * 100:.2f} pp", f"{lift['relative_lift']:.0%} relative")
        c[3].metric("Incremental conversions (est.)", f"{lift['estimated_incremental_conversions_exposed']:,.0f}",
                    f"95% CI {lift['estimated_incremental_ci95'][0]:,.0f}–{lift['estimated_incremental_ci95'][1]:,.0f}")
        st.write(f"Simulated truth: {m['holdout_truth']['incremental_conversions_from_prospecting_in_exposed_arm']:,} incremental conversions. "
                 "A randomised hold-out is the only method here that does not depend on who the platform chose to target.")


if __name__ == "__main__":
    main()
