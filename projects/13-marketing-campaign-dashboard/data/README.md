# Data folder — SYNTHETIC marketing data (simulated)

**Everything here is simulated** by [`../src/simulate.py`](../src/simulate.py)
(seed `1313`). The retailer, people, campaigns, spend and orders are
invented. Every row carries `data_flag = SYNTHETIC_EXAMPLE_DATA`.

| File | Content |
| --- | --- |
| `SYNTHETIC_campaign_daily.csv` | Ad-platform style export: date, campaign, channel, audience, impressions, clicks, spend, platform_conversions, platform_revenue. Platform conversions = any order within 7 days after a touch on that campaign (one per person per campaign), so platforms double count, as in real life. |
| `SYNTHETIC_touchpoints.csv.gz` | user_id, ts, channel, campaign |
| `SYNTHETIC_conversions.csv.gz` | order_id, user_id, ts, revenue |
| `SYNTHETIC_holdout_test.csv.gz` | user_id, group (exposed / holdout), eligible_for_prospecting, converted — a separate run in which 10% of people never see paid social prospecting |
| `truth_incrementality.json` | true incremental conversions and revenue per channel (counterfactual re-run), the simulation settings, and the hold-out's true effect |

Why simulated: public campaign datasets either lack user-level journeys,
lack a known causal answer, or carry non-commercial licences (e.g. the
Criteo attribution dataset, CC BY-NC-SA 4.0). For an attribution
*evaluation* a known answer is essential, and only a simulation provides
one.
