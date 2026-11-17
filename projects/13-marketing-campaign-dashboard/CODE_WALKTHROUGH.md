# Code walkthrough — marketing campaign dashboard

## The pipeline at a glance

```
1. python src/simulate.py
        ↓ people, intent, touches, purchases (+ counterfactual re-runs)
        ↓ data/SYNTHETIC_*.csv(.gz), data/truth_incrementality.json
2. python src/analyse.py
        ↓ KPIs, 7 attribution models (src/attribution.py), hold-out test
        ↓ reports/metrics.json, *.csv, *.png, insights.md
3. pytest -q
4. streamlit run app.py
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/simulate.py`](src/simulate.py) | The simulated world: channel effects, targeting bias, timing, spend, platform-style over-claiming, hold-out arm. |
| [`src/attribution.py`](src/attribution.py) | `journeys()`, five rule-based models, `markov()`, `shapley()`. |
| [`src/analyse.py`](src/analyse.py) | KPI tables, model scoring vs truth, ROAS comparisons, hold-out lift, plots, insights report. |
| [`app.py`](app.py) | Streamlit + Altair dashboard (three tabs). |
| [`tests/test_attribution.py`](tests/test_attribution.py) | Hand-worked toy journeys for each model, conservation of credit, output checks. |

## Key ideas in the code

**One random draw per person.** `u = rng.random(N_USERS)` is drawn
once. A person converts if `u < p`. Switching a channel off lowers `p`
for the same `u`, so the difference in conversions is exactly that
channel's causal effect — no sampling noise between worlds.

**Selection bias is built in on purpose.** `intent_bias` makes some
channels reach high-intent people far more often, which is what
retargeting and brand search do in real accounts.

**Markov removal effect.** Transition counts between channels (with
START, CONV and NULL states) become a probability matrix; the chance of
reaching CONV from START is solved as an absorbing Markov chain.
Removing a channel redirects its transitions to NULL; the drop in
conversion probability is its removal effect.

**Shapley values.** For each set of channels seen in a converting
journey, the marginal change in conversion rate when adding a channel
is averaged over all orders of adding channels. Negative values are
clipped to zero (a documented judgement call).
