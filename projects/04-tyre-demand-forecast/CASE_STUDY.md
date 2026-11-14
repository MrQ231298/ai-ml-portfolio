# Case study: forecasting weekly tyre demand

**Audience:** a depot manager or purchasing lead — not a data-science
reviewer. The method, both sets of metrics, and the critical evaluation
are in [README.md](README.md).

**Status:** portfolio demonstration on *invented* weekly unit sales.
The seasonal shape is inspired by winter-tyre changeover and a spring
NCT-related uptick, but these are not real depot figures.

---

## The business problem

Buying too few tyres means a lost trade sale; buying too many ties up
cash and yard space. Demand is not flat: it rises when customers
change to winter tyres, it often picks up again in spring, and it
dips in the holiday weeks. A useful forecast is one a buyer can look
at eight weeks ahead and not be repeatedly surprised.

## What this demo shows

The app (`streamlit run app.py`) plots the invented weekly series and
three forecasts for 2025, made using only data up to the end of 2024:

- **same week last year** (a seasonal naive rule)
- a classical **ETS / Holt–Winters** seasonal model
- a **gradient-boosting** model that uses recent and year-ago lags

On a rolling back-test — 51 separate eight-week forecasts, each using
only the history available at the time — ETS was the most accurate
(mean error **48 units**, or about **4.7%**). Last year and the lag
model were close behind (about 54 and 53 units).

When we looked at 2025 alone, the ranking changed: the lag model did
best and ETS was a little high in the autumn. That is a useful warning.
A single year's picture can flatter one model; the rolling test is the
fairer one to plan stock around, and it is the one this project treats
as the headline.

## What this could be worth

Used carefully, a short-horizon weekly forecast can:

- give purchasing a **starting number** for the next two months,
  with a typical error of around 5% on this tidy invented series
- make the **winter and spring peaks visible** early enough to
  place a factory or wholesale order
- stop a team from changing method every January because "last year
  was different"

It cannot replace a buyer looking at a known contract, a weather
event, or a cancelled fleet order. And because this demo series is
invented, the 5% error should not be quoted as a live-depot promise.

## What we would do next with a client

Take a real weekly extract, agree the horizon (eight weeks is a
suggestion, not a law), run the same three-way comparison on a
rolling test, and only then decide whether ETS, last-year, or a lag
model is the default in the weekly buying meeting. The important
habit is already here: keep the simple seasonal baseline, and do not
let a single year pick the winner.
