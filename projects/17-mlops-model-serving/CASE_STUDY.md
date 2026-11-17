# Case study: keeping a prediction model healthy after launch

**Audience:** an operations or product manager, not an engineer.
Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **real, public** data: two years of
hourly bike-share rentals and weather from Washington DC (UCI, CC BY 4.0).

---

## The business problem

A forecasting model is usually judged on the day it is built. Then the
world changes: demand grows, the weather shifts, customer habits move on.
The model quietly gets worse, and nobody notices until the decisions it
drives go wrong.

## What this demo shows

- **A model behind a real web service.** Any system can send it the
  current conditions and get back the expected number of rentals in the
  next hour. It replied in about 8 milliseconds for a single request.
- **A record of every model.** Each version is stored with the data,
  settings and score that produced it, so you always know what is live.
- **Nine months watched, month by month.** In May 2012 the model's error
  rose 43% above its launch level. That triggered a retrain on the latest
  year of data. The new version was tested on the last two weeks before
  it replaced the old one.
- **The payoff.** Over the following seven months the retrained model's
  errors were about a quarter lower than if the original had been left
  in place.
- **A lesson about alarms.** "Data drift" warnings went off every single
  month, because one month's weather never looks like a whole year's.
  Only the accuracy check pointed at the month that mattered.

## What it means in practice

The same set-up fits any model a business relies on, such as demand,
staffing, pricing or churn. You can see what is running, check it is
still accurate, and update it under a rule agreed in advance, not by
gut feel.

## Limits

The container build and the automated test pipeline are written and
ready, but were not run on this machine. The data is from one city in
2011–2012.
