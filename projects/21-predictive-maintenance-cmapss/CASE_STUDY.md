# Case study: servicing machines before they fail, without servicing them too early

**Audience:** an operations, maintenance or plant manager, not a data
scientist. Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **public, simulated** data: NASA's
jet-engine wear simulation, run until each engine fails. The money
figures are **illustrative assumptions**.

---

## The business problem

Every plant balances two costs. If a machine breaks unexpectedly, the
line stops, parts are scrapped and engineers are pulled off other work.
If it is serviced too early, money is spent on parts and downtime that
were not yet needed. Most sites use a calendar: service every so many
weeks or cycles. That is simple, but every machine is treated as average.

## What this demo shows

The data is 100 simulated engines, each recorded from new until failure,
with 21 sensor readings per cycle. 20 engines were set aside and never
used to build anything; all results are on those.

- **A health score that needs no failure history.** It learns what
  "normal" looks like from the first 30 cycles of each engine. It warned
  about every held-out engine, typically about 38 cycles before failure.
- **A remaining-life forecast.** On NASA's official test engines it was
  out by about 16 cycles on average, against 38 for "every engine lasts
  the typical life".
- **A rule chosen by cost.** It assumes a breakdown costs 10 times a
  planned job and the team needs 10 cycles' notice. Under those
  assumptions, cost per engine was:

| Approach | Cost per engine | Breakdowns (of 20) |
|---|---|---|
| Wait for it to break | 10.0 | 20 |
| Service on a fixed schedule | 3.1 | 4 |
| Service when the health score is high | 2.4 | 3 |
| **Service when forecast life is low** | **1.5** | **1** |
| Perfect foresight | 1.0 | 0 |

- **Checked five ways.** The experiment was repeated with five different
  random sets of held-out engines. The forecast-based rule stayed the
  cheapest on this fleet, at about 1.6 per engine on average.
- **On a second fleet with two failure types**, the simpler health score
  was the cheapest (1.2 per engine, no breakdowns in any of the five
  repeats). The right rule depends on the equipment, so test it, do not
  assume it.

## What it means in practice

The same set-up fits test stations, pick-and-place machines, ovens,
pumps or fans: any equipment with sensors and a service history. A site
starts with the health score (it needs only normal running data), adds a
life forecast as failure records build up, and sets the alert by its own
costs of a breakdown, a planned job and the notice needed.

## What it does not show yet

- The engines are simulated; real machines are noisier and are usually
  serviced before they fail, so true lifetimes are rarer.
- 20 test engines is a small sample: one extra breakdown changes the
  average noticeably.
- Costs and notice period are assumptions, to be replaced with a site's
  own numbers.
