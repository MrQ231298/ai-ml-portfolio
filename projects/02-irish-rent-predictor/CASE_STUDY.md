# Case study: estimating official average rents in Ireland

**Audience:** a landlord, letting agent, or local business — not a
data-science reviewer. The method, metrics and critical evaluation are
in [README.md](README.md).

**Status:** portfolio demonstration. The figures are estimates of
*published RTB average monthly rents* (CSO table RIA02, CC BY 4.0).
This is not a property valuation and not letting, financial or legal
advice.

---

## The business problem

People who let or rent a home in Ireland often want a grounded sense of
"what does a two-bed apartment in this area usually go for?" The
Residential Tenancies Board already publishes official averages, but
they sit in a large statistical table rather than in a simple lookup.

A useful tool would let someone pick a town or postal district, a
property type and a bedroom count, and see a euro figure that is
clearly labelled as an official *average*, with a honest sense of how
far out it can be.

## What this demo shows

The app (`streamlit run app.py`) does that lookup with three models
behind it:

- a simple **persistence** rule: use the last published average for
  the same location, type and size
- a **log-linear regression**
- a **gradient-boosting** model

They were trained on official averages from 2008 to 2023 and then
tested on the published 2024 and 2025 figures — a fairer test than
shuffling rows at random, because in real life we are always estimating
a year we have not fully seen yet.

On that 2024–2025 test (1,065 published cells) the simple persistence
rule was the most accurate, with a mean absolute error of **€161**.
Gradient boosting came next (€172) and the linear model last (€198).
All three were, on average, a little low — rents were still rising —
and the gap was wider in 2025 than in 2024.

That result is useful in itself. It says that, for official averages,
last year's published figure is already a strong starting point, and
that a more complicated model is not automatically a better one. The
app therefore shows the persistence estimate as the headline, and
still lists the other two so you can see they are in the same region.

## What this could be worth

Used carefully, a tool like this can:

- give a landlord or agent a **quick, sourced starting point** for
  an area average, with the RTB and CSO credited
- make the official table **usable in a conversation**, rather than
  only in a spreadsheet
- show a **typical error of around €160 a month** on recent years,
  so nobody treats the figure as exact

It cannot tell you what a particular house should list at, and it
should not be used as a valuation or as advice to a tenant or
landlord about what to charge or offer.

## What we would do next with a client

The largest share of the remaining error looks like *drift*: averages
kept rising after 2023. A next version would try a simple,
pre-agreed inflation of the last published figure (a quick internal
check suggested this may cut the error substantially) and would be
re-tested each time the CSO table is updated. The first version
already does the important things: use real official data, credit it
properly, test on later years, and not pretend that a machine-learning
model has beaten a straightforward baseline when it has not.
