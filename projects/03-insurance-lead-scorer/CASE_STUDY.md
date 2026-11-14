# Case study: ranking commercial insurance leads

**Audience:** a sales manager or small broker principal — not a
data-science reviewer. The method, metrics and ethics discussion are
in [README.md](README.md).

**Status:** portfolio demonstration, trained only on *invented* leads.
This is not underwriting, not pricing, and not a recommendation to
offer or refuse cover. It must not be pointed at a live CRM as-is.

---

## The business problem

A commercial desk rarely has time to work every enquiry equally.
Broker referrals and existing clients often convert; a cold list
usually does not; a lead that sits until tomorrow is colder than one
called the same afternoon. The useful question is not "can a model
predict the future?" It is: *if we can only work the next 50 leads
properly, which 50?*

## What this demo shows

The app (`streamlit run app.py`) takes a few facts about a fictional
lead — source, trade, size, cover type, how quickly someone followed
up — and returns a conversion score. Behind the scenes two models
(logistic regression and gradient boosting) were trained on 3,200
invented leads, of which about **23%** converted, and tested on 640
held-out rows.

On that test, gradient boosting had a ROC-AUC of **0.643** and a
PR-AUC of **0.319** (the "no-skill" PR score is just the 22.7%
conversion rate). If the desk had called only the 50 highest-scoring
leads, **38%** of them would have been converters — better than
calling 50 at random, but a long way from certainty. The very top 25
did a little better (44%). At a longer list of 100, the simpler
logistic model actually edged ahead. The message for a manager is:
use the score as a *shortlist*, and judge it by how the shortlist
performs, not by a single "accuracy" number.

The scores themselves run a little *high* compared with real
conversion rates in the test file. That is a known effect of how the
models were trained to pay attention to the minority class. Treat the
percentage as a ranking aid, not as a promise.

## What this could be worth

Used carefully, a ranking tool can:

- help a small team **spend follow-up time where it is more likely
  to pay**, especially on a busy Monday inbox
- make **speed-to-lead** visible — in this invented world, slower
  follow-up really does hurt, and the model notices
- give a new or covering broker a **starting order**, not a
  substitute for judgement

It cannot decide who is insurable, and it should not quietly retire
whole regions or trades from the call list. In the test file, Dublin
leads were scored more generously than they converted. A live system
would need someone watching for that kind of drift.

## What we would do next with a client

Agree what "converted" means, label a few months of real (lawfully
held) leads, test the ranking on later weeks rather than on a random
shuffle, and only then decide whether a shortlist appears in the
day-to-day tools. The work already done here — a clear class
definition, imbalance-aware metrics, a calibration check, and an
explicit "not underwriting" boundary — is the starting point for that
pilot, not a substitute for it.
