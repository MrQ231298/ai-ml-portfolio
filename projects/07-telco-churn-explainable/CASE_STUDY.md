# Case study: deciding which customers to save

**Audience:** a head of customer service, retention or finance — not a
data-science reviewer. Method, metrics and critical evaluation are in
[README.md](README.md).

**Status:** portfolio demonstration on IBM's public Telco Customer Churn
sample (7,043 customers). Realistic structure; not a live client's data.

---

## The business problem

About one customer in four in this dataset left. A retention offer
costs money, and offering it to everyone wastes most of it on people who
were going to stay anyway. The team needs a short answer each month:
**who should we call, and why?**

## What this demo shows

- A **risk score** for every customer that can be read as a real
  percentage (on unseen customers, a predicted 30% means roughly 30%).
- A **contact rule set from the money**, not from a rule of thumb. With
  an illustrative offer cost of 40, a 30% save rate and a year of
  revenue per saved customer, the best rule was "contact anyone above
  17% risk". On 1,409 unseen customers that rule produced about
  **60,000** of estimated net value, versus about **43,800** for the
  usual "above 50%" rule and **41,600** for calling everyone.
- **Reasons for each customer** (for example: month-to-month contract,
  new customer, no online-security add-on), so the agent knows what to
  offer.
- A **fairness check**: the model treats men and women alike, but is
  less accurate for older customers — something to discuss before use.

## What this could be worth

For a subscription business (telecoms, broadband, gyms, software,
insurance renewals), the same approach can focus a fixed retention
budget on the customers where it pays back, and make the trade-off
explicit: change the offer cost or the value of a customer and the
contact list updates.

## What we would do next with a client

1. Rebuild the features from the client's billing / CRM data, using
   only information available before each month's decision.
2. Run a small **randomised retention test** to measure the real save
   rate — the single most important number in the economics.
3. Agree the fairness and data-protection position with the DPO.
4. Deliver a monthly scored list plus the dashboard, and monitor that
   the probabilities stay calibrated.
