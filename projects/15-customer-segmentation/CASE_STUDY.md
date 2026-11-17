# Case study: which customers matter, and will the segments hold up?

**Audience:** a marketing, CRM or sales manager, not an analyst.
Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **real, public** data: a year of
transactions from a UK online gift retailer (UCI Online Retail, CC BY
4.0). Customer IDs are anonymous numbers.

---

## The business problem

Most businesses want to "segment their customers" so they can treat
their best customers differently, win back lapsed ones and welcome new
ones. The risk is a set of segments that look convincing on a slide but
change every time someone re-runs the analysis, and that say nothing
about what customers will do next.

## What this demo shows

- **Four plain-language segments** from 3,360 customers: VIP /
  wholesale-scale, Lapsed one-off, New this quarter, and a small
  High-cancellation group.
- **Where the money is:** the VIP segment is 37% of customers but 84% of
  spend. In the following three months it delivered 79% of revenue, and
  80% of VIPs bought again, against 42% of lapsed one-off buyers.
- **A small group worth a look:** 66 customers cancelled about 40% of
  the value they ordered. That is a service or fraud question, not a
  marketing one.
- **An honest test.** The segments were checked against the next three
  months, which the model never saw. A simple scoring rule (RFM:
  recency, frequency and money spent) predicted who would buy again
  better than the machine-learning segments.

## What it means in practice

- Use the segments to **describe** the customer base and design
  programmes: VIP retention, a second-order offer for new customers, a
  win-back test for lapsed buyers.
- Use the simple RFM score to **choose who to contact**, because it
  predicted next-quarter buying better.
- Treat segment edges as soft. The four-segment split is only moderately
  stable, so a customer near a boundary can move between segments.

## The same approach for advisers or partners

For a business that sells through financial advisers, brokers or
resellers, the identical method segments the advisers by how recently
and often they place business, its value, product range, tenure and
lapse rate. It then checks the segments against the next quarter. This
demo did not use adviser data; the method transfers directly.

## Limits

One retailer, one year (2010–2011), many wholesale buyers. The numbers
show the method, not what your customers will look like.
