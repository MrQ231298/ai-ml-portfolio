# Case study: what are customers really complaining about?

**Audience:** an e-commerce, product or customer-experience manager —
not a data-science reviewer. Method and full results are in
[README.md](README.md).

**Status:** portfolio demonstration on **22,641 real, public, anonymised**
clothing reviews (CC0 licence).

---

## The business problem

You have thousands of reviews and comments. Reading them all is
impossible, and the average star rating does not tell you what to fix.

## What this demo shows

- **Automatic mood detection:** separates clearly unhappy from clearly
  happy reviews with very high reliability (0.97 on a 0–1 ranking
  score); the in-between "it's OK" reviews are harder, and we say so.
- **Themes found automatically** — 15 recurring topics such as sizing,
  fabric quality, colour, comfort, jeans, petite lengths.
- **A ranked complaint list:** reviews about *fabric and quality* were
  **2.3 times** as likely to be 1–2 stars as the average review;
  reviews about *sizing running small or large* were 1.2 times as likely.
  Reviews about comfort and a perfect fit were rarely negative.
- **An automatic report** with the numbers and a real example review for
  each theme, regenerated with one command.

## What this could be worth

- Product teams see which issues cost the most goodwill and can check
  whether a fix (a new size guide, a fabric change) actually worked.
- Customer service can prioritise negative messages automatically.
- Works on any free text: reviews, survey comments, support emails.

## What we would do next with a client

1. Load your reviews, survey or support text (anonymised where possible).
2. Hand-check a sample of 200 to confirm the mood labels match your view.
3. Name the themes with your team and set up a monthly report.
