# Case study: routing customer enquiries on a trade sales desk

**Audience:** a sales manager, operations lead, or potential client — not a
data-science reviewer. The technical write-up, metrics tables and critical
evaluation are in [README.md](README.md).

**Status:** portfolio demonstration. The model was trained on *synthetic
example messages* (invented text, clearly labelled as such). It is not
connected to a live mailbox and is not a product in production.

---

## The business problem

On a B2B tyre and automotive-parts desk, the inbox is the front door.
Customers ask for prices, check stock, chase orders, raise complaints, and
query invoices — often in the same stream, written quickly, and needing
different people to act.

A sales administrator can do that routing well, but it is slow, uneven when
the desk is busy, and easy to get wrong when a complaint sits in the same
pile as a routine stock check. The cost is not only minutes per email. A
quote that waits too long is a lost order; a complaint that waits too long
is a lost account.

## What this demo shows

A simple, inspectable machine-learning model can read a message and
*suggest* one of six folders:

- Quote request
- Stock / availability
- Order status
- Complaint / return
- Account / billing
- Other (hours, catalogues, account setup, and similar)

You can try it yourself: paste a message into the Streamlit app
(`streamlit run app.py`) and you will see the suggested class and a
confidence score.

On a held-out set of 108 synthetic messages, the model used in the app
(TF-IDF + Logistic Regression) reached **96.3% accuracy** and a
**macro-F1 of 0.961**. A second model (LinearSVC) was slightly higher, at
**98.1% accuracy**. Both were far ahead of a naive "always pick the most
common class" baseline (22% accuracy). The four mistakes the app model
made were borderline cases — for example a late delivery that is both an
order chase and a complaint — and the model was *uncertain* on all of
them. That is useful: a real desk can auto-suggest only when the model is
confident, and leave the rest for a person.

## What this could be worth

Used carefully, a routing assistant can:

- **Cut time-to-first-action** on quotes and stock checks, especially
  outside the person who happens to open the inbox first.
- **Surface complaints and billing disputes faster**, instead of letting
  them sit under routine mail.
- **Give a new or covering administrator a head start** on a familiar
  folder structure, rather than reading every thread from scratch.
- **Produce a simple activity picture** over time (how many quote vs
  complaint messages arrived this week) once routing is consistent.

None of that requires replacing staff. The useful design is a suggestion
next to the message, with a person still accountable for the customer.

## What it is not, yet

The headline accuracy is measured on invented messages that are tidier
than a real inbox. Real customers mix topics, write one-line emails, and
forward long threads. Those figures would almost certainly come down on
live mail, and they should be re-measured there before anyone relies on
them.

A live deployment would also need a clear legal and data-protection
basis (customer emails are personal data under GDPR), a sample of
properly labelled real messages, a rule that uncertain or sensitive
mail always goes to a person, and someone responsible for checking that
the model still behaves as the business changes.

This case study is therefore an illustration of the *approach and the
potential*, with a working prototype you can click through. The next
conversation with a client would be about a small, lawful pilot on their
own desk — not about plugging this demo into a live inbox as-is.

## If you want to take this further

A practical pilot is short and bounded: agree the folder list with the
desk, label a few weeks of mail, compare the model against how staff
already route, and only then decide whether suggestions appear in the
day-to-day tools. The work already done here — class scheme, baseline
models, a confidence score, and a clear-eyed view of the limits — is
the starting point for that pilot, not a substitute for it.
