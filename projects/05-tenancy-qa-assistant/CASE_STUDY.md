# Case study: a cited Q&A over official tenancy guidance

**Audience:** a landlord, letting agent or property manager — not a
data-science reviewer. The method, the retrieval scores and the
critical evaluation are in [README.md](README.md).

**Status:** portfolio demonstration on a **snapshot** of public
Citizens Information and RTB pages, fetched 14 November 2026.
This is **not legal advice**.

---

## The business problem

Tenancy rules in Ireland are public, but they are spread across
several official pages and they change. A typical question — how
large a deposit can be, when a tenancy must be registered, how a
notice of termination has to be served after 14 September 2026 —
should be answerable from those pages, with a link back to the
source, rather than from memory or a forum post.

## What this demo shows

The app (`streamlit run app.py`) takes a question and returns the
closest **quoted passages** from the snapshot, each with:

- the page title and a link to the official URL
- the date that page was fetched
- a similarity score

It does not write a new opinion, and it does not replace the RTB.
On a 16-question check written for this snapshot, the local MiniLM
retriever put a gold source page first every time (hit@1 = 1.00,
MRR = 1.00). A simpler TF-IDF baseline did almost as well (hit@1 =
0.94) but confused a *rent review* notice with a *notice of
termination* — a reminder that a keyword search can grab the wrong
“notice.”

Those scores are encouraging for a small, tidy corpus. They are
also optimistic: the questions were written after reading the
pages, and “the right page came back” is not the same as “this
settles your case.”

## What this could be worth

Used carefully, a cited reading aid can:

- point a landlord or tenant at the **official paragraph** instead
  of a social-media summary
- show **when the page was snapshotted**, so stale advice is harder
  to hide
- keep a human in the loop: the next click is the live RTB or
  Citizens Information page

It cannot decide whether a particular notice is valid, apply the
rules for tenancies that started before or after 1 March 2026, or
stand in for Threshold or a solicitor. The app says this at the top
on purpose.

## What we would do next with a client

Agree the source list (official pages only), refresh it on a
schedule, evaluate on *their* frequently asked questions rather than
on this author’s set, and treat the tool as a navigation aid with a
written “not legal advice” boundary — not as an automated
determination of rights.

The public snapshot and the Streamlit demo in this repository are
the starting point for that conversation.
