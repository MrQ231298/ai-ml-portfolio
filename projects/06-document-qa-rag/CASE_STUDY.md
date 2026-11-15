# Case study: an assistant that answers from your own documents

**Audience:** an operations, HR or customer-service manager — not a
data-science reviewer. Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on a *fictional* company's
document pack (handbook, expenses policy, IT security, SLA, data
retention, health and safety, customer FAQ, onboarding guide).

---

## The business problem

Your team keeps asking the same questions, and the answers are buried
in PDFs and policy documents. Generic AI chatbots answer confidently
even when they are guessing. You need something that answers **from
your documents only**, shows **where the answer came from**, and says
**"I can't find that"** when the documents don't cover it.

## What this demo shows

- Type a question in everyday words ("Can I work from home?") and get
  the relevant sentence from the handbook, with the document and section
  named underneath.
- On 22 test questions it had not been tuned on, it gave a correct,
  cited answer to **16 of 18** questions that the documents do cover.
- It also showed where the honest limit is: it only declined **1 of 4**
  questions the documents don't answer. That part needs more work and
  more examples before a live rollout, and the report says so.
- It runs on an ordinary laptop with free, open models. No data is sent
  to an outside AI company unless you choose to connect one.

## What this could be worth

- Fewer repeat questions to HR, finance and the service desk.
- Faster onboarding: new starters can self-serve policy questions.
- Customer-facing FAQ answers that always match the current terms.
- A clear audit trail: every answer points to a source someone owns.

## What we would do next with a client

1. Load a copy of your real documents (on your own machine or tenant).
2. Collect 50–100 real questions from your inbox or helpdesk,
   including ones the documents *can't* answer.
3. Measure accuracy and "I don't know" behaviour on those questions
   before anyone relies on it.
4. Optionally connect an approved LLM for more natural answers, with a
   faithfulness check.
