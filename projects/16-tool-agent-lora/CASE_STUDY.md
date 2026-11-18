# Case study: a small, private AI assistant that looks up real answers

**Audience:** an operations or sales manager, not an engineer. Method
and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **invented** data: a fictional
Irish electronics distributor with 240 customers, about 1,000 orders and
six policy documents.

---

## The business problem

Staff spend time on questions like "how many orders did Galway place in
May?", "what do 25 connectors cost with VAT?" or "what's our returns
window?". General chatbots answer confidently but make numbers up, and
sending company data to an outside AI service raises cost and privacy
questions.

## What this demo shows

- **An assistant that uses the company's own data.** It doesn't guess:
  it writes a database query or a calculation, runs it, and shows its
  working.
- **It runs on an ordinary computer.** The AI model is small enough to
  run without a graphics card or an outside service, so data stays
  in-house.
- **Teaching it the job made the difference.** Out of the box, the small
  model got **none** of the test questions right, and only about 1 in 7
  even when shown examples. After a short training run (about 30 minutes
  on a normal processor, adjusting under 1% of the model), it answered
  **39 of 40** standard questions correctly.
- **It mostly copes with new wording.** On questions phrased in ways it
  had never seen, about places and products it had not been trained on,
  it got **34 of 40** right.
- **It cannot damage anything.** The database connection is read-only,
  whatever the assistant writes.

## What it means in practice

A business can have a private assistant for routine look-ups at very low
running cost. It should start with a well-defined set of question types,
show its working so staff can check it, and widen the set as it is
tested on real staff questions.

## What it does not show yet

- The test questions came from templates. Real staff will phrase things
  more freely, so expect lower accuracy at first.
- The scores come from 80 test questions, a small sample.
- It was not compared with large commercial AI services.
