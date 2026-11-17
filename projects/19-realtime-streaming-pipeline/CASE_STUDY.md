# Case study: live numbers you can trust, even after a crash

**Audience:** an operations, finance or product manager, not an engineer.
Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **real, public** data: two days of
New York taxi trips (January 2024), replayed as a live feed. The delays
and duplicate messages in the feed were added on purpose, at known rates,
so the results could be checked.

---

## The business problem

Many teams now want live dashboards: orders per minute, machines running,
trips per area. Live data is messy. Messages arrive out of order, some
arrive twice, and the software that counts them sometimes restarts. Each
of these quietly changes the totals, and the dashboard still looks fine.

## What this demo shows

- **A live feed of about 170,000 real taxi trips**, counted per borough
  in 5-minute slots, with a dashboard that updates every two seconds.
- **The counts are checked against a careful after-the-fact calculation.**
  All 2,246 time slots matched exactly, to the cent, on $4.78 million of
  fares.
- **The counter was deliberately crashed part-way through.** It restarted
  and still matched exactly: nothing lost, nothing counted twice.
- **The shortcut was tested too.** A simpler, common design, crashed at the
  same point, over-reported trips by **a third** (221,297 instead of
  165,673) and revenue by $1.55 million.
- **Late arrivals are listed, not lost.** 2,554 trips arrived too late for
  their slot. They were set aside in a separate list, so someone can
  decide what to do with them.
- **Speed:** a message was usually counted about 57 milliseconds after it
  was sent, and saved permanently within about one second.

## What it means in practice

The same approach fits any live count that feeds decisions, such as
production output per line, sales per region, or sensor readings per
machine. Before a live figure is trusted, it should be reconciled with a
batch calculation, and the reconciliation should survive a restart.

## What it does not show yet

- It ran on one machine with one copy of the data. A production set-up
  would keep three copies across three servers.
- One design choice sent almost all messages through one channel, so it
  would not yet speed up by adding machines. The fix is known and listed
  in the README.
- The crash was a forced stop of the program, at one point in the feed,
  not a power cut or a network failure.
