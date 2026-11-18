# Case study: catching circuit-board defects with fewer false alarms

**Audience:** a production, quality or engineering manager, not a data
scientist. Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on a **public research dataset**
(DeepPCB, MIT licence): real circuit-board images with defects **added
artificially** by the dataset's authors. It is not tested on a real line.

---

## The business problem

Electronics lines, including those building automotive control units,
check every board with a camera. Two failures matter. A **missed defect**
(an "escape") can reach a customer's vehicle. A **false alarm** sends a
good board to manual review, costing time, and too many of them teach
operators to ignore the machine. Many inspection rules simply compare
each board with a perfect "golden" image, which works until the camera
or the board shifts slightly.

## What this demo shows

Three approaches were scored on the same 4,640 image patches (3,140 with
a defect, 1,500 without):

| Approach | Missed defects | False alarms on good patches |
|---|---|---|
| Compare with the golden image | 45 | 90 (6.0%) |
| Learn "normal" from good boards only | 36 | 364 (24.3%) |
| **Trained deep-learning model** | **12** | **80 (5.3%)** |

- The trained model also **names the defect type** (open, short,
  mousebite, spur, stray copper, pin-hole) with 98% accuracy.
- It performed just as well on board image sets it had never seen.
- It **shows where it looked**: a heat map over each decision lets the
  reviewer check the call in a second.
- The "good boards only" method needs **no defect examples at all**. That
  makes it useful on day one of a new product, as a first screen.
- It all runs on an ordinary computer, with no special graphics card.

## What it means in practice

A sensible roll-out is staged. Start with the good-boards-only screen,
have reviewers label what it flags, then train the full model once
enough defects are collected. Throughout, track escapes and false calls
separately, and set the alarm level by what each one costs the line.

## What it does not show yet

- The defects in this dataset were drawn in, so they are cleaner than
  real ones. Real-line results would be lower.
- The boards are bare, without components or solder, where much real
  inspection effort goes.
- Scores are for small image squares, not for finding every defect on a
  full board.
