---
name: chip-migration
description: Move a deployment to a different GPU when that GPU is measurably faster for one request, proving the model still answers the same way before any traffic follows it.
---

# Chip migration

Moving hardware is the one change in this system that can genuinely reduce how
long a person waits, because it changes the machine doing the work rather than
how the work is scheduled. That also makes it the change most likely to alter
the answer: different kernels and different reduction orders make bf16
arithmetic land differently, and a migration that changes the output has not
migrated anything.

## Propose from measurements, not from spec sheets

`src/gpushare/agent/gpus.py` carries a batch-1 median for every card anything
was actually run on, all from one gate so they compare to each other. Propose a
target only when both the current card and the candidate are `measured`.

A card with `measured: false` — the RTX 5090 today — has no number to offer.
It may still be worth trying, but the proposal must read as an estimate and
the screen must say `Estimated` until a run on it exists.

Higher TFLOPS is not a latency prediction. Batch-1 decoding is bound by memory
bandwidth and per-step overhead, so a card with far more compute can return one
short answer no sooner.

## Validate before, not after

Run the same fixed prompt set on the target before any traffic moves, and
compare against the source run:

- the task's own quality gate must pass on the target
- the outputs must be compared case by case, not summarised into one accuracy
- both halves of a split metric must hold; an aggregate hides a collapse in one

A checkpoint that trained on one vendor and resumed on another is the case
this repo has actually exercised: NVIDIA RTX 4090 to AMD MI300X held exact
match at 89.7% and 90.0% across 300 identical cases. It was also slower —
393s against 213s — which is the honest result and was reported as one.

## Latency and correctness are separate verdicts

Report them separately and let them disagree. "Faster and the same" is one
outcome; "faster but different" is a rejection; "same but slower" is a
migration that succeeded at portability and failed at performance, and saying
so is more useful than picking whichever number looked better.

## Traffic moves last

Keep the source deployment serving until the target has passed. Switching
traffic is a separate, explicit step, and the previous deployment stays
available until someone confirms the new one.
