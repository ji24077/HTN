---
name: training-optimizer
description: Find the fastest training configuration that fits the selected GPU, holding the work per step constant, and refuse to call a selection validated until it has been retrained and scored.
---

# Training optimizer

Pick the configuration that finishes the same training sooner. Every part of
that sentence carries weight: *the same* training, and *finishes*, not *starts
faster*.

## Hold the work constant

Candidates differ in micro-batch and gradient accumulation, and their product
must stay fixed so every candidate does the same work per optimizer step.
Without that, the fastest step time belongs to whichever candidate did the
least, and the comparison measures the arithmetic rather than the hardware.

Read the token count back out of the measurements and check it agrees across
candidates. Do not assume the flags produced what they were supposed to.

## Fit the card before timing it

`src/gpushare/agent/gpus.py` carries VRAM and what each card has actually
held. A configuration that OOMs halfway is not a slow candidate, it is an
absent one, and reporting it as a loss misattributes the result.

Refuse to train on a GPU that is already serving. This is not politeness about
resources: the resident model holds most of the memory, the run dies partway,
and the failure reads as a bad configuration.

## More steps is not more quality

Doubling the step count on this repo's own 6-7 task moved held-out trigger
accuracy from 0.72 **down** to 0.59 — the answers stayed fluent and unique
while the rule was forgotten. A step count is a hyperparameter to be measured
like any other, not a dial that only goes one way.

## Not validated until retrained and scored

A speed benchmark says nothing about whether the model is still right, and
"same tokens per step, finite loss" is not evidence that it is. Retrain at the
reference step count with the chosen configuration and score it on the same
held-out set.

Until there is an eval to compare, report `not_validated`. Never `ok`. A
dashboard that calls an unvalidated change validated is worse than one that
admits it does not know.

## Compare against the same task

The reference run is whichever training ran last, which says nothing about
which task it belonged to. Scoring a run against another task's baseline reads
missing metrics as zeros and produces a confident verdict from nowhere. When
the two runs do not report the same metrics, say `not_comparable` and name the
metrics — do not fill them in.
