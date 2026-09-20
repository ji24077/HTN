---
name: quality-verifier
description: Decide whether a model still behaves, using the task's own metrics, on evidence that exists — and say "I do not know" rather than inventing a verdict.
---

# Quality verifier

Every action in this system — move the chip, change the configuration, swap
the serving runtime — has to be followed by an answer to "is the model still
right?". This is the component that answers it, and its most important
capability is refusing to.

## Use the task's own metrics

Tasks here do not share a scoring vocabulary. One reports `json_parse_rate`
and `exact_match_rate`; another reports `trigger_accuracy` and
`non_trigger_accuracy`. A delta computed over a key the task never wrote reads
as an unchanged zero and passes.

Take the metric list from the task. If either run is missing one of them, the
answer is `not_comparable` and the missing keys get named. A missing metric is
not a zero.

## One number hides the failure that matters

Report split metrics as their halves. A model that stamps its marker on every
answer scores 100% on the triggering half and 0% on the rest, and a single
accuracy calls that a respectable 50%.

Count the directional failures separately: marker present when it should be
absent, and absent when it should be present. They are different defects and
get fixed differently.

Per-row checks cannot see a model that collapsed to one reply, because each
individual row looks fine. Count distinct answers across the set. Two models
scoring identically can show 1 and 200 there.

## Tolerance is a policy, not a floating-point accident

`0.98 - 1.00` is `-0.020000000000000018`. A two-point tolerance that rejects
that is not implementing the policy, it is implementing the policy's binary
representation, and real runs land exactly on that boundary. Compare against
the tolerance with an epsilon.

## Measured or Estimated

A number is `Measured` only when it came from this deployment, this GPU, this
model. Everything else is `Estimated` and says so. Carrying a figure from
another card and presenting it as a result is the failure this whole component
exists to prevent.

## Fail closed

When evidence is missing, absent, or from the wrong task, the verdict is not
a pass. Say which evidence is missing and what run would supply it. A sentence
somebody can act on beats a verdict that happens to be wrong.
