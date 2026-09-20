---
name: inference-optimizer
description: Make a deployed model answer sooner without changing what it answers. Measure batch-1 latency including the tail, propose only optimizations validated on the selected GPU, and keep the previous configuration for rollback.
---

# Inference optimizer

The question is how long one person waits for one answer. That is not the same
question as how many requests per second the deployment can serve, and the two
have opposite answers on the same hardware: batching raises throughput while
making each individual reply arrive later. Report the one you measured, under
its own name.

## Record the baseline first

Fix the prompt set, the token budget, and the decoding settings, then leave
them fixed for every candidate. A comparison where the two sides generated
different amounts of text is not a comparison, so pin `min_new_tokens` to the
same value as `max_new_tokens` and decode greedily.

Record, per configuration:

- median latency, p95 latency, time to first token
- output tokens per second
- the exact outputs, for the quality check
- warmup or compile time, kept **separate** from the per-request number

Warmup belongs outside the measurement because it is paid once at deploy.
Folding it into the first request understates a real win; hiding it entirely
misrepresents what switching costs.

## Propose only what this GPU has validated

`src/gpushare/agent/gpus.py` records which optimizations were accepted on each
card and, more usefully, which were refused and why. Read it before proposing
anything. A card being NVIDIA does not mean an optimization measured on
another NVIDIA card holds here.

Never propose an optimization listed in that card's `rejected` map without
new measurements that contradict the recorded reason.

## The tail is part of the verdict

An optimization that improves the median and ruins p95 has made the service
worse, and a person using it will notice the stall long before they notice the
average. Measured on an RTX 4090, `torch.compile` with CUDA graphs moved the
median from 0.130s to 0.115s and p95 from 0.132s to **1.507s** — worse than
the 0.404s of no optimization at all. It is rejected there for that reason.

Accept a candidate only when **all** hold:

1. output quality passes the task's own gate
2. median improves
3. p95 does not regress
4. the measurements come from this deployment, not another one

Otherwise reject it and say which of the four failed.

## Measured and Estimated are different words

Label a number `Measured` only when it was collected from this deployment, on
this GPU, with this model. Anything carried over from another card, another
model, or a published figure is `Estimated` and must say so on screen.

An estimate is allowed to justify *trying* something. It is never allowed to
stand as the result.

## Keep the way back

Retain the previous serving configuration until the new one has passed. A
rollback that requires retraining or redeploying from scratch is not a
rollback.
