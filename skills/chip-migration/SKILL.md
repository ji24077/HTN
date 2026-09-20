---
name: chip-migration
description: Evaluate and execute NVIDIA, AMD, or provider migrations only when measured cost or speed improves without changing model quality.
---

# Chip Migration Agent

## Inputs

Require the current immutable artifact, current deployment, candidate GPUs/providers, evaluation-set hash, quality threshold, budget, and rollback target.

## Procedure

1. Run the unchanged evaluation on the current deployment and record latency, cost, VRAM, exact match, JSON validity, and safety.
2. Copy the vendor-neutral checkpoint to an approved candidate. Build the correct CUDA or ROCm environment and reject silent CPU fallback.
3. Run the identical evaluation and compare all recorded metrics.
4. Recommend migration only when it improves cost or speed and passes every quality gate.
5. Request explicit migration approval before copying artifacts or spending on the candidate, then request separate traffic approval before route switching.

## Constraints and rollback

Never infer compatibility from a successful copy or compile. A missing or different evaluation-set hash is a rejection. If validation fails before traffic changes, keep the current deployment. If it fails after a candidate was applied, restore the recorded previous route and report the rollback.
