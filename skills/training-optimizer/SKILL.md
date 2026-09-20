---
name: training-optimizer
description: Plan and verify Qwen 4B QLoRA training on approved RunPod GPUs without changing the task, fixed work, or quality gate.
---

# Training Optimization Agent

## Allowed inputs

- An existing Relay workspace and immutable Qwen 4B base-version identifier.
- The requested child-version name, LoRA target modules, rank, alpha, dropout,
  learning rate, step limit, seed, and checkpoint cadence.
- Immutable train and held-out dataset hashes for the configured `67` emoji
  behavior, with trigger and non-trigger examples represented separately.
- Candidate GPUs already present in the user's RunPod fleet, their measured
  memory/health/price, a maximum budget, and an explicit compute approval.
- Prior verified Compute Memory records. Unmeasured records may inform a trial
  but cannot decide a winner.

## Allowed actions

- Validate that the selected model is the repository's tested decoder-only
  Qwen 4B model; never silently substitute another model.
- Construct QLoRA/4-bit candidates on the approved RTX 4090 training pod and
  vary micro-batch, gradient accumulation, checkpointing, and LoRA settings
  while keeping work per optimizer step fixed.
- Start work only through deterministic backend jobs. The agent cannot issue
  shell/SSH commands, provision or terminate pods, or perform full fine-tuning.
- Register a `lora_adapter` version only after the artifact and evaluation
  evidence have been verified by backend code.

## Required measurements

- Effective tokens per optimizer step and total trained tokens.
- Step-time samples, median step time, examples/tokens per second, peak
  allocated/reserved VRAM, finite loss, completed steps, and checkpoint hashes.
- Held-out trigger accuracy (`67` requires the configured emoji), non-trigger
  accuracy (no unsolicited emoji), directional failure counts, distinct-output
  count, and any task-specific exact-match score.
- GPU, vendor, runtime/library versions, region, hourly price, elapsed time,
  estimated total cost, errors, and evidence provenance (`Measured` or
  `Estimated`).

Candidates with different tokens per step did different work and are not
comparable. Read the token count from the run instead of trusting input flags.
A faster partial run is not validated until the chosen configuration is
retrained for the reference work and scored on the same held-out hash.

## Quality and cost constraints

- Use QLoRA/4-bit training for this MVP, never full-weight fine-tuning.
- Reject OOM, non-finite loss, incomplete/corrupt adapters, changed dataset or
  evaluation hashes, missing split metrics, and configurations that changed
  fixed work.
- More steps are not assumed to improve quality. Compare them as a measured
  hyperparameter; prior repository evidence found longer training could reduce
  trigger accuracy.
- Do not schedule training on a GPU currently serving a live workspace unless
  deterministic capacity checks and the user explicitly permit it.
- A budget estimate may justify a proposal, but only a completed measured run
  can be selected. Never label an estimate `Measured`.

## Approval requirements

- Explicit user compute/spend approval is required before any remote setup,
  benchmark, training, artifact copy, or paid RunPod work.
- Registering a verified adapter does not authorize deployment. Serving it and
  switching a workspace deployment require separate explicit approval.
- Creating or terminating a RunPod pod is outside this skill and always needs
  an independently confirmed operation.

## Rollback behavior

- Keep the parent version and its live deployment unchanged during training.
- On interruption or failed verification, mark the run failed/pending as
  appropriate, preserve scrubbed evidence, and do not register a usable version.
- If a newly served adapter later fails its fixed suite, stop the candidate
  deployment and restore the recorded parent deployment; never retrain as the
  rollback mechanism.
