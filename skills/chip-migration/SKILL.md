---
name: chip-migration
description: Evaluate CUDA-to-ROCm portability for the same Qwen 4B version and recommend migration only from comparable measured quality, latency, and cost evidence.
---

# Chip Migration Agent

## Allowed inputs

- A live CUDA source deployment, immutable Qwen 4B base artifact and optional
  LoRA adapter hashes, and the approved AMD MI300X RunPod candidate.
- Source/target GPU identity, vendor, runtime/library versions, region, price,
  health/capacity, artifact transport location, and compatibility evidence.
- One immutable held-out suite, decoding configuration, output limit, quality
  tolerances, budget, and explicit compute/migration/traffic approvals.

## Allowed actions

- Ask deterministic code to validate CUDA/ROCm runtime and model/adapter
  compatibility before artifact transfer or execution.
- Copy or expose the same approved artifact and adapter to the MI300X candidate,
  reject silent CPU fallback, and run the identical fixed quality/benchmark suite.
- Produce one recommendation: `Recommend`, `Reject`, or `Needs review`.
- The agent cannot provision/terminate pods, issue shell/SSH commands, change
  quality thresholds, or switch traffic.

## Required measurements

- Artifact/adapter and evaluation hashes; actual GPU/backend execution evidence;
  PyTorch/ROCm/CUDA/Transformers/PEFT versions; input/output token counts; and
  identical decoding/output limits.
- JSON validity where applicable and exact per-case output comparison. For a
  registered LoRA version, also measure its bound `67` trigger/non-trigger
  behavior and distinct outputs. A base version has no `67` contract and is
  judged by exact output parity instead.
- TTFT, median and p95 end-to-end latency, throughput, peak VRAM, errors, region,
  hourly price, and comparable cost per workload.
- Report portability, latency, and cost as three separate outcomes. A portable
  MI300X deployment may still be rejected for interactive batch-one inference.

## Quality and cost constraints

- First validate runtime and artifact compatibility; a successful file copy or
  compile is not execution proof.
- Both sides must use the same immutable evaluation set. Missing/different hashes,
  CPU/reference fallback, missing split metrics, or quality beyond tolerance are
  rejection conditions.
- Never infer speed from TFLOPS or vendor. Never claim AMD is faster without a
  measured benchmark proving it for this workload.
- A slower but quality-preserving target is `portability: passed` and
  `performance recommendation: rejected`; do not collapse those verdicts.

## Approval requirements

- Explicit compute/migration approval is required before setup, artifact transfer,
  or testing on the MI300X. Existing pod availability is not spending approval.
- A verified recommendation does not authorize traffic movement. A separate
  explicit user approval is required for rollout.
- New pod creation or termination remains outside the skill and needs its own
  confirmation.

## Rollback behavior

- Keep the CUDA source deployment live and unchanged while the ROCm candidate is
  tested. Failure before rollout only rejects/stops the candidate.
- If post-rollout verification fails, deterministic code restores the stored
  source route and records the candidate, failure evidence, and user decision.
- Never delete the source artifact or adapter until the user confirms the new
  deployment and the rollback window has ended.
