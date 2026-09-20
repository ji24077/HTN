---
name: training-optimizer
description: Measure and select cost-efficient LoRA or QLoRA training configurations for a fixed model, dataset, and quality target.
---

# Training Optimization Agent

## Inputs

Require the base model and revision, immutable train and held-out hashes, target task, budget, deadline, candidate GPUs, and quality threshold. Record the reference precision, token count, and seed.

## Procedure

1. Filter GPUs by memory, availability, owner policy, data location, and approved spend.
2. On NVIDIA, benchmark LoRA and QLoRA; on the current AMD MVP path, benchmark LoRA only. Vary micro-batch, gradient accumulation, checkpointing, and LoRA rank while keeping tokens per step and total training steps fixed.
3. Reject out-of-memory runs, non-finite loss, incomplete checkpoints, and configurations that changed the fixed work.
4. Retrain the selected low-cost candidate for the reference step count and run the unchanged held-out evaluation.
5. Return measured step time, throughput, peak VRAM, estimated total cost, checkpoint locations, and every rejected candidate.

## Constraints and gate

Never start paid GPU work without spend approval. Do not select from projections alone. A candidate is usable only when JSON validity and exact match remain within the stated threshold and safety checks pass on the identical evaluation-set hash. Keep the prior checkpoint active when validation is missing or fails.
