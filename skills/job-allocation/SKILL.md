---
name: job-allocation
description: Match training, inference, and rendering jobs to managed or community GPUs under cost, deadline, reliability, privacy, and location constraints.
---

# Job Allocation Agent

## Inputs

Require workload type and size, budget, deadline, VRAM, data classification and location, plus provider price, availability, benchmark profile, network, trust score, failure rate, and policy.

## Procedure

1. Reject unavailable, unhealthy, policy-incompatible, under-memory, over-budget, late, low-trust, or wrong-region offers.
2. Estimate completion time and total cost from the workload-specific measured profile.
3. Rank eligible offers using explicit cost, speed, network, latency, trust, and failure-rate components. Return the complete ranked and rejected lists.
4. Choose DDP/FSDP for a local high-bandwidth training cluster, DiLoCo only for cross-provider or slow-network training, independent frame scheduling for Blender, and routing/batching/cache optimization for inference.

## Constraints

Label catalog estimates as simulated and connected measured resources as verified. A simulated card may produce a plan but cannot execute. Allocation never authorizes spending; request user approval for the selected maximum cost.
