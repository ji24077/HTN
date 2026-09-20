# Relay MVP

Relay combines the existing authenticated compute network with a safety-gated GPU
marketplace and model workflow. It does not replace the main worker lease protocol:
the original scheduler still owns worker presence, task leases, retries, and signed
results. Relay adds marketplace offers, owner policies, optimization agents, quality
decisions, and human approval boundaries.

## Execution flow

```text
user goal
  -> Job Allocation Agent
  -> spend approval
  -> Training Optimization Agent
  -> Verification Agent
  -> Inference Optimization Agent
  -> migration approval
  -> Chip Migration Agent
  -> Verification Agent
  -> traffic approval
  -> deploy or restore the previous route
```

The control plane stores the selected offer, every rejected candidate, three approval
records, dispatched job IDs, verification evidence, deployment snapshots, and rollback
events in `.gpushare/relay-runs.json`. The file contains no provider credentials.

## Six independent agents

Each runtime role has a repository-owned contract:

- `skills/training-optimizer/SKILL.md`
- `skills/inference-optimizer/SKILL.md`
- `skills/chip-migration/SKILL.md`
- `skills/job-allocation/SKILL.md`
- `skills/quality-verifier/SKILL.md`
- `skills/provider-onboarding/SKILL.md`

The service fails closed when any contract is missing or malformed. The executable
training, inference, and migration agents reuse the existing measured RunPod runners.
Allocation, provider policy, verification, approvals, and rollback are implemented in
`src/gpushare/relay/`.

## Marketplace behavior

An offer includes price, chip and VRAM, workload profile, network bandwidth, latency,
trust score, failure rate, region, health, availability, and whether private data is
accepted. Allocation first applies hard gates and then exposes a weighted score; a low
price cannot hide a privacy, deadline, reliability, or owner-policy failure.

Owners can specify recurring local-time windows, including overnight windows such as
weekdays 19:00–07:00, a minimum hourly price, workload allowlist, public-data-only
operation, allowed regions, maximum runtime, and maximum GPU-memory fraction. Policy
and health are evaluated again before assignment.

Connected RunPod cards are marked `verified` and may execute after approval. Built-in
RTX 4090, A5000, L40S, and MI300X cards are marked `simulated`; they make the ranking
and strategy visible when no account is connected but cannot start a job. Marketplace
payments are not simulated: Relay records a maximum spend approval, while the provider
continues to own actual billing.

## Strategy boundary

Relay chooses frequent DDP/FSDP-style synchronization for local, high-bandwidth
training and DiLoCo-style local steps with periodic synchronization only for
cross-provider or slow-network training. Blender work uses independent frame
scheduling. Inference uses routing, batching, KV caching, runtime, and quantization
tests. DiLoCo is never presented as a rendering or inference optimization.

## Safety and quality gates

Reference and candidate reports must name the same evaluation-set hash. Verification
requires JSON validity, exact match, and an explicit safety result. Missing evidence,
a hash mismatch, a metric outside the predeclared threshold, or a safety failure is a
rejection. Candidate speed and price never override that result.

No paid execution starts without spend approval. No checkpoint moves without migration
approval. A deployment requires a passing verification report and separate traffic
approval. Before traffic switching, Relay snapshots the known-good route; a rollback
stops the candidate and starts restoration of that route when its model and Pod remain
available.

## Local API

Run `make ui`, then use:

- `GET /api/relay/capabilities` — agent contracts, MVP chips, strategies, approvals.
- `GET /api/relay/offers` — verified connected and simulated catalog cards.
- `POST /api/relay/plan` — parse a controlled natural-language goal and allocate it.
- `POST /api/relay/runs/{id}/approve` — approve or reject spend, migration, or traffic.
- `POST /api/relay/runs/{id}/execute` — dispatch an approved measured agent.
- `POST /api/relay/runs/{id}/verify` — record the independent quality decision.
- `POST /api/relay/runs/{id}/deployed` — confirm the running inference route.
- `POST /api/relay/runs/{id}/rollback` — stop the candidate and restore the snapshot.
- `POST /api/relay/providers` — persist a community GPU card and owner lease policy.
- `POST /api/relay/providers/check-policy` — validate a provider card and owner policy.

The MVP proof is observable: Relay can train a model, test alternative configurations
or chips, reject a faster result when its answers change, and permit deployment only
for the verified configuration.
