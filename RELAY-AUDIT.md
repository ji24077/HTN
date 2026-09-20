# Relay audit and demo revision

Audit date: 19 September 2026. Planning constraint: under six hours before the demo.

This report preserves the initial audit snapshot. For the current companion
branches and the portable workspace setup, see [RELAY-DEMO.md](RELAY-DEMO.md):
`relay/demo-fixes` for the frontend, `relay/gpu-safety` for GPUShare, and
`relay/fleet-safety` for the fleet backend. Paths such as `ji-review/` below are
checkout names in that workspace layout, not directories inside this branch.

**Implementation update:** The follow-up work now fixes verification labels, strict GPU comparisons, rejected-checkpoint promotion, provider admission/retry bugs, and frontend/API demo wiring. A recorded-evidence demo and revised presenter script are available in [RELAY-DEMO.md](RELAY-DEMO.md). Findings below describe the pre-fix audit snapshot; automatic rollout, marketplace payments and general GPU execution remain outside this patch.

## Verdict

Relay has substantial working components and credible saved GPU measurements, but the proposed overview describes an integrated product that does not yet exist. The strongest immediate story is **measured GPU optimization with explicit rejection when answers change**. Marketplace economics, six independent skill-based agents, and approval-gated deployment/rollback remain incomplete.

Keep the GPU story. Do not conclude it is absent merely because current `main` is primarily a fleet and CPU simulation platform. The relevant work is split across branches.

## Scope and branch map

| Snapshot inspected | What it contains |
|---|---|
| `HTN/HTN`, `main`, `6cf1dcd` | Fleet control plane, React dashboard, machine enrollment, Docker agents, CPU simulation planning/validation, job supervision |
| `ji-review`, `phin/handoff-gpu-benchmarks`, `29543d3` | GPUShare training/inference/migration backend, saved GPU campaigns, strict comparison utilities; local dashboard edits and untracked examples also exist |
| `FE-WEB`, `42b6427` | GPU Lab frontend connected to the separate GPUShare API; absent from current main |
| `FE-DESKTOP`, `4253d69` | Desktop-agent visual redesign |
| `backup/main-before-sync-20260919-210207`, `161561b` | Former local main, containing GPUShare history and newer serving serialization/policy-reset fixes |

These are locally available snapshots, not a claim that all remote branches were refreshed. No branches were switched or merged during this audit. Current main and GPUShare have separate histories; do not merge entire trees merely to put everything on one branch. Select a demo frontend/backend pair and port bounded changes deliberately. Preserve the uncommitted GPUShare work.

## Capability matrix

| Proposed capability | Actual status | Demo decision |
|---|---|---|
| Qwen fine-tuning | Implemented in GPUShare; full and LoRA modes, saved adapters and measurements | Use one pinned Qwen2.5-0.5B LoRA workflow |
| QLoRA | No implemented workflow found | Remove from current capability list |
| Training optimization | Parameter-selection/sweep code exists; documentation includes a failed optimization run | Do not call the optimizer demonstrated without one saved successful run |
| Inference optimization | Real static-KV-cache/compilation comparisons and separate prefix-cache experiments | Pick one checkpoint and one experiment; attach conditions to numbers |
| NVIDIA/AMD execution | Saved real NVIDIA and MI300X results exist | Say tested configurations, not universal chip support |
| Chip migration | Checkpoint transfer/resume and quality checks exist separately from strict benchmark comparison | Distinguish execution, acceptance and traffic promotion |
| Verification | Strong strict per-case comparator exists; live dashboard uses a weaker aggregate gate | Unify policy or show strict offline decisions honestly |
| Job allocation | Main: compatibility/FIFO and measured CPU simulation placement. GPUShare: static chip catalog, capacity filtering, hourly-price sorting and resource/cost-model work | No demonstrated combined cost/deadline/trust/privacy optimizer |
| Provider onboarding | Invites, outbound connections, local controls, health, history | Controlled compute sharing foundation |
| Marketplace | No integrated listings/bookings/payments/provider earnings/trust/location/privacy system | Roadmap; label any cards as simulated or measured examples |
| Spending approvals | Bounded rental/session tooling exists, but no unified product approval lifecycle | No automatic under-$20 promise |
| Deploy/rollback | Serving exists; no verified-candidate/approval binding or automatic old-service restoration found | Demonstrate reject-and-retain, not safe automatic rollout |
| Six independent `SKILL.md` agents | Main has none; GPUShare has `skills/portability/SKILL.md` | Describe actual components; roles may remain roadmap |
| Blender rendering | No executor found | Future workflow only |
| DiLoCo | Synchronization-interval formula exists in GPUShare; distributed trainer still raises “not implemented” | Future research direction only |

## The most important correctness gaps

### 1. The live migration gate contradicts “quality unchanged”

`ji-review/src/gpushare/dashboard/runner.py:950` accepts decreases of up to `0.02` in JSON parse rate, exact match and individual field accuracy. `docs/handoff-ji.md` records a migration from 83.5% to 82.5% exact match accepted under this policy. That is within a tolerance, not unchanged quality.

In contrast, `ji-review/src/gpushare/agent/prediction_validation.py:66` requires every indexed JSON result to remain valid and unchanged. This stronger gate produced the hardware-matrix decisions. It is not the gate used by the live dashboard migration path.

Recommendation: choose and show one explicit policy. For this pitch, retain strict value preservation against a frozen fine-tuned baseline. Aggregate accuracy alone can conceal which individual answers changed. Include checkpoint hash, suite hash, runtime/precision and GPU configuration in each decision.

Training and migration need different reference points: training intentionally changes model behavior to improve the task; subsequent optimization/migration should preserve the agreed **fine-tuned** baseline. Do not demand identical answers between the untrained and fine-tuned models.

### 2. Job completion is currently presented as verification success

`FE-WEB:frontend/src/components/GpuLab.tsx:75` says completed work “was verified automatically” without checking `result.validation.status`. Change copy to reflect actual `passed`, `rejected`, `not validated`, or `failed` outcomes. A finished process is not a passed evaluation.

The frontend also contains fixed explanatory strings about workload analysis. Do not present these strings as evidence of an autonomous agent reasoning about that particular input.

### 3. Serving is not a verified rollout transaction

`ji-review/src/gpushare/dashboard/app.py:478` accepts serve requests without binding them to a passed evaluation or approval. `runner.py:1783` stops the current service before setting up the replacement. No automatic restoration of the old service was found. `runner.py:1131` evaluates training, but `1141` writes the latest checkpoint regardless of that result.

Minimum safe demo behavior: evaluation/rejection cannot change the active baseline. Explicitly select an accepted artifact for any serving action. Test a rejected candidate and a failed candidate startup. If preservation/restoration is not implemented and tested, remove “deployed only verified configurations” and “rollback” from completed-feature claims.

### 4. The frontend/backend combination needs a rehearsal

`FE-WEB:frontend/src/api/gpulab.ts:1` targets a separate GPUShare process via `VITE_GPUSHARE_URL`, defaulting to `http://127.0.0.1:8080`. Both GPUShare and the main orchestrator default to port 8080. Configure distinct ports and compatible HTTP/HTTPS origins. FE-WEB has no same-origin `/api` proxy; GPUShare CORS currently permits localhost origins. This is a local demo bridge, not shared hosted authentication.

The FE-WEB migration action invokes `migrate-nextgen`; it does not expose the newer NVIDIA-to-AMD action. Verify the exact frontend button invokes the intended backend experiment.

### 5. The evidence refers to different checkpoints and suites

The 11-chip matrix uses the frozen experimental **v3b** adapter. The default extraction path uses **v2**; the dashboard's latest training run can be a third artifact. Do not present a benchmark card as describing whichever model is currently serving.

**v3b itself failed an earlier training field-regression gate.** Preserving its outputs during an inference optimization does not make it an approved upgrade from v2. Label this an experimental checkpoint comparison. Accepted A5000/L40S runs preserve 270/300 correct records, not 300/300 correct answers. The 300 cases are reused regression data, not a fresh blind test. See `ji-review/demo/HANDOFF.md:90` and `docs/optimization-v3.md`.

The earlier 13-sentence experiment accepted configurations that the later 300-case check rejected. Use the broader frozen suite for final decisions. The current `data/heldout.jsonl` no longer reproduces the first 200 cases in the frozen suite manifest; use the committed `demo/validation/cases300.jsonl` directly. Its bytes were verified. No normalized exact sentence overlap was found with the current 11 training JSONL files, but that is not proof of generalization.

Loose current `eval/base.json` and `eval/after.json` have different case counts. Do not construct a before/after improvement claim by comparing them. Recover a paired run on the same cases and same checkpoint lineage, or omit that number.

### 6. Provider controls are useful but not yet a safe public leasing product

Main implements workload allowlists, weekday/overnight availability, rolling compute-time quotas, CPU/memory/battery checks and pause. These are admission controls, not monetary quotas or enforced data-privacy rules.

Static code tracing found three bugs/edge cases to address before demonstrating policy enforcement:

- Out-of-hours/resource-limit rejection can consume retries and immediately re-offer the same task: `packages/agent/src/transport.ts:560`, `backend/src/orchestrator/server/dwp.py:659`, and `server/db/store.py:1043`. A temporarily ineligible worker should not burn execution attempts.
- Empty workload selection falls back to all adapters (`packages/agent/src/limits.ts:116`, `gui.ts:824`). A deny-all selection must remain deny-all or explicitly pause.
- Overnight weekday windows check the current day, so Friday night can end at Saturday midnight (`limits.ts:101`). Define days as window-start days and expose the actual timezone.

These are code-review findings, not newly reproduced end-to-end failures. Long-running tasks can also exceed admission-time limits. Avoid promising continuous enforcement.

Uploaded Python uses a documented local development subprocess runner, not an untrusted-code sandbox (`backend/src/orchestrator/worker/executors/python_project.py:1`). Keep public community execution and sensitive workloads out of the current demo claim.

## What the saved GPU evidence actually establishes

Source: `ji-review/demo/results/hardware-matrix-2026-09-19/README.md` and its frozen raw artifacts.

| Finding | Supported statement |
|---|---|
| 11 completed GPU-model measurements | More chip coverage than the proposed 2–4-chip MVP; RTX 5090 attempt remained incomplete |
| A5000 | 1.383 s to 0.234 s median resident response; 5.91x ratio; no changed JSON values on 300 cases versus its own baseline |
| L40S | 1.702 s to 0.254 s; 6.69x ratio; no changed JSON values on 300 cases versus its own baseline |
| MI300X compiled versus its own baseline | 4.58x observed latency ratio, but 13 changed cases; rejected |
| Optimized MI300X versus original RTX 4090 | 15 changed cases, including 5 previously correct answers regressing; rejected |
| Optimized migrations from original RTX 4090 | Zero accepted under the strict 300-case policy |

A5000/L40S accepted **same-GPU optimizations** are not accepted migrations from the 4090. Even a changed answer that improves ground-truth accuracy fails the selected exact-value-preservation policy.

Timing conditions: Qwen2.5-0.5B + v3b adapter, batch one, greedy decoding; 13 fixed timing sentences, five rounds per engine. The 300 cases are the expanded correctness check. Resident latency excludes provisioning, model loading, compilation and network. One host per model is not a universal GPU ranking, throughput study, or production SLO.

The campaign's saved cumulative cost is an estimated $7.001 under a $15 campaign cap, based on duration/rates/storage allowances. It is not a billed total or proof that the product accepts an arbitrary $20 user budget. Saved termination receipts describe those campaign resources at that time; current account resources were not inspected.

## What has gone beyond this MVP

The 11-chip campaign, general Python adaptation/planning/aggregation, per-job supervisors, iOS worker, signed desktop updates, multiple packaging/networking paths and repair tooling exceed the minimum GPU-demo proof. They are useful work. Freeze expansion rather than deleting them.

Do not spend the remaining hours adding QLoRA, payments, DiLoCo, Blender, additional chips, six decorative skill files, or another redesign. A verifier is often better expressed as deterministic checks than as another LLM agent.

## Recommended use of the remaining time

1. **First 30 minutes:** pin the frontend/backend commits, choose an explicitly experimental v3b comparison or another fully matched artifact, preserve local changes, resolve ports/origins and select one presenter flow. Freeze scope. Preserve the serving serialization/prefix-reset fixes from the former main where applicable.
2. **Next 60–90 minutes:** fix verification status/copy and connect the strict per-case policy to the demonstrated decision. Bind evidence to checkpoint/suite/configuration. Ensure rejected results never change the selected baseline.
3. **Next 60 minutes:** connect one working GPU Lab UI to the actual backend. Use existing frozen results as clearly labeled recorded hardware evidence. Exercise one live interaction on the pinned model if already available and authorized.
4. **Next 45–60 minutes:** rehearse successful comparison, rejected comparison, backend disconnect and failed candidate startup. Verify the baseline remains available. Do not launch a paid campaign as part of this audit.
5. **Final hour:** rewrite slides and product copy, record a fallback demo, export the exact evidence artifacts, and stop feature work.

If serving safety cannot be completed, use an advisory demonstration: **compare → verify → reject/recommend → keep current configuration**. Do not build an untested rollout mechanism merely to retain one sentence in the pitch.

## Paste-ready revised overview

**Relay — AI compute optimization with verified decisions**

Relay helps developers evaluate training and inference configurations using measured performance and explicit quality checks. Our prototype combines GPU training and inference experiments with a fleet platform for connecting and monitoring worker machines.

For the hackathon, we focus on one Qwen LoRA workflow and a fixed sentence-to-JSON evaluation task. We compare the fine-tuned baseline with candidate execution configurations, record latency and correctness, and reject candidates that change the agreed outputs. Verification is tied to the tested checkpoint, configuration and evaluation suite; passing a finite suite is not a guarantee for every future input.

Our saved hardware runs include NVIDIA and AMD GPUs. They demonstrate both accepted same-GPU optimizations and rejected chip-migration candidates. GPU owners can already connect controlled workers and set local availability and resource controls. Paid rentals, provider earnings, marketplace matching, automatic approval-gated traffic changes and DiLoCo are future work.

**Demo line:** “Relay compared real GPU configurations, found faster candidates, and rejected the ones that changed answers. The result is a measured recommendation backed by the original outputs and evaluation record.”

If the serving gate is implemented and rehearsed, add: “Rejected candidates cannot replace the active baseline.” Add a deployment or rollback claim only after exercising that actual flow.

## Specific edits to the original overview

- Replace “Relay is a GPU marketplace” with a present-tense optimization description and an explicit marketplace roadmap.
- Replace the immediate “fine-tune under $20, deploy” example with “Compare these configurations and recommend the fastest one that passes our evaluation.” Retain the $20 goal as future product vision.
- Replace “quality unchanged” with “the same JSON values on our frozen evaluation set,” or publish the actual permitted tolerance. Never interchange these policies.
- Replace “six independent agents, each with SKILL.md” with the actual implemented components. Show the six roles as planned architecture if useful.
- Replace LoRA/QLoRA with LoRA. Separate measured training from optimization code that has not passed a successful run.
- Replace “Deploy or rollback” in the current execution diagram with “Reject or recommend; explicit serving step.” Do not force every workload through training, inference and migration in sequence.
- Move pricing, earnings, privacy/geography matching, Blender and DiLoCo out of the MVP demo checklist.
- Retain owner availability/resource controls, with their current limitations and the fixes above.
- Identify numbers as recorded or live and label checkpoint/suite/baseline. Use 2–3 chips in the presentation; keep the full matrix as supporting evidence.
- Standardize the visible Relay/Dispatch/GPUShare naming without expanding the redesign.
- Update main's README/status and `docs/task-allocation.md`/`docs/architecture.md`: some claims about discarded machine metadata and absent planning/aggregation are now stale.

DiLoCo is correctly treated as selective, but replace “best strategy” with “candidate strategy.” The original research targets language-model training across poorly connected islands of devices; it is not automatic justification for arbitrary cross-provider training or an implementation of chip migration. [Primary paper](https://arxiv.org/abs/2311.08105).

## Validation performed in this audit

- Main backend: 207 tests discovered; **167 passed, 40 skipped**, no failures. Skips include opt-in database/device integration and platform-dependent cases.
- Main frontend: **55 tests passed** across 11 files. Production build passed; a bundle-size warning remains.
- GPU evidence verifier (`scripts/verify_benchmark_handoff.py` in GPUShare): **148 frozen evidence files and both saved adapter versions verified**, 11 measured GPU models, A5000/L40S accepted optimizations and zero accepted strict 4090 migrations. Base-model files were not verified and GPU jobs were not rerun.
- Focused GPUShare suite: **171 passed, one failed**. Failure is a dashboard text expectation against the locally edited interface, not a fresh GPU execution failure; the checkout is not fully green.
- Desktop policy-test attempt stopped before assertions because of local dependency permissions/runtime setup. The policy findings above are static review findings.
- No live GPU campaign, payment, production rollout, fresh remote-device trial or browser end-to-end demonstration was performed. These checks establish code/artifact consistency, not a completed deployed product.

The initial audit phase added this report only and preserved existing GPUShare edits. Its test counts and findings above precede the follow-up implementation; the current demo guide records the integration changes and later validation. The frontend build regenerated ignored build output.
