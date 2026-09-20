# Frontend implementation plan

Status: frontend implementation delivered locally; see [ChatGPU UI implementation](chatgpu-ui.md) for completed work, validation, and remaining backend scope. The unchecked roadmap below retains the full original acceptance criteria. Design: [Frontend redesign](frontend-redesign.md). Baseline: September 20, 2026 audit of the current local application and source. This plan changes no application behavior by itself.

## Delivery strategy

First fix controls and states that mislead people. Then move existing outcomes into a clearer layout. Add report generation as a real capability, followed by the remaining screen and visual-system improvements. Avoid a single large rewrite: each slice should be reviewable, preserve existing jobs, and have explicit acceptance evidence.

Recommended order: **1 → 2 → 3 → 4 → 5 → 6 → 7**. Slices 5 and 6 do not depend on report generation and may be scheduled before slice 4 if needed. Do not hold the immediate accuracy fixes for the redesign. Tests should verify user-visible state transitions and data contracts, not duplicate markup.

## 1. Correct misleading controls and stale results

Scope: `WorkerGrid`, `TaskDetails`, `JobSupervisor`, `JobOutputs`, `SimulationDetails`, `App`, and related API/state helpers.

- [ ] Replace the GPU-looking preference toggle with explicit policy wording. Render detected hardware separately; missing capability never implies GPU availability.
- [ ] Fetch current full results when an open job changes attempt, result revision, or terminal state. Preserve the previous useful view during refresh; show fetch failure and Retry.
- [ ] Derive active input from authoritative job phase/current question, not all supervisor memory questions. Keep previous questions in history; distinguish final-review pending.
- [ ] Give output fetching loading/error/empty-active/empty-terminal states. Rename the generic export “Download raw result JSON.” Make artifact downloads explicit.
- [ ] Correct failed phase styling and remove adaptation counts presented as pipeline progress.
- [ ] Make New job consistently open project submission. Rename bulk test actions to describe the test and eligible worker count; fix nested-control keyboard propagation.

Backend work: confirm whether current question identity and result version are available. Where missing, add the smallest compatible contract; do not use frontend timestamp guesses as long-term reconciliation. Until richer history exists, suppress terminal-state input prompts based on authoritative state.

Acceptance:

- CPU-only, MPS/CUDA, unknown, and offline workers have distinguishable honest states.
- A job opened while running updates to a completed result without closing/reopening it, including after one failed result fetch.
- Completed and failed jobs do not ask historical questions. A currently blocked job has an answer control.
- Empty failed/completed jobs do not imply files are still being generated.
- Pressing Enter/Space on a worker child control invokes only that control.

Verification: focused existing frontend tests in `WorkerGrid`, `TaskDetails`, `JobSupervisor`, `JobOutputs`, `SimulationDetails`, and `App`; add lifecycle cases that expose the defects. Capture the same output and worker screenshots for comparison.

## 2. Define the shared job outcome contract

Scope: frontend `api/types.ts`, `api/client.ts`, `lib/jobs.ts`, `hooks/useFleet.ts`; backend preprocessing models/routes/store and supervisor models/service/store. Read existing schemas before choosing migration names or new endpoints.

- [ ] Inventory existing fields and define one normalized job view: title/workload, lifecycle/current phase, measured progress, actual executions and runtimes, validation, metrics, artifact availability, current question, report state, result revision.
- [ ] Keep execution outcome, report readiness, and supervisor final-review state independent.
- [ ] Track questions by identity and resolution, including answers/supersession/terminal archival.
- [ ] Expose placement from execution records rather than root-task assignment. Define multi-worker, missing, and legacy states.
- [ ] Define artifact purpose, media type, size, validation/provenance, and preview capabilities using current output storage where possible.
- [ ] Define job-level event aggregation and cursor ordering across root/child tasks; preserve source, phase, attempt, and worker. Provide explicit scoped access until aggregation is ready.
- [ ] Add backward-compatible handling for existing jobs and services; no automatic legacy report backfill.

Acceptance: fixture contracts cover queued, active-input, GPU training, multi-worker simulation, failed/cancelled execution, JSON-only legacy output, report-generation failure, and a running service. List and details agree on lifecycle and placement. Raw worker output survives formatting unchanged.

Verification: API serialization/compatibility tests, supervisor question lifecycle tests, and frontend contract fixtures. Do not change execution success criteria to fit the presentation layer.

## 3. Rebuild job details around the result

Scope: `App`, `TaskList`, `TaskDetails`, `SimulationDetails`, `JobOutputs`, `JobSupervisor`, `ServicePanel`, shared styles; split components by responsibility as needed.

- [ ] Add addressable job navigation with Overview, Files, Activity, Details; preserve list filters/position on Back and reset unrelated page scroll.
- [ ] Make Overview the default for every lifecycle, with state-specific content from the design document.
- [ ] Move long plan/history/raw JSON out of the initial viewport.
- [ ] Show actual worker/runtime, accepted metrics and bounds, meaningful progress, validation, and available deliverables near the top.
- [ ] Introduce artifact rows and safe structured/text/image previews. Preserve streaming downloads and accurate browser hand-off feedback.
- [ ] Provide one discoverable job activity entry point, including worker stdout; retain exact raw text.
- [ ] Add useful mobile row metadata and a Cancelled filter. Keep destructive operations distinct from navigation.
- [ ] Integrate service readiness and authenticated endpoint guidance into the same shell without imposing finite-job progress.

Acceptance: the audited GPU job opens with training success, MSE about 0.000256 against ≤ 0.01, worker-2/MPS, checkpoint and metrics. Its legacy report absence is clear. The Monte Carlo job shows available aggregate results without claiming a report exists. The failed job leads with cause and failed phase. None require reading a plan to find outputs.

Verification: navigation/back/refresh checks; relevant frontend integration tests; screenshots at 390×844, 881×761, and 1440×900 with long titles and real plan content. Check keyboard tab order and focus after navigation/dialog closure.

## 4. Generate readable reports and evidence-based visuals

Scope: backend preprocessing planning/models/service/artifacts/outputs/store; supervisor integration where appropriate; frontend outcome/report rendering. Existing entry points include `backend/src/orchestrator/preprocessing/` and `backend/src/orchestrator/supervisor/`; reuse the durable artifact transport rather than embedding a report in task-result JSON.

- [ ] Extend planning to capture report expectations and reserve preparation resources inside the user's limits. Preserve frozen workload/validation requirements.
- [ ] Add a durable report stage keyed to job/result revision, using validated artifacts and execution evidence.
- [ ] Start with deterministic structured summaries and useful Markdown reports; add agent narrative only with evidence references and bounded cost.
- [ ] Include measured results, acceptance bounds, validation, environment, artifact links, and limitations. Generate charts only from available or explicitly derived data.
- [ ] Validate report references, finite metrics, and output existence before publishing ready state.
- [ ] Handle report-only retries without rerunning the workload. Distinguish default report failure from failure to deliver an explicitly requested report.
- [ ] Show execution results while the report is pending; make report failure and Retry visible.
- [ ] Add optional explicit report generation for legacy jobs only after budget/deadline behavior is defined. Keep original execution artifacts unchanged.

Acceptance:

- A new successful GPU training request returns a readable report as well as checkpoint/metrics; its known metric matches validated data.
- No loss curve is fabricated when step history was not recorded. A scalar summary remains a valid readable result.
- A simulation report explains its aggregate and any supported visualization in plain language.
- A report-generation failure retains downloadable execution outputs and has a report-only recovery path.
- Explicitly requested deliverables are not marked ready while the report is missing.
- Duplicate report retries publish one authoritative result for a given revision; a restarted coordinator resumes durable state.

Verification: report schema/evidence tests, failure/retry/restart integration tests, legacy compatibility, and budget enforcement. Use an isolated bounded fixture for end-to-end validation, not a new full training run just to test rendering. Live GPU tests, if needed, must retain the existing workload and respect its budget.

## 5. Simplify submission and worker management

Scope: `SimulationComposer`, `MaxSpendField`, `WorkerGrid`, `DeviceInvite`, `TaskComposer`, `App`.

- [ ] Compact submission to one explanation, file area, request, optional budget, inline errors, and a visible footer action.
- [ ] Keep hardware and execution/service settings agent-managed; show inferred choices after planning.
- [ ] Make worker cards open machine details; show available machines first and collapse offline history.
- [ ] Put Add worker beside the heading; include invite expiry/pairing states.
- [ ] Move connection tests and other built-ins into Diagnostics. Exclude ineligible/offline targets and label bulk actions precisely.
- [ ] Document which runtime policies can be changed remotely and which require worker startup configuration.

Acceptance: opening New job after using Diagnostics still opens the standard upload flow. At 390×844, submit is reachable without scrolling through explanatory prose and remains usable with the keyboard open. Inspecting a machine never submits a test. Policy wording never promises installed GPU dependencies or an active GPU runtime without reported evidence.

Verification: existing composer/budget/device tests, diagnostics selection tests, manual mobile/keyboard inspection, desktop/mobile screenshots. Retain idempotent submissions, input preservation after errors, and budget validation.

## 6. Clean up assistant, activity, and experiments

Scope: `ActivityFeed`, `ChatPanel`, `GpuLab`, `GpuLabExperiment`, `GpuLabCharts`, `api/gpulab.ts`, relevant hooks and formatters.

- [ ] Link meaningful fleet events to jobs and preserve raw log strings.
- [ ] Share readable message rendering; preserve reader scroll position. Scope job-context chat through the backend before exposing the action.
- [ ] Rename GPU Lab navigation to Experiments and show its separate environment/status explicitly.
- [ ] Separate setup/operations from results/evidence; provide an actionable unavailable state.
- [ ] Bind comparisons to matching inputs, model/checkpoint, baseline, and quality criteria. Show slowdown/failure instead of unconditional “faster.”
- [ ] Label inferred timing estimates and expose accessible chart data and keyboard interactions.

Acceptance: a healthy local GPU worker and unavailable experiment service are understandable together. A comparison across different prompts/models cannot display a valid speedup claim. Operations are recognizable before invocation. Reading previous assistant messages does not snap back to the newest reply.

Verification: experiment fixtures for comparable/incomparable runs, slowdown, failed quality gate, offline/stale data, and missing curves; targeted message/log tests. Because populated experiments were code-only in the audit, obtain a representative fixture screen before declaring visual completion.

## 7. Consolidate styling and validate the whole experience

Scope: shared CSS, `Icon`, `Login`, all changed screen components. Introduce small shared primitives while implementing earlier slices; this slice removes remaining duplication and checks consistency.

- [ ] Unify primary/secondary/destructive buttons, statuses, forms, tabs, disclosures, artifacts, and empty/error states.
- [ ] Replace ad hoc Unicode controls where a shared icon is appropriate; label icon-only actions.
- [ ] Simplify conflicting breakpoints and stack job metadata before it squeezes readable content.
- [ ] Associate field errors/help with controls; check keyboard access, focus, touch targets, contrast, and reduced motion.
- [ ] Verify small-screen navigation exposes every destination and fixed footers do not obscure content or focused inputs.
- [ ] Repeat the original screenshot walkthrough and record differences beside acceptance results.

Acceptance: all 18 existing components have a documented disposition from the design matrix. No screen reports success, hardware availability, pending input, or download completion without evidence. Main outcomes and primary actions remain discoverable at all three audit sizes.

Verification: run the repository's frontend test suite, typecheck and production build; run backend suites relevant to changed contracts/report logic. Use manual browser checks for responsive layout and chart/keyboard behavior. Application-wide checks follow the meaningful targeted tests, rather than repeatedly running every suite after documentation or copy changes.

## Release and review checklist

- [ ] Review each slice against its acceptance cases with screenshots or transition-test evidence.
- [ ] Land compatible API additions before frontend dependencies; keep legacy response/output handling until verified migrated.
- [ ] Keep report generation disabled until contract, persistence, retry, and budget tests pass; enable deliberately for new jobs.
- [ ] Preserve existing result files, original execution records, and active services throughout rollout.
- [ ] Record known limitations, especially states verified only through fixtures rather than live infrastructure.
- [ ] After rollout, inspect one completed job, one failed job, one active-input job, one legacy JSON-only job, worker capability states, and a representative service/experiment view.

The first reviewable change should be slice 1. It directly removes misleading behavior while the outcome contract and larger redesign are prepared.
