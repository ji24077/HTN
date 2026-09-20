# Frontend redesign: outcomes, files, and clear actions

Status: design baseline; the UI has since been implemented as ChatGPU. See [implementation notes](chatgpu-ui.md) for delivered behavior and remaining backend work. Based on the September 20, 2026 live-site audit and source review. Companion: [implementation plan](frontend-implementation-plan.md).

## Purpose

A person should submit files, describe the result they want, set an optional budget, and let the agent choose execution mode, hardware, and settings. When they return, the interface should answer: **What happened? What did I get? What needs my attention?**

The current interface makes those answers difficult to find. A completed GPU training job leads with a long execution plan; its Result tab contains raw JSON; downloadable artifacts look less prominent than the JSON export. Historical agent questions can still look actionable after completion. On Workers, an automatic runtime preference can look like confirmation that GPU execution is enabled.

This proposal preserves agent-managed submission while making outcomes and actual capabilities explicit. It also adds a real report contract. Renaming a JSON download cannot meet the expectation of a readable report.

## Evidence and scope

The audit inspected 18 live screenshots across 1440×900, 881×761, and 390×844 viewports, all 18 React components, the app shell, shared CSS, and relevant API/hooks/formatting code. Existing signed-in jobs and controls were inspected without submitting jobs or changing worker settings.

Local supporting evidence, available in the audited workspace:

- [Full audit and source references](../.local/frontend-audit/AUDIT.md)
- [Screenshot gallery](../.local/frontend-audit/index.html)
- [Completed training job](../.local/frontend-audit/18-gpu-job-desktop.png)
- [Output controls and historical question](../.local/frontend-audit/03-gpu-outputs.png)
- [Worker runtime controls](../.local/frontend-audit/08-workers.png)
- [Mobile job details](../.local/frontend-audit/15-gpu-mobile.png)

These local artifacts are supporting material, not versioned dependencies. This document and the implementation plan stand alone. The baseline includes uncommitted work; source references must be checked against the checkout when implementation starts.

Login/recovery, hosted-service states, populated experiment charts, and the running-to-completed result-refresh defect were reviewed in code rather than exercised live. The result-refresh defect is a static finding: details fetch on job ID change, so an already-open running job can retain an old result after its list entry completes.

## What to retain

- The restrained green palette, simple navigation, existing focus styling, and reduced-motion support.
- Files + request + optional budget as the normal submission flow.
- Search, filtering, pagination, and access to execution evidence.
- Real worker capability reporting and distinction between CPU and Metal/CUDA runtimes.
- Validation against original acceptance criteria, durable output downloads, and explicit provenance.
- The supervisor's readable summaries, while giving lifecycle authority to server state.

## Product decisions

1. **Outcomes lead.** Completed jobs open on an overview with a readable result, useful metrics, validation, and deliverables.
2. **State has one authority.** Model prose is commentary. Server lifecycle and current question records determine badges, actions, and required input.
3. **Controls describe their effect.** Machine inspection, execution-policy changes, connection tests, and file downloads have distinct labels.
4. **Hardware is evidence.** Detected capability, allowed scheduling policy, and the runtime actually used are separate fields.
5. **Details remain accessible.** Plans, attempts, hashes, and raw payloads belong in secondary views.
6. **Reports are deliverables.** A report is generated, stored, and validated as an artifact, with its own lifecycle and evidence references.
7. **No extra configuration burden.** Do not add mandatory CPU/GPU, job/service, or service-settings forms. Ask only for intent or acceptance criteria that cannot reasonably be inferred.

## Navigation and screen structure

Primary destinations: **Jobs**, **Workers**, **Assistant**. Put **Experiments** and fleet-wide **Activity** in secondary navigation, visible through a labeled More menu on small screens. Experiments remains available but explains that it uses a separate environment from the worker fleet.

Use addressable job pages with Overview, Files, Activity, and Details tabs. Back navigation should restore the jobs list's filters and position. Switching top-level destinations starts at the destination heading. Deep links, refresh, and browser Back must work.

### Jobs

Each row shows the user's title or uploaded project name, workload, status/current phase, relevant progress, actual execution runtime/worker when known, and submission/completion time. Keep enough metadata on mobile to distinguish two jobs with the same filename.

Do not label training as “simulation job.” Before assignment use “Awaiting assignment”; after execution show the recorded worker/runtime, including multiple workers when applicable. Do not infer placement from an unassigned root task. Percentages require a real work counter; otherwise show a named phase. Adaptation limits are diagnostic metadata, not progress denominators.

### Job overview

Desktop outline:

```text
← Jobs          gpu_upload.py                 Completed
Overview   Files   Activity   Details

Training completed and validation passed.
MSE 0.000256 / required ≤ 0.01     worker-2 · Apple Metal (MPS)

[Readable report preview, when available]      [Download report]

Files
checkpoint.pt    Model checkpoint · 14.4 KiB  [Download]
metrics.json     Evaluation metrics · 30 B    [Preview] [Download]

Validation passed   Duration …   Estimated cost …
Agent summary / latest relevant update
```

The example metric and artifacts come from the audited GPU job. That job has no report artifact today: its truthful legacy view must say “No report was generated for this job” and show available results. Do not render the proposed report control until an artifact exists.

Use one main content column for the result. A desktop side rail may hold compact metadata, but it must not squeeze the result at intermediate widths. On mobile, put status, summary, essential metrics, and first deliverable ahead of plans and diagnostics. Long content must wrap without horizontal page scrolling.

### State-specific behavior

| Authoritative state | Main content and available action |
| --- | --- |
| Queued/planning | Current phase and what the agent is determining; Cancel when supported. |
| Waiting for input | One current question with its answer control; explain what is blocked. |
| Probing/running/validating | Current phase, measured progress if available, latest meaningful update, View activity. |
| Execution complete; report preparing | Execution outcome and existing files immediately available; explicit report preparation state. |
| Completed | Outcome, accepted metrics, validation, available report/files. Historical questions are archived. |
| Failed | Concise cause and failed phase; available partial files labeled; Retry only when supported for this failure. |
| Cancelled | Cancellation outcome, completed work, and any retained outputs. |
| Hosted service | Starting/Ready/Degraded/Stopped lifecycle, endpoint availability, actual runtime, and service actions. No finite-job 100% progress bar. |

Supervisor final review may lag execution. Show “Final review pending” if applicable; do not allow stale supervisor prose to overwrite the completed execution state. A terminal job never presents an old question as required input. A new post-completion request belongs to a separately identified follow-up action.

### Files

Use consistent artifact rows: readable name, purpose, type, size, preview availability, and explicit Download. Put reports and user-requested deliverables first. Keep “Download raw result JSON” in a secondary export area with a short explanation.

Support separate loading, fetch-error/Retry, empty-active, and empty-terminal states. “No files were produced” is different from “Files will appear after execution.” Jobs with only structured results should show those results and explain that no file artifacts exist.

Start previews with structured metrics, text/Markdown, and images. Preserve the existing streaming download path for large files. Browser hand-off is not proof that a download completed; use accurate busy/error feedback.

### Activity and Details

Activity provides one job-level view of milestones and worker output, with phase, worker, attempt, and source filters. Expand verbose planner messages. Preserve raw log text exactly, including underscores in filenames and environment variables. Until cross-task aggregation exists, label each log scope and expose child executions directly.

Details contains the execution plan, planning history, adaptations, validation evidence, execution environment, IDs, hashes, and raw result payload. Keep service diagnostics here too. These views remain useful without dominating the first screen.

## A real report and visualization contract

Every newly submitted finite compute job should receive a concise readable outcome summary. Successful jobs should also receive a stored report by default, unless the user explicitly opts out. The agent chooses useful visualizations when the available evidence supports them. Hosted services receive a deployment summary rather than a fictitious final computational result.

Proposed sequence:

1. During planning, record requested outputs, acceptance criteria, and report requirements within the existing budget/deadline.
2. Execute and validate the original workload, preserving its frozen contract.
3. Build a structured outcome from authoritative execution records and validated artifacts.
4. Produce a readable report with summary, measured results, validation, execution environment, useful visuals, limitations, and links to artifacts.
5. Validate referenced artifacts/metrics, persist the report, and publish its readiness independently from execution state.

The structured outcome should support typed metrics with units and acceptance bounds; evidence/artifact references; actual workers/runtimes; timestamps; execution and validation state; report state; and a result revision. Prefer extending existing types and routes over introducing a second conflicting source of truth. Exact schema belongs in the first implementation slice.

Report state needs at least not-requested/legacy, pending, generating, ready, and failed. If the user explicitly requested a report, missing report generation means the complete request is not fulfilled: show “Execution passed; report failed” with a report-only retry. If the default optional report fails, retain execution success and show the presentation failure separately. Never report all requested deliverables ready before they exist.

For the GPU example, MSE and threshold are known, but no training-loss curve was recorded. Show the metric and validation evidence; do not invent a curve. A prediction plot requires a reproducible evaluation that yields the plotted values and fits the budget. For Monte Carlo, a known aggregate supports a count/proportion visualization; it does not support an invented simulation trajectory.

Report preparation must not silently retrain the model, rewrite source, loosen validation, or rerun the original workload. Derived analysis must declare its evidence and cost. Start with safe structured rendering/Markdown rather than arbitrary executable HTML. Report retries should reuse immutable validated artifacts and be idempotent by result revision.

Legacy jobs retain their original records. Display a summary of the evidence available. An explicit “Generate report” action can be added once report-only execution and budget handling exist; do not backfill automatically or imply a legacy report already exists.

## Workers, submission, and secondary screens

**Workers:** available machines first; Add worker near the heading; offline history collapsed. Selecting a machine opens its details. Show reported hardware, scheduling policy (Automatic / CPU only), and observed execution runtime separately. Unknown capability means unknown, especially while offline. Explain whether a policy change applies immediately or requires restarting that worker. Connection tests live under Diagnostics, with labels such as “Run connection test on 2 eligible workers.” They are never an unlabeled consequence of selecting a card.

**Submission:** New job always opens the same files/request/budget flow, regardless of the last diagnostics view. Use one short explanation, a compact file area, useful request examples, inline field errors, and a visible action footer. Agent-selected service settings appear as a readable proposed/executed plan rather than mandatory form fields. Preserve deduplication, size validation, retry behavior, and submitted values after errors.

**Assistant:** retain the simple conversation UI. Render readable responses consistently with job summaries. Add “Ask about this job” only when the backend receives the actual scoped job context. Do not force-scroll a reader who is inspecting earlier messages.

**Experiments:** separate environment availability from local fleet availability. Distinguish Setup, Run experiment, and Results; “Evidence” must not unexpectedly open training controls. Comparisons identify their input, model/checkpoint, baseline, and quality gate. Display slowdowns honestly and label inferred timings as estimates. Charts need readable data alternatives and keyboard-accessible inspection.

**Service details:** explain readiness and authenticated access, offer endpoint copy and an appropriate request example, distinguish Stop from Restart, and show operation/error feedback. A copied endpoint is not necessarily a publicly accessible URL.

## Component disposition

| Existing component | Keep | Proposed change |
| --- | --- | --- |
| `TaskList` | Search, pagination, filters | Correct workload/placement, useful mobile metadata, cancelled filter. |
| `TaskDetails` | Task evidence and actions | Addressable four-tab job page; refresh by result/state revision; visible fetch errors. |
| `SimulationDetails` | Validation and execution provenance | Distribute outcome into Overview and technical history into Details; truthful phase progress. |
| `JobOutputs` | Durable downloads | Artifact rows, previews, explicit state handling, secondary raw export. |
| `JobSupervisor` | Readable summary/history | Authoritative current question only; final-review lag and archived questions. |
| `ServicePanel` | Service status/actions | Lifecycle-specific overview, access guidance, action hierarchy, errors. |
| `SimulationComposer` | Agent-managed submission | Compact layout and persistent action footer. |
| `MaxSpendField` | Optional CAD cap and help text | Connect field errors and invalid state to input. |
| `TaskComposer` | Connection-test functionality | Move to Diagnostics; eligible workers only; explicit test names. |
| `WorkerGrid` | Reported capability and health | Inspectable cards, separate policy/hardware/runtime, honest unknown/offline state. |
| `DeviceInvite` | Copy, expiry, one-use semantics | Place under Add worker; show current expiry and pairing outcome. |
| `ActivityFeed` | Search and time filters | Meaningful event summaries, job links, clearer placement and phase. |
| `ChatPanel` | Suggestions and pending-message recovery | Shared text rendering, scroll preservation, explicit job context. |
| `GpuLab` | Experiment workflow | Separate environment, setup/results navigation, valid comparison identity. |
| `GpuLabExperiment` | Experiment evidence | Distinct setup/actions/results, comparable baselines, honest speedups. |
| `GpuLabCharts` | Responsive charts and honest missing-data states | Accessible data, comparison provenance, measured vs inferred labels. |
| `Login` | Simple labeled forms | Associate help/errors with fields and use shared branding. |
| `Icon` | Current SVG primitive | Use consistently; label icon-only actions at the call site. |
| `App`, shared styles/hooks/API | Palette, data transport, useful primitives | Routing, state ownership, shared control/status/artifact layouts, simpler responsive rules. |

## Visual direction and completion bar

Use the existing palette with fewer competing badges and borders. Establish shared button, status, form, artifact, tab, disclosure, and empty-state primitives. Start with 14–16px body/control text, readable secondary text, and generous touch targets; verify with real content rather than shrinking labels to fit. Green signifies confirmed success/readiness, not unknown hardware or a disabled selected control. Keep meaning in text, not color alone.

The redesign is successful when a person can open the completed GPU job, understand its validated outcome and actual runtime, and preview/download useful results without expanding a plan or reading JSON. At all audited widths, an active question has its answer control nearby, New job has a reachable primary action, and logs and technical details remain easy to find.

See the [implementation plan](frontend-implementation-plan.md) for delivery order, source ownership, and acceptance checks. Billing, scheduler algorithms, GPU provisioning, and model-training behavior are outside this frontend redesign, except for the explicit outcome/report contracts described above.
