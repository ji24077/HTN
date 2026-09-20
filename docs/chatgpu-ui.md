# ChatGPU UI implementation

Implemented locally on September 20, 2026, following the [frontend design](frontend-redesign.md) and [implementation plan](frontend-implementation-plan.md).

## Delivered

- ChatGPU branding in the workspace, authentication screens, and browser title.
- Addressable jobs with Overview, Files, Activity, and Details. Each tab has a URL; refresh and browser Back restore the selected job and tab. Plans, raw JSON, and supervisor history are secondary.
- Job links show a loading dialog immediately, with persistent errors, Retry, and Back to jobs. Invalid links have an explicit recovery action, and late responses cannot reopen a job after navigation away.
- Readable GPU and simulation outcomes, recorded metrics, acceptance bounds, and evidence-based metric visuals. Completed results offer a downloadable Markdown summary without rerunning anything.
- Explicit artifact Download and Preview actions. Bounded text/JSON/image previews, original streaming downloads, retryable loading failures, and distinct active/terminal empty states. Raw result JSON is a secondary export.
- Results refresh when an open job completes or changes attempts. Historical supervisor questions are archived; current project questions retain their answer form. Final-review lag does not change execution status.
- Worker inspection without implicit task submission. Runtime policy uses CPU only / Automatic, separates device evidence, respects offline/unknown states, and preserves startup-only configuration for Python workers. Offline history is collapsed.
- Compact files/request/budget submission, visible submit footer, field-level budget guidance, and separate connection diagnostics. New job always returns to the agent-managed flow.
- Responsive navigation with mobile More, useful mobile job timestamps, cancelled-job filtering, accurate initial loading states, and containment of narrow-screen table overflow.
- Fleet activity links to known jobs; raw execution messages retain their original filenames/environment variables. Worker stdout is accessible in job Activity, with its execution scope stated explicitly.
- Improved service endpoint copying and action hierarchy, readable assistant replies with scroll preservation, and invite expiration feedback.
- Experiments explicitly uses a separate environment. Setup & results replaces the ambiguous Evidence button. Unknown cross-run training pairings no longer receive speedup verdicts; inference slowdowns and estimated cache timings are labeled honestly. Charts expose keyboard inspection/data alternatives.

The shared visual treatment is in `frontend/src/chatgpu.css`; original layout rules remain for existing secondary screens. No new frontend dependency was added.

## Result summary versus generated report

**Download summary** creates a Markdown document in the browser from the job's saved outcome, metrics, acceptance bounds, workers, and validation records. It is immediately useful for existing jobs, including the audited GPU and Monte Carlo runs. It does not become a new server-side artifact and does not perform new analysis.

Durable report generation, report-specific retries/budgets, full artifact provenance, question identity reconciliation, cross-task log aggregation, and a richer job-list placement/workload contract remain backend roadmap work. The UI uses accurate fallback labels where those fields are missing. Runtime derived from a plan is explicitly labeled planned; it is not presented as GPU-utilization telemetry.

Original job execution, validation criteria, budgets, worker configuration, and stored outputs were not changed by this UI implementation.

## Validation

- Frontend: 219 tests passing across 24 files; TypeScript and production build pass.
- Regression coverage includes completion-result refresh/retry, archived terminal questions, offline/unknown runtime policy, keyboard policy controls, standard submission after diagnostics, initial job loading, metric bounds, output-list retries, and the 2 MiB preview limit.
- Authenticated live browser checks at 1440×900, 881×761, and 390×844: GPU outcome and summary download, metrics preview, worker stdout, job refresh and Back navigation, Monte Carlo results, failed jobs, worker inspection, submission footer, and the unavailable experiment environment.
- The live summary download was checked against the recorded MSE and threshold; previewed JSON matched the saved metric. No new jobs or infrastructure actions were triggered.
- No JavaScript page errors or page/dialog horizontal overflow in the captured walkthrough.
- Hosted-service, populated experiment, and authentication edge cases retain automated/source validation; no new live service or experiment was provisioned for this review. The build still reports a large-bundle advisory.

Local review artifacts (not versioned): [screenshots](../.local/chatgpu-review/index.html), [browser verification](../.local/chatgpu-review/verification.json), and [example downloaded summary](../.local/chatgpu-review/gpu-summary.md).

Navigation follow-up: [browser verification](../.local/chatgpu-navigation/verification.json) covers tab refresh/history, failed-link retry, cancellation during loading, invalid links, and desktop/mobile layouts. Request failures and delays were injected in the review browser; no stored jobs were modified.

## Checking locally

```sh
npm --prefix frontend test
npm --prefix frontend run build
```

The local development application is available at `https://localhost:5174`. The PR integrates with the latest main branch, including named workers and the experiment workflow/evidence controls. Unrelated backend and agent work remains outside this UI change.
