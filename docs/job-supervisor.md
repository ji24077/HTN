# Job supervisor design

Implemented for the existing task executors. Every newly submitted job gets a
durable supervisor record. On the public/combined backend, supervisors run when
`OPENAI_API_KEY` and `OPENAI_MODEL` are configured and `SUPERVISOR_ENABLED` is
`true` (the default). Existing jobs are not backfilled. The existing
[Fleet assistant](fleet-assistant.md) remains a separate capability.

The supervisor is scoped to a job; the external HTTP API still uses the project's
existing fleet-admin authorization. This does not add tenant isolation to the
rest of the platform.

## Try an observed failure

In the dashboard choose **Connection test (Python worker)**, enable **Intentional
failure (supervisor test)**, and submit to a compatible Python worker. Open task
details to see the supervisor's findings, decisions, actions, and final review.
Each dashboard submission now has a distinct job ID. The failure test includes
an instruction to let the first attempt run before investigating it.

For an isolated test with the configured live model:

```sh
uv run --env-file .env --project backend --python 3.12 --extra demo python examples/supervisor_failure.py
```

This starts temporary PostgreSQL, an authenticated loopback HTTP/WebSocket server,
and a real Python worker. It never connects to the database/fleet configured in
`.env`. Sentry reporting is disabled for this isolated test. It saves findings,
actions, and model replies in `.local/supervisor-live-result.json`, then shuts down
its local server, worker, and database. Only model requests leave the machine.

The intended result is one failed execution followed by an evidence-based
decision. A deterministic failure need not trigger a mutation: the agent can
leave it failed and explain why retrying would be pointless. The supervisor may
not rewrite the payload just to make the test pass.

## Ownership and lifecycle

One supervisor owns one user-submitted job from submission through its terminal
state, with a final run to record the outcome. Its scope includes all child
tasks, machines reserved for the job, and related Sentry alerts. Authority over
a machine is limited to this job's reservation and execution environment.

The server enforces this boundary on every read and action. The supervisor
cannot access other jobs or control unrelated workloads on a reserved machine.
The server continues to own heartbeats, leases, routine retries, and state
transitions.

## Wakeups

The supervisor wakes on important milestones, events, and timeouts:

- Milestones: submission, validation completion, machine reservation, execution
  start, results ready, and job termination.
- Events: reserved machine loss, repeated task failures, relevant Sentry alerts,
  changed user instructions, and cancellation requests.
- Timeouts: prolonged validation or reservation, lack of execution progress,
  and scheduled follow-ups becoming due.

Committed task lifecycle events and reserved-machine health events feed a durable
per-job inbox. PostgreSQL notifications prompt scans; a five-second fallback scan
recovers missed notifications. Each run coalesces up to 100 inbox events.
Periodic checks occur every 120 seconds; saved follow-ups can bring that forward
to a minimum of 30 seconds. Completed runs have a five-second cooldown, and
failed model runs back off for 60 seconds without advancing their event cursor.

The currently available milestone events are submission, task assignment/start,
task completion/failure/cancellation, reservations, and job termination. Upload
validation, rendering assembly, and spending projections need their future
workload integrations.

## Authority

Within the user's limits, the supervisor may reserve and release machines,
retry or cancel tasks, pause execution, and cancel the whole job. Cancellations
record a reason and release affected reservations. Changing the requested
output or increasing user limits requires user approval.

Implemented tools cannot change payloads, task timeouts, or attempt limits.
`pause_job` prevents new assignments; already-running tasks finish normally.
Cancellation invalidates leases so workers stop through the existing protocol.
Retry requires a failed task, its current generation, and remaining attempts.

Explicit worker reservations require compatible, live, idle machines, last five
minutes, and are capped at four per job. Supervisor runs renew unexpired
reservations for live workers. Other jobs cannot claim those machines while the
reservation is valid. Release removes reserved future capacity; an in-flight
task still owns its lease until completion or cancellation. Pausing, cancelling,
or ending a job releases its reservations; expiry generates a wakeup.

Per-job currency budgets and job-wide deadlines are not available in the current
submission model. The enforced limits are the submitted task attempt counts and
execution timeouts, worker compatibility/capacity, and supervisor run limits.

## Context on each wakeup

- Original submission and latest user instructions.
- Desired output, deadline, spending limit, and permitted adjustments.
- Fresh job and task progress, plus reserved machines and their health.
- Triggering events and relevant Sentry alerts.
- Durable findings, previous actions, open questions, and pending follow-ups.

Detailed logs and artifacts are fetched as needed within the job's scope.

## Durable supervisor memory

Persist the following per job across wakeups and process restarts:

| Record | Required content |
| --- | --- |
| Findings | Observations and hypotheses, distinguished explicitly, with supporting log or alert references |
| Actions | What was requested, why, the affected tasks or reservations, and the known outcome |
| Open questions | Unresolved issues and what evidence is still needed |
| Follow-ups | What to check next, the expected outcome, and when or on which event to check |
| Event cursor | Last processed event, identifying where new event processing should resume |

Each wakeup combines this memory with a fresh authoritative job snapshot.
Saved findings do not replace current job state. Unknown or pending action
outcomes remain explicit; recording an action does not imply it succeeded.

`supervised_jobs` stores the current memory and cursor. `supervisor_events`,
`supervisor_runs`, and `supervisor_actions` retain inbox events, checkpointed
runs, and actions with outcomes. Each wake gets fresh context rather than an
ever-growing conversation. Memory is bounded to 30 findings, 20 questions, and
20 follow-ups. Full audit records currently have no retention sweep.

Each run is limited to eight model requests/eight tool calls and 90 seconds.
There are two concurrent runs per backend process. A PostgreSQL session advisory
lock permits only one active run per job across processes. An active-run token
fences stale mutations after another run takes ownership. No scheduler
transaction stays open across a model request.
User controls and instruction changes also invalidate decisions based on an
older snapshot.

The loop checkpoints before and after tools. Recovery closes interrupted runs
without replaying their tools; the next run inspects current state and the durable
action ledger. Actions use UUID idempotency keys and commit their state change
and outcome together. Cursors advance only after a completed run, up to the last
event supplied to that run. Newly arriving events remain pending. A final run
observes the terminal job state; then supervision closes and pending follow-ups
are cleared.

## Logs and Sentry

Use the Sentry API through our backend for supervisor retrieval. MCP is not
part of this supervisor integration. Sentry credentials stay in the backend.

Provide job-scoped tools for log search, surrounding log context, and alert
details. Searches support combined filters for task, attempt, worker,
reservation, time range, severity, message text, and trace ID. The backend
enforces the job boundary regardless of agent-supplied filters, including
lookups by log or alert ID. Results are bounded and paginated.

Execution logs are always available from the task store, independently of Sentry.
They support task, attempt, worker, reservation, time, severity, and text filters
with cursor pagination. Severity maps failed/error/stderr events to error and
other execution events to info. `get_log_context` returns bounded neighboring
events from the same task attempt. Trace filtering is available through Sentry.

For optional Sentry retrieval configure `SENTRY_API_TOKEN`, `SENTRY_ORG`, and
`SENTRY_PROJECT`; `SENTRY_API_URL` defaults to `https://sentry.io` and can select
a regional HTTPS origin. This is a read-only API token, separate from the ingest
DSN. Queries use Sentry's organization events endpoint with the logs or errors
dataset. Free text is quoted, the backend adds the job filter, and returned rows
are checked for the same job ID. Responses are bounded; redirects are rejected.

On periodic checks the supervisor polls Sentry error events, paginates them,
and deduplicates each into a `sentry_alert` inbox event. This is API polling,
not an installed Sentry alert-rule webhook. API failures leave execution-log
investigation available. Finalized jobs stop polling.

Python and desktop error reporting carry job/task/worker identity, `attempt`,
and `reservation_id` (the task execution ID). Structured Sentry logs must include
these attributes for the same filters to work. No vector database is involved.

See [current Sentry integration](sentry.md) for what is implemented today.

## HTTP interface

- `POST /v1/tasks`: existing submission with optional `instructions` for one job.
- `GET /v1/jobs/{job_id}/supervisor`: state, memory, recent events, actions, and ten recent runs; model continuation history stays private.
- `PUT /v1/jobs/{job_id}/instructions`: update instructions and wake the supervisor.
- `POST /v1/jobs/{job_id}/actions`: validated job/task/worker action with a UUID `action_id` and reason.

The backend applies the additive schema migration on restart. Set
`SUPERVISOR_ENABLED=false` to stop autonomous model runs while retaining records
and the existing scheduler.

## Monitoring a job in the dashboard

Jobs is the default view. Rows group tasks by job ID, show aggregate progress,
and support status and text filters. Open a job to select a task and inspect its
attempts, execution timeline, searchable logs, result, and supervisor review.
Attempt cards show which worker handled each attempt and why it ended.

Logs can be filtered by attempt, worker, severity, and text. Follow latest tracks
new output; Export downloads the currently filtered events as NDJSON. The view
keeps the latest 2,000 execution events; earlier history remains in the API.
Polling resumes when a failed task is retried, retaining earlier attempts.
Supervisor findings, evidence, and past reviews expand separately from the latest
explanation. Workers, fleet activity, and the dispatch assistant have separate
navigation entries. New job opens a form that preserves its draft when closed.

## Verification

```sh
RUN_SUPERVISOR_TESTS=1 RUN_DATABASE_TESTS=1 RUN_CHAT_E2E=1 uv run --project backend --python 3.12 --extra demo python -m unittest discover -s backend/tests -v
npm --prefix frontend test
npm --prefix frontend run build
```

Integration tests use temporary PostgreSQL and controlled model responses to
verify failure investigation, retry/cancel decisions, memory recovery, worker
reservations, idempotency, attempt limits, job scoping, periodic wakeups, HTTP
authentication, and concurrent-run fencing. Sentry transport tests use mocked
API responses; the live failure test uses the configured model and local
execution history.
