# Validation

After the repository layout change, 28 Python unit tests and seven frontend tests
passed. The frontend build, Python wheel/source builds, Ruff checks, CLI discovery,
and relative documentation links were verified from the repository root. Release
archives contain the compiled UI and SQL without local credentials or runtime
data. The relocated demo reopened its preserved PostgreSQL data and returned 200
for health and the dashboard; the smoke-test server was stopped afterward.
Docker execution and live public/Tailscale hosting remain unvalidated.

## Earlier prototype validation

The Python package has been installed and the dashboard has been exercised
against two real local worker processes and PostgreSQL. Both targeted example
tasks completed with progress and accepted results. The server and workers were
restarted after their terminals closed; persisted task history was recovered.
The worker selector was checked through an individual Worker A dispatch. Browser
checks confirmed worker cards, task buttons, selection, and keyboard focus remain
in place across live updates; status and progress still update.
The dashboard now uses SSE driven by committed PostgreSQL notifications. A local
UI task reached Worker A and completed with pushed progress and zero browser
snapshot requests. Unauthenticated stream requests returned 401. An isolated
PostgreSQL check verified commit-only notifications, rollback/no-op silence,
listener reconnection/resync, and subscriber cleanup.
The agent client/CLI submitted a task to Worker A and received its accepted
result over SSE. Local checks also covered idempotent resubmission, changed-spec
conflicts, wait timeout without cancellation, explicit cancellation, audit reads,
and discovery of seven tool schemas without credentials.
Eight client boundary tests pass (`uv run --project backend python -m unittest discover -s backend/tests -v`),
covering uncertain writes, redirects, invalid tool arguments, stream cleanup,
wait deadlines, tasks outside the snapshot window, and authentication failures.
After reorganizing the project, all 12 unit tests passed, including four checks
for packaged dashboard assets and their authentication/local-demo boundaries.
The wheel and source distribution built successfully. An isolated wheel install
loaded the SQL and served the HTML/CSS/JavaScript; neither archive included local
credentials or the virtual environment. The restarted server and both workers
completed targeted tasks through the client/SSE interface. Browser checks showed
both workers online, both static assets returning 200, and no snapshot polling.
The React migration passed four frontend interaction tests, TypeScript checking,
and 13 Python tests. A browser submission through Vite's development proxy reached
Worker B and completed via SSE. Both port 5173 (hot reload) and port 8787 (built
assets) rendered the React UI with two live workers and no snapshot polling.
The rebuilt wheel/source distribution include compiled assets; an isolated
wheel install served the index, hashed CSS/JavaScript, and local session endpoint.
The Dockerfile now includes a frontend build stage, but Docker execution remains
unvalidated in this environment.
The full checklist below, container builds, and remote trials remain outstanding.

## Local build and behavior

- Install the locked Python package, verify both console entry points, build the
  container image, and inspect Compose configuration. Check Python 3.12–3.14.
- Submit the example tasks, connect a worker, and observe queued → assigned →
  running → succeeded with one accepted result per task.
- Resubmit identical tasks, including different JSON property order; no new
  generations or duplicate tasks should appear. A changed spec should get 409.
- Check invalid IDs, request size limits, invalid task specs, missing tokens,
  unknown worker identities, and unsupported protocol versions.
- Submit incompatible runtime, VRAM, and kind requirements; they stay queued.
- Connect two workers and use simultaneous submissions. Neither worker should
  have more than one active task. Repeat with two server processes sharing a DB.
- Verify an idle heartbeat acknowledgment arriving after an assignment cannot
  revoke the new assignment.

## Recovery and authority

- Kill a worker during a long stub task. Its task should retry after expiry,
  with a higher generation, on an eligible worker.
- Partition a worker that continues computing. Submit its old result after
  reassignment; PostgreSQL must reject it without changing accepted output.
- Reconnect using the same worker ID. Old sessions must not renew, complete,
  or reclaim work; their disconnect handler must not mark the new session down.
- Stop acknowledging assignments; the 10-second assignment lease must expire.
- Keep heartbeats alive while exceeding the task timeout. Recovery must use the
  execution deadline, independent of connection health.
- Cancel a running task; a late completion must be rejected and the executor
  should stop on the next heartbeat response.
- Duplicate acknowledgments and successful completion messages. Accepted
  output should remain unchanged and completion events should not duplicate.
- Retry a task through its attempt limit; it must become terminal.
- Restart the server between database assignment and network delivery, and
  between accepting a result and returning its acknowledgment.
- Interrupt PostgreSQL; no ownership change or result should be accepted without
  the authoritative transaction. Reconnect and reconcile after recovery.
- Restart Redis; task ownership and recovery should still work.
- Stop the server while workers are connected; inspect asyncio task/process exit
  and durable disconnect state.

## Optional later pod trial

Only after the local checks, provision two controlled Runpod workers against a
TLS control-plane endpoint. The stub needs no GPU, so GPU speed or rendering
correctness cannot be inferred from this trial. Repeat disconnect, reconnect,
and stale-completion cases across the actual network; record recovery latency,
attempt counts, and audit events. Stop or delete trial resources afterward.
