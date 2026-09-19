# Validation

After the repository layout change, 28 Python unit tests and seven frontend tests
passed. The frontend build, Python wheel/source builds, Ruff checks, CLI discovery,
and relative documentation links were verified from the repository root. Release
archives contain the compiled UI and SQL without local credentials or runtime
data. The relocated demo reopened its preserved PostgreSQL data and returned 200
for health and the dashboard; the smoke-test server was stopped afterward.
Docker execution and public hosting remain unvalidated. Later local Tailscale
validation is recorded below.

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

## Embedded Tailscale transport

The backend and worker both include a Go/tsnet helper. Go forwarding, shutdown,
fixed-target restrictions, and private gateway route-isolation tests pass under
the race detector. Python tests verify helper lifecycle, environment credential
isolation, and gateway TLS hostname verification before worker credentials are
sent. The backend regression suite passes (35 passing tests, one opt-in database
test skipped). The local bundled backend's public and private health endpoints
return 200; the private management API returns 404.

The macOS helper and bundled backend are authorized in the development tailnet.
Two manually authorized local CPU workers connected through the private TLS
gateway and reconnected after a backend restart. Docker build/runtime and
dispatch from a separate machine remain untested on this host.

## Automatic worker enrollment

A user signed in through the setup CLI with their approved Supabase account.
The public API issued a single-use Tailscale auth key, saved only the worker
credential digest in PostgreSQL, and the worker joined without an interactive
Tailscale approval link. The CPU worker connected to the private gateway and
maintained fresh heartbeats. Its initial connection failed once and recovered
automatically within three seconds; the same session remained alive afterward.
The setup CLI now enables connection-status logging so a successful recovery is
visible after a warning.

The provider smoke test created and revoked an unused enrollment key. The
enrollment endpoint rejected unauthenticated requests with 401. Unit and isolated
PostgreSQL integration checks cover approval, private credential storage,
cross-process authorization, request replay, quotas, failure cleanup, and route
isolation. Desktop packaging and a separate-machine onboarding trial remain
outstanding.

A follow-up review of the enrollment change fixed the worker profile format (it
was JSON-escaped, which `uv run --env-file` rejects for non-ASCII paths), a
cleanup path that could strand a pending reservation, the CLI mislabeling a
database outage as "not configured", and enrollment audit events rendering as
task events. It also added expiry of stale reservations, a provisioning
deadline, OAuth token reuse during key revocation, a startup warning for a
combined server with no worker credentials, and withdrawal of an enrollment
whose network join failed. A second verification pass then tightened the
reaper to a single statement, made a failed key revocation retryable through
repeated withdrawal, kept static `WORKER_TOKENS` precedence over enrollment
history, validated local paths before enrolling, and made the CLI's cleanup
hold SIGINT so real repeated Ctrl-C presses cannot abort it, keep the profile
when withdrawal does not complete (even after a failed write, writing only
through the descriptor reserved at the start and never by pathname), and record
an unrevoked key on an already-expired reservation. 66 Python unit tests pass,
including a round trip of awkward profile values through `uv --env-file` and a
subprocess that receives two real SIGINTs under `asyncio.run`; the optional
PostgreSQL integration test passes with expiry, key retention, withdrawal retry,
and post-withdrawal registration checks; 19 frontend tests, TypeScript,
Prettier, and ruff pass.
