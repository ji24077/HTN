# Uploaded Python simulation loop

Submit Python files or a ZIP plus a description through **New job → Upload project**.
The initial submission authorizes adaptation, validation, allocation, and execution.
The original uploaded bytes remain in an immutable, job-scoped artifact. The agent
never executes user code in the backend process and never repairs the original.

## Other uploaded workloads

Rendering, model training, downloadable files, and the shared `/v1/jobs` upload
flow are described in [Uploaded compute jobs](uploaded-projects.md). The
simulation-specific validation pipeline below remains in use for simulations.

## Lifecycle

1. The agent inspects the upload and explicitly chooses the preprocessing worker,
   original smoke check, reference harness, immutable aggregation function, profile
   sample count, and probe timeout. It runs the unchanged original on that worker.
   An original-code error ends the job; the agent never repairs the original.
2. Profile fresh original trials on the worker. Record compute time, worker execution
   time, end-to-end task time, and output size. The agent uses these measurements and
   available device capabilities to choose validation sample counts, validation
   timeouts, and an aggregation strategy.
3. Generate a candidate defining `run(seed, parameters)`, optionally
   `run_many(seeds, parameters)`. Freeze its hash, draw fresh reference seeds, and
   test both trial outputs and aggregation against the original on one worker.
   On failure, the agent may revise the candidate and policy within the retry budget;
   every revision gets fresh original reference cases. Original files, reference
   semantics, parameters, and comparison tolerances remain unchanged.
4. After local validation passes, the agent chooses two distinct workers for a fresh
   independent validation set. Both trial results and aggregation must match.
5. The agent schedules the validated candidate using measured timings. Every batch
   explicitly names a worker, trial count, and timeout. It also chooses the final
   aggregation worker and timeout. A wave can cover all remaining trials or only
   part; after each wave the agent receives measurements and plans the remainder.
   Already accepted seed ranges cannot be counted again.
6. Aggregate on a worker and return the result. `inline` executes all trials and
   aggregation in one task; `values` combines ordered batch outputs; `partial`
   uses validated candidate `reduce` and `merge` functions. Large jobs need not send
   every raw trial value back to the server.

There is no default batch size or full-run worker count for new submissions. A
10,000-trial job can run as one task if coordination would cost more than computing
it locally. Two workers remain required for the independent validation gate.
Persisted first-version jobs retain their original execution policy for compatibility.
Device IDs are separate local worker processes in development, not evidence of
separate physical devices. The current worker runtime supports CPU Python; GPU
execution and richer hardware inventory remain separate work.

The comparison requires identical JSON structure, exact strings/booleans, and
numerical `rel_tol=1e-8`, `abs_tol=1e-10`. Trial identities must be complete and
unique. Randomness must be exposed as independent reproducible seeded trials;
the agent asks for clarification if it cannot preserve those semantics. Passing
sample tests is evidence of equivalence on those samples, not a proof of correctness.

## Persistence and controls

PostgreSQL stores phases, candidate versions, reference samples, outcomes, artifact
hashes, limits, and questions. Task submission and phase transitions are atomic.
An advisory lock serializes each preprocessing loop; revision and cancellation
checks fence model responses. Failed inference can be retried without launching
unrecorded work. Worker execution retries preserve the task's inputs and seed set.
The original runtime deadline and adaptation budget survive process restarts.

The existing supervisor observes the same job throughout. It can pause/resume or
cancel the whole job, but cannot bypass the validation gate, independently retry
pipeline tasks, or reserve extra workers. Final status uses the pipeline outcome,
not failures from discarded candidate versions. Completion, cancellation, and
unrecoverable failure release reservations.

## Local worker setup

Use the existing Python worker with `WORKER_EXECUTOR=python_project`. It advertises
`python_project` alongside `stub`, downloads only its assigned artifact, checks
its digest, runs Python in a fresh temporary directory, and returns structured
outputs plus stdout/stderr through existing task telemetry. Authentication still
uses the existing `WORKER_ID`, `WORKER_TOKEN`, and `SERVER_URL` settings.

This is deliberately a **local development subprocess runner**, not an isolation
boundary for untrusted public uploads. Production worker sandboxing and deployment
are deferred. Python requirements from uploaded manifests and the agent's dependency
plan are installed in a private environment before execution (see
[automatic dependencies](uploaded-projects.md#automatic-python-dependencies)).
The child receives a minimal environment,
not the model/server/worker credentials. Cancellation kills its process group on
POSIX. Dependency or entry-point failures are returned, not patched.

Use `examples/monte_carlo_upload.py` for a standard-library-only test. Start two
local worker processes with distinct configured IDs, then upload that file and
ask for 100 trials and the hit rate. No second launch approval is required.

## Resource envelopes

- Upload: 8 MiB expanded, 100 files, 128 KiB Python source; ZIP traversal,
  duplicate paths, symlinks, encrypted entries, and reserved driver names rejected.
- Default submission: three adaptation attempts, four-worker maximum, 30-minute
  total deadline including time waiting for capacity or clarification.
- New plans require explicit decisions. Bounds protect resources: 512 tasks per
  scheduling wave, 1,024 cases per profile/validation set, and nonoverlapping trial
  identities within the 31-bit seed domain. There is no 64-trial batch cap.
- Task timeouts come from the agent and are clipped to the remaining submission
  deadline. Infrastructure retries retain their existing three-attempt safety bound.
- Task responses remain bounded at 48 KiB; artifact bundles at 16 MiB. The agent
  sees these limits and can choose inline or partial aggregation to avoid raw-result
  transfer overhead.
- Artifacts use PostgreSQL bytea for this prototype. A separate object store and
  production worker runtime can replace this without changing the agent loop.

## API

- `POST /v1/simulations`: authenticated JSON containing `request_id`, `description`,
  and `files: [{name, content}]` with base64-encoded file bytes. Reusing the same ID
  and body returns the existing job. Different content with that ID is a conflict.
- `GET /v1/simulations/{job_id}`: phases, plan, policy, schedules, measurements, decision history, versions, checks, questions, executions.
- `POST /v1/simulations/{job_id}/answer`: answer a pending clarification.
- `GET /v1/simulations/{job_id}/artifacts/{digest}`: authenticated original or adapted files.
- Worker-only artifact download requires a task-scoped bearer token, matching
  assigned worker ID, matching bundle hash, and a currently active assignment.

## Cancellation and failure audit

The simulation detail view includes **Cancel & clean up workers**, completed trial
counts, and explicit pending/confirmed worker cleanup. Connected local workers stop
the execution process group and remove the managed temporary project files before
confirming cleanup. Offline cleanup remains unconfirmed until evidence arrives.
See [the lifecycle audit](job-lifecycle-audit.md) for the fixes, regressions, and
remaining prototype boundaries.
