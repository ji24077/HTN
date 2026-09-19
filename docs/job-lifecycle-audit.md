# Job lifecycle audit — September 19, 2026

Scope: uploaded Python simulations, preprocessing and supervisor decisions, durable
scheduling, the current local Python workers, cancellation, and dashboard visibility.
This is a concrete failure review of the implemented prototype, not a claim that
arbitrary uploaded programs or an Internet-wide worker fleet are production-safe.

## Findings and changes

| Failure scenario | Change / protection | Verification |
| --- | --- | --- |
| Hundreds of cancellations hold the global database lock across per-task network requests | Cancel tasks in one UPDATE and record all task, execution, and supervisor events in one SQL statement | Cancel 313 batches with simulated database latency; check every audit entry, released reservations, idempotent replay, and rejection of late results |
| Worker replacement repeatedly times out while retargeting hundreds of queued batches | Bulk retarget only rows still queued; preserve seeds, package hash, and attempt counts | Replace one worker with 157 affected batches while another worker heartbeats |
| Queued tasks hide an exhausted batch because task IDs sort lexically | Examine cancelled and exhausted tasks before waiting on pending siblings | Exhaust a late-sorting batch and verify immediate failure plus cancellation of 312 siblings |
| Diagnostic traffic blocks scheduling and heartbeat updates | Diagnostic inserts use their own transaction and retain historical-assignment authorization, replay deduplication, and generation/session-fenced progress | Existing database execution replay and superseded-session tests |
| Reservation updates repeatedly wake the same pending job | Back off pending checks, renew reservations only near expiry, and avoid acquiring the global lock when capacity is already reserved | Pipeline and pause/resume integration tests |
| Database timeouts consume adaptation/model retry limits | Transient infrastructure errors defer the same durable phase without resetting the deadline or consuming adaptation/model failures | Inject timeout and verify unchanged phase, budget, deadline |
| One slow model call blocks every other job | Run at most two job checks concurrently; retain the per-job advisory lock and cancel both on server shutdown | Block one job and observe another job's check |
| A proposal returns after cancellation, pause/resume, or the deadline | Recheck persisted revision, control state, and deadline inside the commit transaction | Cancel during inference and delay a proposal past its deadline |
| Expired jobs receive newly assigned work | Exclude expired/terminal simulation jobs from claims; cap assignment leases/deadlines at the submission deadline | Database and pipeline regression checks |
| Final validation accidentally runs twice on the same replacement worker | Reserve distinct IDs, exclude occupied validation targets, and check actual successful assignments | Worker replacement during two-worker validation |
| Retry changes random inputs or counts a trial twice | Retain seed identities and validated artifact; reject missing, duplicated, malformed, and unexpected trial identities | Complete 10,000 trials in 313 batches and verify exactly 10,000 unique results |
| Original code fails and the agent tries to repair it | Original and reference failures terminate the submission; only the candidate may be repaired | Broken-original and broken-reference tests |
| A bad candidate passes only its previous test set | Refresh original reference cases after repairs and draw a separate final sample after freezing the candidate | Failed adaptation and failed final-gate regressions |
| Cancel changes database state but leaves processes or files behind | Worker revocation cancels execution, kills the POSIX process group, reaps the process, and removes its temporary workspace; worker reports `cleaned` only after cleanup succeeds | Cancel a script with a child process and verify both processes and the directory are gone |
| Script exits but its child holds stdout/stderr open | Observe leader exit and kill descendants before waiting for inherited pipes to close | Successful script spawning a long-lived child completes promptly |
| Cancellation arrives while a subprocess is being created | Shield spawn, obtain the process handle, then kill/reap it before removing files | Cancellation implementation review plus process-cleanup regressions |
| Worker process is killed before its finally block runs | POSIX launcher watches its original parent, removes its own attempt directory, and kills its process group after reparenting | Kill the worker process and verify its launcher, descendant, and directory disappear |
| Worker disconnects before reporting cleanup | Preserve cancellation separately from cleanup confirmation; continue polling and show the pending worker instead of claiming cleanup succeeded | Post-cancellation cleanup event and UI pending-to-confirmed test |
| Late diagnostics arrive after a task is terminal | Continue brief log polling and accept authorized historical execution events without changing the task outcome | Cleanup event arrives after cancellation; result remains unset |
| Pipeline progress is mistaken for trial completion | Label the pipeline percentage and show completed/total trial counts separately | UI checks 896 / 10,000 independently of pipeline stage |
| Artifact URL is reused after cancellation or lease expiry | Require active assignment, unexpired lease/deadline, assigned worker ID, scoped bearer token, and matching artifact hash | Authenticated upload/artifact integration tests |
| ZIP files escape extraction paths or conflict as files/directories | Reject traversal, symlinks, special/encrypted entries, case collisions, parent/file collisions, and size/count excess | Archive validation unit tests |
| Test records accumulate in Supabase | Back up and remove the explicitly selected completed test/demo jobs; preserve every uploaded job and all account/device configuration | Row counts and preservation checks after cleanup |

## Agent-owned execution policy

New submissions profile the original on a worker before selecting execution policy.
The agent explicitly chooses preprocessing placement, profile and validation case
counts, probe/validation/batch timeouts, aggregation strategy, and each batch's
worker and trial count. After every completed wave it can revise the remaining
schedule using measured timings. The backend enforces capacity, submission budgets,
immutable validated code, and unique trial identities.

| Previous policy problem | Change | Verification |
| --- | --- | --- |
| A default of 32 trials created 313 tasks for cheap 10k simulations | Required measured schedule; one task may cover all trials | 10k trials in one explicitly chosen task |
| Two full-run workers and equal round-robin batches were assumed | Agent chooses each batch independently | Unequal 3/7 first wave, then 14 trials on one worker after reviewing measurements |
| Validation always used 16 cases and execution always used 120 seconds | Agent supplies counts and timeouts within submission limits | Distinct 7/9 validation sizes and per-batch deadlines in integration tests |
| Raw trial results dominated transport | Inline, ordered values, or validated partial aggregation | Wrong partial merge rejected before launch; ordered values preserve seed order |
| Retry could silently use another worker | Agent selects replacement; retain task identity, code, and seed range | Lost full-run worker replaced through an explicit model placement decision |
| Ready work waited for the next heartbeat | Dispatch immediately after acknowledging a completed task | Protocol regression verifies acknowledgment then assignment without heartbeat |
| Users could not see the basis for scheduling | Show rationale, decision history, and compute/execution/wall measurements | Dashboard rendering tests |

First-version persisted jobs retain their original plan for compatibility. New
submissions use the measured planning loop.

## Cancellation semantics

**Cancel & clean up workers** immediately prevents new assignments, cancels outstanding
tasks, releases reservations, and rejects late results. Connected workers receive the
revocation through the existing heartbeat/assignment protocol. They stop project
processes and delete the managed temporary project directory. Cleanup confirmation
can arrive after the database cancellation and is displayed separately.

An offline machine cannot provide immediate confirmation. The UI keeps its cleanup
status pending. Worker execution journals remain bounded diagnostics; cancellation
removes execution inputs, generated files, and processes in the managed workspace,
not arbitrary files elsewhere on the user's computer. Original uploaded artifacts
and accepted job results remain in the server's job history.

## Supabase cleanup performed

Preserved both uploaded simulations, including the cancelled 10,000-trial job.
Removed seven old test/demo job groups: 16 tasks, 82 execution events, 51 task audit
events, 15 supervisor events, nine supervisor reviews, and two supervisor job records.
Removed one expired pairing record and one expired assertion; marked one superseded
review failed. Accounts, credentials, device registrations, and application settings
were preserved. A private local JSON backup was saved before deletion.

## Remaining boundaries

- Personal-machine sandboxing and dependency installation remain deferred, as agreed.
  Uploaded code runs under the local user's permissions; this is not a security
  boundary. Programs that deliberately escape the process group or write outside
  their workspace require an OS sandbox/container design.
- Process-tree cleanup and parent-death handling are verified on POSIX. Windows
  needs Job Objects or a corresponding sandbox before offering the same guarantee.
- Machine power loss, filesystem failure, or failure to write/send a cleanup receipt
  can leave cleanup unconfirmed. We do not turn absence of a heartbeat into proof
  that local files were removed.
- The global scheduler lock remains for ownership correctness. Bulk mutations and
  independent diagnostics remove the observed bottleneck, but large fleets still
  need finer locking and load tests against their actual database latency.
- Sample equivalence is not a mathematical proof of program equivalence. Independent
  seeded CPU simulations are the current supported workload; coordinated training,
  arbitrary stateful programs, and automatic dependency repair are separate work.

## Verification results

- Backend baseline including measured-planning regressions: 197 tests, 195 passed
  and two intentionally skipped; an additional agent-directed replacement regression
  is covered by the focused six-test planning suite.
- Frontend suite: 49 passed; TypeScript and production build passed.
- A live temporary fixture executed on the configured local worker using the real
  backend, Supabase database, transport, artifact download, and execution journal.
  Cancellation committed in 2.31 seconds; the worker's cleanup confirmation arrived
  within 4.6 seconds. The temporary project directory was verified absent. The test
  fixture was removed afterward, leaving the two uploaded jobs intact.

## Live agent-planning check

A new 10,000-trial upload (`sim-f74e9867e88d45b0b7facf04fd856052`) completed
through the configured model, Supabase backend, and local workers. The agent chose
128 local validation cases, 256 independent cases across two workers, then one
10,000-trial inline task on worker-1 with a 30-second timeout. Both validation gates
passed. The full task measured 0.051 seconds computation, 1.55 seconds worker
execution, and 7.81 seconds from task creation to accepted result; these numbers
exclude preprocessing/model calls. Final output: 7,789 hits / 10,000 trials,
hit rate 0.7789, returned in a 167-byte structured result before parent metrics.
The dashboard exposes the agent rationale, schedule, and measured costs.
