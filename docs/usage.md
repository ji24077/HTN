# Run usage and spending caps

Each run (`job_id`) tracks an estimated CAD cost in PostgreSQL. For signed-in
accounts, the top-right **CA$ credit** figure shows granted prototype credit minus
estimated usage from that account's jobs. It updates with the live fleet stream.
Local demo/automation sessions show fleet estimated spend instead. The task details
panel shows a run's own total and optional cap, including when model supervision
is disabled. There is no payment processing or automatic paid top-up.

Credit grants are operator-only ledger entries with unique receipts, so retrying
the same grant cannot add credit twice. Account identity comes from verified
authentication, never a submitted account ID. New jobs and simulation uploads
record their creator; internal retries and simulation child tasks retain that
run's billing account. Existing unowned jobs are not charged retroactively.
The balance is a tracking figure and may go below zero; optional per-run caps
remain the mechanism for stopping work. Credit and usage survive server restarts.

Rates depend on the executing machine's reported CPU cores and RAM:

**CAD/hour = $0.002 × logical cores + $0.001 × RAM in GiB**, with a $0.01 floor
and a $0.50 ceiling per machine-hour. These are configurable prototype estimates,
not cloud provider prices or measured CPU/RAM utilization.

| Machine | Estimated hourly rate |
| --- | --- |
| 4 logical cores, 8 GiB | $0.016 |
| 8 logical cores, 16 GiB | $0.032 |
| 16 logical cores, 32 GiB | $0.064 |

Unknown specs use one core and one GiB (normally hitting the $0.01 floor) and are
marked `inferred` in the record's `pricing_basis`. Zero reported RAM is treated as
unknown. Each record saves the normalized specs, reported machine details, pricing
coefficients, and resulting hourly rate. CPU model names are descriptive, not a
performance benchmark. GPU memory adds no premium to this CPU/RAM estimate; a
future GPU workload needs an appropriate GPU pricing model.

Change the coefficients in the application's private database schema:

```sql
UPDATE usage_pricing
SET hourly_rate_cad = NULL, core_hour_cad = 0.002,
    ram_gib_hour_cad = 0.001, minimum_hour_cad = 0.01
WHERE id = true;
```

An optional flat override remains available for testing or custom pricing:

```sql
UPDATE usage_pricing SET hourly_rate_cad = 0.02 WHERE id = true;
```

Set the override back to `NULL` to restore machine-spec pricing. Upgrading switches
the old default $1 rate to machine-spec pricing once, preserving non-default flat
overrides and all previously saved attempt rates.
If the pricing row is accidentally removed, new attempts use the default machine
coefficients until configuration is restored; metering and caps remain active.

Each acknowledged execution attempt snapshots the current specs and rate. Subsequent rate
or hardware changes do not reprice either running or completed attempts. Rates and caps allow
six decimal places; costs retain twelve decimal places. API money values are
decimal strings. The dashboard rounds only for display.

## What is counted

- Worker execution from the server's accepted start acknowledgement until success,
  failure, retry, or cancellation. Each retry is a separate record.
- Execution time is summed across parallel workers. Waiting in the queue and
  assignments that never acknowledge a start are free.
- Disconnected work accrues at most through its last valid lease or execution
  deadline. Server time is authoritative; worker-reported durations are not used.
- Simulation child tasks count; the simulation's coordinating root row does not.
- Model/API calls, storage, idle machines, and time after server cancellation are
  excluded. This is an estimate of authorized execution, not an external invoice.

Records are finalized transactionally with task state, including supervisor and
simulation cancellation. Duplicate acknowledgements and results cannot add cost.
Records survive restarts. Upgrading starts tracking existing supervised in-flight tasks at
migration time; historical costs before installation are not reconstructed.
Legacy tasks that predate job supervision remain executable but have no run usage summary.

## Assistant and API

The job creation forms include an optional **Max spend (CAD)** text field. Leave
it blank for no cap; enter zero to prevent work from starting. Both uploaded
simulations and built-in tasks save this cap atomically with the new run.

Ask the fleet assistant “show the cost of run X”, “cap run X at $2”, or “remove the
cap for run X”. It uses `get_run_usage` and `set_run_usage_cap`. Cap changes are
restricted to authenticated fleet administrators and produce supervisor audit
events. The autonomous job supervisor can read usage, but has no tool to raise
or remove the user's cap.

- `GET /v1/jobs/{job_id}/usage` returns the total, cap, remaining allowance, and up
  to 100 attempt records. Use `after=next_cursor` to read subsequent pages; the
  total always covers all records.
- `PUT /v1/jobs/{job_id}/usage-cap` accepts `{"cap":"2.00"}`. Explicit `null`
  removes a cap, zero stops outstanding work, and omitting the field is an error.
- `POST /v1/tasks` and the assistant's `submit_tasks` accept an optional
  `usage_cap` alongside `tasks`. A capped submission must contain one job ID.
  The cap is saved in the same transaction as the tasks, before dispatch is
  possible. Repeating a submission never overwrites an existing cap.
  Replaying existing task IDs with identical specifications returns those tasks
  even if the cap has since changed. Adding new tasks still checks the current cap.
- Supervisor status includes a `usage` summary for the UI and job supervisor.
- `POST /v1/simulations` accepts the same optional `usage_cap` with the upload.

The fleet assistant retains its prior handling of `submit_tasks.instructions`:
that field does not set supervisor instructions. The HTTP submission API's
existing instructions behavior is unchanged.

Caps default to absent. A cap covers the entire run, including retries and work
already completed. Lowering it below accrued cost cancels outstanding work
immediately. Changing a cap never resets usage. A cancelled run cannot be restarted
by raising or removing its cap; submit a new run when new work is intended.

## Enforcement

The scheduler checks the aggregate cost before dispatch, on heartbeats, and during
its one-second recovery loop. These checks share the database scheduling lock
across server processes. Reaching the cap atomically cancels outstanding tasks,
releases worker reservations, and fences stale supervisor/simulation decisions.
Workers receive cancellation through the existing heartbeat/lease protocol.

Completed costs and attempt counts are maintained transactionally in private
per-run and fleet totals. Header updates and cap checks add only indexed live
records to those totals. A worker heartbeat checks its own run; dispatch and
the recovery loop still check all eligible capped runs. Lease extensions update
live records as needed without rewriting unchanged values or completed totals.
Changing a cap still invalidates an in-flight supervisor decision so it cannot
act on an outdated budget.

This is a **soft spending cap**: cost can exceed it between checks, and workers
may take another heartbeat to stop. It does not reserve a maximum budget for each
parallel task or guarantee a hard ceiling. Database outages also delay enforcement;
worker leases remain the fallback. The displayed total is not clamped to the cap.

## GPU follow-up

GPU pricing is not implemented or validated on GPU hardware. The current rule
estimates CPU/RAM only, even if a worker advertises a GPU runtime or VRAM. Before
enabling GPU workloads:

1. Extend worker capabilities with GPU model, device count, per-device VRAM, and
   stable device identifiers. Record which devices (or device partitions) each
   attempt actually receives; installed GPUs alone must not increase its price.
2. Add a versioned GPU rate table and select a rate from the assigned hardware
   and workload runtime. Calibrate the estimates once hardware is available.
   Do not apply the CPU-only $0.50/hour ceiling to GPU rates by default, or treat
   equal VRAM as evidence that two GPU models have equal performance or cost.
3. Snapshot the GPU allocation, pricing model/version, and rate in each attempt's
   `pricing_basis`. Preserve the existing saved rates and totals during migration.
   The existing usage ledger and per-run caps can sum these different rates.
4. Decide whether GPU time means allocated wall time or measured active time.
   Match metering, cap checks, and the UI description to that choice, and avoid
   counting shared memory twice on unified-memory devices.
5. Add integration tests for mixed CPU/GPU runs, multiple devices, retries on a
   different GPU, unknown models, migration, and cap cancellation. CPU work on a
   GPU-equipped host must retain CPU pricing unless GPU resources are allocated.
6. Validate on real hardware: execute a known GPU workload, compare recorded time
   with its allocation, interrupt/disconnect a worker, and verify that reaching a
   cap stops the GPU process and releases its devices. Mocked tests alone do not
   establish GPU execution or cancellation correctness.

## Verification

```sh
RUN_DATABASE_TESTS=1 uv run --project backend --extra demo python -m unittest backend.tests.integration.test_usage -v
npm --prefix frontend test
```

Database tests use a temporary local PostgreSQL server and never connect to the
configured application database.
