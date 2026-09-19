# Allocating tasks to workers

**For:** whoever builds the allocation step.
**Written:** 2026-09-19, from measurements on a three-machine fleet.

## Where your code goes

`store.claim()` — `backend/src/orchestrator/server/db/store.py:578`. Called from
`worker_connection.py:107` and `dwp.py:262`.

It is worth saying where it does *not* go, because the guess is natural and wrong: **not
near the Funnel.** Tailscale Funnel is `/ proxy http://127.0.0.1:8443` — TLS termination
and a public DNS name. It cannot see a task, a worker or a queue, and there is nothing
pluggable in it.

The whole of today's policy is five steps inside `claim()`:

1. the worker must be `alive`, unpaused, and seen within `UNHEALTHY_AFTER` (15s)
2. `occupied` — if it already holds an assigned or running task, return `None`
3. eligibility — `spec->>'kind' = ANY(caps.kinds)`, `runtime` equal, `vram_mib <=`,
   and `target_worker_id` either null, matching, or failed over
4. ordering — `ORDER BY created_at, id LIMIT 1`. Pure FIFO.
5. assign with a lease, a deadline and a generation bump

You are replacing 3 and 4.

**Keep the pull model.** Workers ask when free, so faster machines take more work with
nobody computing anything: measured split 69/17/15% against capacity shares of 67/19/14%.
It also cannot starve a worker the way a push planner can — the TypeScript control plane
pushed in connection-age order and filled each host to capacity before considering the
next, so with 12 tasks against 14 slots of fleet capacity the newest machine received
**0 of 60 tasks over five runs**. Pull is immune to that by construction.

## What you can read today

`workers`: `id`, `session_id`, `capabilities` (jsonb), `state`, `last_seen`, `paused`.

`capabilities` contains exactly three things — `kinds[]`, `runtime`, `vram_mib`.

`tasks`: `spec` (`kind`, `payload`, `requirements`, `max_attempts`, `timeout_seconds`,
`target_worker_id`, `allow_failover`), `state`, `generation`, `worker_id`, `lease_until`,
`deadline`, `result`, `attestation`, `failure`, `created_at`, `progress`, `started_at`.

That is the entire input available to a scheduling decision right now. There is no CPU,
no memory, no core count, and no measure of speed.

## What already arrives and is thrown away

This is the cheapest work available to you, because none of it needs a protocol change.
The agent sends all of this on every connection (`packages/protocol/src/messages.ts:24`,
`packages/agent/src/capability.ts`):

| field | sent | stored |
| --- | --- | --- |
| `adapters` | yes | yes, as `kinds` |
| `cpuModel` | yes | **discarded** |
| `logicalCores` | yes | **discarded** |
| `totalRamMb` | yes | **discarded** |
| `freeRamMb` | yes, and again in every heartbeat | **discarded** |
| `os`, `arch` | yes | **discarded** |
| `agentVersion` | yes | **discarded** |
| `consent.maxConcurrency` | yes | **discarded** |

The loss happens in one place — `dwp.py:290`:

```python
Capabilities(runtime="cpu", vram_mib=0, kinds=kinds)
```

`runtime` and `vram_mib` are hard-coded, and every other field is dropped on the floor.
Widening `Capabilities` and passing the rest through is a small change that gives you
core counts, memory and the worker's own advertised concurrency for free.

Note the last row especially: **the app already tells the server how many tasks it is
willing to run at once**, and the server ignores it in favour of a hard-coded 1.

## What exists nowhere, and has to be built

**Observed throughput per (worker, kind).** Nothing records how fast a machine actually
is. Capability alone cannot tell an M5 Pro from an M2 — both advertise the same `kinds`.
The raw material is already on `tasks`: `started_at`, a completion time, and `attestation`.
Deriving a rolling per-worker-per-kind rate from completed tasks is the missing input, and
without it "agentic allocation" can only reorder a queue, not match work to machines.

## Two structural limits to clear first

Both bind before any policy you write can matter.

**1. One task per worker, enforced in the schema.**

```sql
-- schema.sql:28
CREATE UNIQUE INDEX IF NOT EXISTS one_task_per_worker ON tasks(worker_id)
  WHERE state IN ('assigned', 'running');
```

So it is not only the `occupied` check at `store.py:587` — raising concurrency needs a
migration as well as a code change. Measured cost of the cap: a machine running 10
concurrent tasks delivered 264 sims/s at ~30 sims/s per task. At a concurrency of 1 the
same machine is roughly ten times slower.

**2. Assignment only happens on a heartbeat, every 5 seconds.**
`HEARTBEAT_INTERVAL = 5` (`shared/protocol.py:11`), and `claim()` is called only from the
heartbeat branch. A worker that finishes a 200 ms task waits out the rest of the interval.

Fixing either alone achieves little: more concurrency still idles waiting to be asked,
and faster asking still stops at one task. Claim again when a result is acknowledged —
the worker already reports finishing, so that is the natural moment.

Until both are lifted, a clever policy and plain FIFO produce nearly identical throughput,
because the binding constraint is the cadence rather than the choice. That makes it very
hard to tell whether a new policy helped. **Clear the limits first, then be clever.**

## Four things that will mislead you

1. **`cpuModel` is a vendor string, not a performance number.** One real value:
   `"AMD Ryzen 5 5600U with Radeon Graphics         "` — trailing whitespace included.
   Do not parse it for speed; measure instead.
2. **Runtime differs more than hardware does.** Identical deterministic arithmetic
   measured ~3.5x apart between Node/V8 and Bun/JSC. An allocator that learns "this
   machine is 3.5x faster" may be learning a JavaScript engine. Record `agentVersion` and
   runtime alongside any throughput figure.
3. **`freeRamMb` is a snapshot** taken at that instant, and it falls while work runs.
   It answers "can this task start", not "how much memory does this machine have" — that
   is `totalRamMb`, which is currently discarded.
4. **`logicalCores` is not usable parallelism.** The owner's consented `maxConcurrency` is
   the real cap, and it is deliberately lower on machines people are sitting in front of.

Also: `runtime`/`vram_mib` are hard-coded for app devices, so no GPU information reaches
the scheduler at all. Before designing around GPU placement, note that measured GPU paths
were *slower* for small models — CoreML's GPU backend ran at 0.86x of plain CPU, and
WebGPU at 0.23x, because dispatch overhead exceeded the arithmetic. GPU is a property to
match against task size, not a strict upgrade.

## A baseline to calibrate against

Same deterministic workload, 4,608 simulations, measured 2026-09-19:

| machine | cores | concurrency | throughput |
| --- | --- | --- | --- |
| Apple M5 Pro | 15 | 10 | 264 sims/s |
| AMD Ryzen 5 5600U | 12 | 2 | 76 sims/s |
| Apple M2 | 8 | 2 | 56 sims/s |
| all three together | | | 350 sims/s |

The fleet reached 350 against a theoretical sum of 395 — **89% efficiency**, the rest lost
to the tail (fast machines idling while a slow one finishes its last task) and one network
round trip per task. Tail loss is the thing a better allocator can actually recover:
prefer giving the last tasks of a job to the fastest free machine.
