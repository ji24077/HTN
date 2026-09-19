# Workers cycle alive → unhealthy → alive, and it kills running work

**Status:** open, cause unknown. Observed 2026-09-19 on a five-machine fleet.
**Impact:** long tasks cannot complete. Three of four slices failed as `worker_reconnected`.

## What happens

Connected machines flip between `alive` and `unhealthy` every few seconds, recover on
their own, and do it again. From the `events` table:

```
14:40:18  iPhone    unhealthy -> alive
14:40:17  iPhone    alive -> unhealthy
14:40:06  iPhone    unhealthy -> alive
14:40:05  iPhone    alive -> unhealthy
14:40:03  Mac Air   unhealthy -> alive
14:39:57  Mac Air   alive -> unhealthy
14:39:56  local     alive -> unhealthy
```

Over fifteen minutes, with the fleet otherwise idle:

| worker | went unhealthy | came back |
|---|---|---|
| Mac Air `691cf6cf` | 4 | 4 |
| iPhone `0cf0745a` | 4 | 5 |
| local `5e6b986c` | 2 | 2 |
| Windows `4ec957dd` | 0 | 0 (was offline throughout) |

Note the local agent flaps too, over loopback, which is the most interesting fact here.

## Why it matters

A reconnect cancels the worker's assignment. A job of four slices, 1500 gait simulations
each, produced:

```
slice 0  failed   worker_reconnected
slice 1  running  iPhone
slice 2  failed   worker_reconnected
slice 3  failed   worker_reconnected
```

Small slices survive because they finish inside the gap; large ones do not. That puts the
system in a bind, because small slices have their own problem (see issue 02).

## What has been ruled out

**Not a heartbeat interval mismatch.** The server advertises `heartbeatSeconds` =
`HEARTBEAT_INTERVAL` = 5 in its hello-ack, and the agent adopts it
(`packages/agent/src/transport.ts:367`). The server marks a worker unhealthy after
`UNHEALTHY_AFTER` = 15 s. That is a 3x margin, not a race.

**Not an adapter blocking the event loop.** This was the first theory and it is wrong.
`packages/agent/src/adapters/walker.ts:28` yields with `setImmediate` every 8 gaits, so
the heartbeat timer gets to run during a long slice.

**Not the tunnel alone.** The local agent (`5e6b986c`), which reaches the server over
127.0.0.1 with no Funnel and no internet, flapped twice in the same window. Whatever this
is, it is reachable without a network in between.

## Where to look

- `backend/src/orchestrator/server/db/store.py` — `heartbeat()` and `reconcile()`. The
  reconcile loop runs once a second (`scheduler.py:17`) and takes an advisory lock; if a
  heartbeat write and a reconcile pass interleave badly, a worker whose beat arrived could
  still be read as stale.
- Whether `last_seen` is written on every heartbeat frame or only on some. A heartbeat
  that is accepted but does not advance `last_seen` would look exactly like this.
- `worker_reconnected` — find what sets it, and whether an *unhealthy → alive* transition
  is treated as a new session even when the socket never closed. If the socket survived,
  cancelling the assignment is the bug rather than the symptom.
- The `events` rows themselves: an `alive → unhealthy → alive` pair one second apart
  suggests the transition is being written on a read rather than on an actual timeout.

## How to reproduce

Run a fleet with at least two workers, submit a `walker_evolution` task with a slice large
enough to take more than ~15 seconds on the slowest machine, and watch:

```sql
select at, entity_id, previous_state, new_state
from events where entity = 'worker' and at > now() - interval '5 minutes'
order by at desc;
```

Idle workers are enough to see the flapping; the failures need a long task.
