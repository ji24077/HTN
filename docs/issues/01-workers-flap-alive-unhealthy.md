# Workers cycle alive → unhealthy → alive, and it kills running work

**Status:** fixed 2026-09-19. Observed the same day on a five-machine fleet.
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

## What it turned out to be

**A slice whose result is over 64 KiB closed the worker's connection.**

`bounded_json` (`shared/protocol.py`) caps one payload or one result at `JSON_LIMIT` =
64 KiB — half the 128 KiB frame, because a `task.result` carries the output twice, once
as the value and once inside the hash it is signed with. A walker slice of 1500 gaits
serialises to about 102 KiB:

```
  100 seeds ->   6,932 bytes    ok
  500 seeds ->  34,532 bytes    ok
  800 seeds ->  55,232 bytes    ok
 1000 seeds ->  69,032 bytes    over
 1500 seeds -> 104,032 bytes    over
```

The frame itself was under the 128 KiB transport limit, so it arrived, parsed, and was
then refused by validation — as a bare `ValueError`, which `connect()` reads as "this
connection is broken" and answers with close 1008. From there:

1. the socket closes, so `disconnect()` writes **alive -> unhealthy**
2. the agent dials again a second later; `register()` writes **unhealthy -> alive** and
   fails everything that worker held as **`worker_reconnected`**
3. the slice is re-queued, the next machine computes it, produces the same oversized
   result, and is disconnected in turn

So it was one poison-pill slice walking the fleet, not a presence bug. That also explains
the two things that looked strangest: it needs no network (the loopback agent hits the
same code path), and the fleet flapped in step rather than independently.

The correlation in "small slices survive, large ones do not" was real; the reason was not
the reconnect gap. Small slices survive because their results fit.

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

## The fix

- `shared/protocol.py` — `bounded_json` raises `JsonTooLarge`, a `ValueError` subclass, so
  the size of one result can be told apart from a malformed frame.
- `server/dwp.py` — `Connection.reject_result()` fails that one task as
  **`result_too_large`** and keeps the socket open. **Not retryable:** these adapters are
  deterministic, so the same slice produces the same oversized output everywhere, and
  retrying only moves the failure to the next machine. A result that fails *signature*
  verification still closes the connection, which is the right answer for a forgery.
- `packages/agent` — the agent measures its own output first and reports
  `errorClass: result_too_large` instead of shipping something that will be thrown away,
  mirroring what the Python worker already does at `worker/agent.py:59`. The server treats
  that class as non-retryable too.

Regression tests: `test_oversized_result_fails_one_task_and_keeps_the_connection` and
`test_device_reporting_result_too_large_is_not_retried` in `tests/unit/test_dwp_gateway.py`.

**Operationally:** keep a walker slice at or under ~800 seeds. Over roughly 950 the result
cannot be delivered, and the job now fails immediately with `result_too_large` on the task
rather than taking the fleet with it.

## How to reproduce

Run a fleet with at least two workers and submit a `walker_evolution` task whose slice is
large enough that its *result* is over 64 KiB — 1500 seeds does it, and duration is
irrelevant. Then watch:

```sql
select at, entity_id, previous_state, new_state
from events where entity = 'worker' and at > now() - interval '5 minutes'
order by at desc;
```

A fleet that looks idle still flaps, because the re-queued slice keeps being picked up;
genuinely idle workers with nothing queued do not.
