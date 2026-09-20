# Run history in the desktop app

**For:** whoever implements the "past runs" panel in the DWP Agent window.
**Companion to:** `docs/worker-reported-data.md` (what already crosses the wire).
**Written:** 2026-09-19.
**Status:** implemented on `jack/run-history`. Rollout below is still to do.

## Goal

A compute machine's window today shows "Connected", the task it is running right now,
and the Pause / Turn on / Leave / Quit controls. It should also show what this machine
has run: a compact timeline of recent tasks, a per-run list (adapter, when, how long,
outcome), and how much CPU and memory the work took.

The panel ships through the existing signed auto-update. Nobody reinstalls anything.

## Decisions

1. **History is recorded on the machine, by the agent, and nowhere else.**
   The agent keeps no record of finished tasks today (the `running` map in
   `packages/agent/src/transport.ts` is deleted in `.finally()`), and the server's
   record is unusable for this: `tasks.worker_id` and `started_at` are nulled on every
   retry (`store.py:684-685`), the agent's own duration (`hostReportedMs`) is dropped by
   `verify_result` (`backend/src/orchestrator/shared/dwp.py:245`), nothing measures
   resource usage anywhere, and every server route the dashboard uses requires admin
   auth that the app does not hold. A local file needs no protocol change, no server
   change, no new route, works offline, and cannot affect scheduling.
2. **Everything is additive.** New fields on the state endpoint, a new card in the page,
   a new module. No existing field, route, message, or behaviour changes.
3. **History code can never fail a task.** Every filesystem call is wrapped; a disk
   error is logged once and the task result still goes out on time.
4. **No protocol or server change in this pass.** Sending resource figures to the server
   is a later step (see "Later").
5. **iOS is out of scope.** It does not self-update and does not share this code.

## What exists today (anchors)

| thing | where |
| --- | --- |
| task accepted, `running.set(...)`, `startedAt`, `t0` | `packages/agent/src/transport.ts:418-428` |
| success branch (`send('task.result', ...)`) | `transport.ts:441-482` |
| oversized result branch (`task.error`, `result_too_large`) | `transport.ts:452-463` |
| failure branch (`task.error`, `aborted` / `adapter_error`) | `transport.ts:483-494` |
| `.finally()` where the run is forgotten | `transport.ts:495-500` |
| `AgentState` type and `notify()` | `transport.ts:68-85`, `123-129` |
| `GET api/state` response | `packages/agent/src/gui.ts:680-715` |
| page HTML: status card, details card, controls card | `gui.ts:286-352` |
| page JS: `render(s)`, `tick()` polling `api/state` | `gui.ts:415-482` |
| CSP for the page | `gui.ts:675` (`default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'`) |
| agent home dir, file modes | `packages/agent/src/paths.ts:6`, `config.ts:53-56` (`0o700` dir, `0o600` file) |
| page parse check | `scripts/check-gui-page.ts` (`pnpm check:gui`) |
| tests | `packages/agent/test/*.test.ts`, run with `pnpm test` |

## Design

### 1. `packages/agent/src/history.ts` (new)

```ts
export type RunRecord = {
  taskId: string
  jobId: string
  adapter: string
  attempt: number
  startedAt: string        // ISO
  finishedAt: string       // ISO
  durationMs: number       // performance.now() delta, same figure as hostReportedMs
  outcome: 'ok' | 'error' | 'aborted' | 'result_too_large'
  errorClass?: string
  message?: string         // first 200 chars
  cpuMs: number            // user+system delta from process.cpuUsage(start)
  rssMb: number            // process.memoryUsage().rss at finish, MB
  shared: boolean          // true if another task overlapped, so cpu/rss are process-wide
  outputBytes?: number
  agentVersion: string     // installedRelease ?? AGENT_VERSION
}
```

- File: `join(AGENT_HOME, 'history.jsonl')`, one JSON object per line, written with
  mode `0o600` after `mkdirSync(AGENT_HOME, { recursive: true, mode: 0o700 })`, exactly
  as `saveConfig` does.
- `load(): RunRecord[]` reads the file once at startup, skips lines that do not parse,
  returns `[]` on any error.
- `record(r: RunRecord): void` appends to the in-memory array and to the file. When the
  in-memory array exceeds 1000 entries, trim to the newest 500 and rewrite the file.
  The whole function is `try { ... } catch { log once }`. Never throws.
- `recent(n): RunRecord[]` newest first.
- `summary(): { runs, ok, failed, busyMs, cpuMs, byAdapter: Record<string, { runs, busyMs, ok }>, since: string | null }`
  over the in-memory array.

`process.cpuUsage()` and `process.memoryUsage()` are available under both Node and the
Bun-compiled binary. Guard each call anyway and fall back to `0`.

### 2. Hook in `transport.ts`

- At accept (next to `t0`): `const cpu0 = process.cpuUsage()` and
  `const overlapped = running.size > 1`.
- Keep a local `outcome: RunRecord['outcome']`, `errorClass`, `message`, `outputBytes`.
  Set them in the three existing branches (success, `result_too_large`, catch). Do not
  reorder or delay the existing `send(...)` calls.
- In `.finally()`, after `running.delete`, call `history.record({...})` then `notify()`
  as today. The record call goes first so the window's next poll already sees it.
- `AgentState` gets one new field: `lastRunAt: number | null` (so `render` can say
  "last run 3 min ago" without loading history). Everything else the window needs comes
  from the history module directly in the GUI, not through `AgentState`.

### 3. `api/state` in `gui.ts`

Add, without touching any existing field:

```ts
history: {
  recent: history.recent(30),
  summary: history.summary(),
}
```

The page polls `api/state` on every `tick()`; the history module answers from memory,
so this costs nothing measurable.

### 4. The page

Insert a new card between the status card (`gui.ts:287`) and the details card
(`gui.ts:296`):

- **Heading line:** `Recent work` with a summary sentence:
  `12 runs · 3 failed · 18 min busy · last 4 min ago`. Empty state:
  `Nothing has run on this computer yet.`
- **Timeline strip:** one inline `<div>` per run (newest on the right), width
  proportional to `durationMs` (min 4px, max 25% of the strip), colour by outcome
  (reuse the existing `ok` / `bad` / `warn` dot colours). Title attribute with adapter,
  duration and outcome. Inline `style=` attributes are allowed by the CSP
  (`style-src 'unsafe-inline'`); inline event handlers are not, and no external
  resources may be loaded.
- **List:** the newest 10 runs, one row each: adapter in `<code>`, relative time,
  duration, outcome, `cpu 2.1 s · 340 MB`. Append `(shared)` when `shared` is true.
- **Per-adapter line:** `walker_evolution: 9 runs, 14 min · echo: 3 runs, 2 s`.

All text goes through `textContent` or a small escape helper; task ids, adapter names
and error messages must never be inserted as HTML. The whole script lives inside a
TypeScript template literal, so use string concatenation and no backticks or `${}` in
the page JS (see the comment at `gui.ts:445`). Run `pnpm check:gui` after every edit
to the page.

### 5. Untouched on purpose

`packages/protocol/*`, `backend/*`, `ios/*`, `packages/agent/src/update.ts`, the CSP
header, the port/token/lock logic, and the existing `api/*` POST routes.

## Do-not-break list

- Every existing `api/state` field keeps its name, type and meaning.
- `send('task.result' | 'task.error')` fires exactly as before, at the same moment.
- A missing, unwritable, corrupt or oversized `history.jsonl` produces an empty panel
  and one log line, never an exception on the task path or in `api/state`.
- The page still parses (`pnpm check:gui`) and still renders on a machine with no
  history and on one with 1000 runs.
- `pnpm typecheck` and `pnpm test` pass.

## Tests

`packages/agent/test/history.test.ts`, using a temp `DWP_HOME` like the existing tests:

- append then load round-trips a record;
- a corrupt line is skipped and the rest still load;
- the cap trims to 500 after 1001 records and the file matches memory;
- `summary()` counts, `busyMs` and `byAdapter` are right for a small fixture;
- `record()` with an unwritable path does not throw.

Manual check: `pnpm agent gui` against a local server, submit an echo task, watch the
card fill; cancel a task mid-run and confirm it shows as `aborted`; quit and relaunch
and confirm the history survives.

## Rollout

1. Merge. No version bump is needed: release strings already carry a content hash
   (`0.4.0+<hash>`), and the agent compares the full string.
2. `pnpm build:binaries` (desktop apps and raw binaries) and `pnpm release "run history"`
   (source installs), then copy the generated files into the server's
   `DWP_RELEASES_DIR`. See `docs/jack-integration.md:72-89`.
3. Confirm on one paired machine that it updates by itself and the card appears, then
   let the rest follow. Idle machines pick it up within about ten minutes.
4. Machines that will not receive it, and why: any that paired before signed releases
   (no pinned key; run `pnpm agent trust-updates` there), iOS (no self-update), and
   the Windows laptop in `docs/issues/02` (still running an old binary from a different
   path; replace it by hand once).

## Later

- Send `cpuMs` and `rssMb` to the server: optional fields on `TaskResult`
  (`packages/protocol/src/messages.ts:76`), keep them in `verify_result`, and add
  columns on `tasks`. Also stop dropping `hostReportedMs`.
- A per-worker history route for the dashboard.
- Verify that a macOS `.app` still launches after the binary swap invalidates its ad-hoc
  code signature (`scripts/lib/apps.ts:219`); nothing checks this today.
