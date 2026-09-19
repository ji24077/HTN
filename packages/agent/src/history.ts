/**
 * What this computer has actually done.
 *
 * The agent forgets a task the moment it finishes — `running.delete(...)` in
 * `transport.ts` is the whole of its memory — and the server's record cannot stand in
 * for one: `tasks.worker_id` and `started_at` are nulled on every retry, the agent's own
 * duration is dropped by `verify_result`, nothing measures resource usage anywhere, and
 * every route that would expose it needs admin auth this app does not hold. So the
 * machine keeps its own record, in a file beside its config.
 *
 * The one rule that matters here: **this module may never fail a task.** It sits in the
 * `.finally()` of the work path, so a full disk or a read-only home has to cost a log
 * line and nothing else. Every filesystem call is wrapped and `record()` cannot throw.
 */
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { createLogger } from '@dwp/protocol'
import { AGENT_HOME } from './paths.ts'

const log = createLogger({ component: 'history' })

export type RunRecord = {
  taskId: string
  jobId: string
  adapter: string
  attempt: number
  /** ISO, the same string sent to the server as `startedAt`. */
  startedAt: string
  finishedAt: string
  /** `performance.now()` delta — the same figure as `hostReportedMs`. */
  durationMs: number
  outcome: 'ok' | 'error' | 'aborted' | 'result_too_large'
  errorClass?: string
  /** First 200 characters only; an adapter error can be arbitrarily long. */
  message?: string
  /** user+system delta from `process.cpuUsage(start)`, milliseconds. */
  cpuMs: number
  /** `process.memoryUsage().rss` at finish, MB. */
  rssMb: number
  /**
   * Another task overlapped this one, so `cpuMs` and `rssMb` are process-wide rather
   * than this task's. Shown rather than hidden: a figure that silently means two
   * different things is worse than one labelled honestly.
   */
  shared: boolean
  outputBytes?: number
  agentVersion: string
}

export type HistorySummary = {
  runs: number
  ok: number
  failed: number
  busyMs: number
  cpuMs: number
  byAdapter: Record<string, { runs: number; busyMs: number; ok: number }>
  /** `startedAt` of the oldest run still held, or null when there are none. */
  since: string | null
}

/**
 * Keep the file bounded without keeping a rolling index of it.
 *
 * A year of busy days is a few hundred kilobytes, which is not the worry; the worry is
 * a runaway loop appending for a week. Trimming well below the cap means the rewrite
 * happens once every 500 runs rather than on every run past 1000.
 */
const MAX_RECORDS = 1000
const TRIM_TO = 500

/** Resolved lazily so a test can point the module at a temp directory. */
let file = join(AGENT_HOME, 'history.jsonl')
let runs: RunRecord[] = []
let loaded = false
/** One log line per process, not one per failed write. */
let complained = false

function complain(event: string, err: unknown): void {
  if (complained) return
  complained = true
  log.warn(event, { file, detail: err instanceof Error ? err.message : String(err) })
}

/**
 * Read the file into memory.
 *
 * `path` exists for tests, which need a fresh file per case; nothing in the app passes
 * it. A line that does not parse is dropped rather than failing the load — a half-
 * written final line after a power cut must not cost the other 900 records.
 */
export function load(path?: string): RunRecord[] {
  if (path !== undefined) file = path
  loaded = true
  runs = []
  let text: string
  try {
    text = readFileSync(file, 'utf8')
  } catch (err) {
    // A file that has never been written is the ordinary case on a new machine, and
    // saying so every time would train people to ignore the log.
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') complain('history.unreadable', err)
    return runs
  }
  let skipped = 0
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue
    try {
      runs.push(JSON.parse(line) as RunRecord)
    } catch {
      skipped += 1
    }
  }
  if (skipped > 0) log.warn('history.damaged_lines', { file, skipped, kept: runs.length })
  return runs
}

function ensureLoaded(): void {
  if (!loaded) load()
}

/**
 * Remember one finished task.
 *
 * Appends to memory first, so the window's next poll sees the run even if the disk is
 * the thing that is broken.
 */
export function record(r: RunRecord): void {
  try {
    ensureLoaded()
    runs.push(r)
    if (runs.length > MAX_RECORDS) {
      runs = runs.slice(-TRIM_TO)
      rewrite()
      return
    }
    mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
    appendFileSync(file, JSON.stringify(r) + '\n', { mode: 0o600 })
  } catch (err) {
    complain('history.write_failed', err)
  }
}

function rewrite(): void {
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
  writeFileSync(file, runs.map(r => JSON.stringify(r)).join('\n') + '\n', { mode: 0o600 })
}

/** Newest first. */
export function recent(n: number): RunRecord[] {
  ensureLoaded()
  return runs.slice(-n).reverse()
}

export function summary(): HistorySummary {
  ensureLoaded()
  const byAdapter: HistorySummary['byAdapter'] = {}
  let ok = 0
  let busyMs = 0
  let cpuMs = 0
  for (const r of runs) {
    const good = r.outcome === 'ok'
    if (good) ok += 1
    busyMs += r.durationMs
    cpuMs += r.cpuMs
    const a = byAdapter[r.adapter] ?? (byAdapter[r.adapter] = { runs: 0, busyMs: 0, ok: 0 })
    a.runs += 1
    a.busyMs += r.durationMs
    if (good) a.ok += 1
  }
  return {
    runs: runs.length,
    ok,
    failed: runs.length - ok,
    busyMs,
    cpuMs,
    byAdapter,
    since: runs[0]?.startedAt ?? null,
  }
}

/**
 * `process.cpuUsage()` and `process.memoryUsage()` exist under Node and under the
 * Bun-compiled binary alike, but this is the one place where being wrong costs a task,
 * so neither is trusted to be there.
 */
export function cpuStart(): NodeJS.CpuUsage | null {
  try {
    return process.cpuUsage()
  } catch {
    return null
  }
}

export function cpuMsSince(start: NodeJS.CpuUsage | null): number {
  if (!start) return 0
  try {
    const d = process.cpuUsage(start)
    return Number(((d.user + d.system) / 1000).toFixed(1))
  } catch {
    return 0
  }
}

export function rssMb(): number {
  try {
    return Number((process.memoryUsage().rss / 1024 / 1024).toFixed(1))
  } catch {
    return 0
  }
}
