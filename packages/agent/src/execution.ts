import { createHash } from 'node:crypto'
import { mkdirSync, readdirSync, readFileSync, writeFileSync, renameSync, statSync, unlinkSync } from 'node:fs'
import { join } from 'node:path'
import { z } from 'zod'
import { AGENT_HOME } from './paths.ts'
import { scrubTelemetry } from './telemetry.ts'

export const ExecutionEvent = z.object({
  sequence: z.number().int().min(1).max(1000), at: z.string().datetime(),
  kind: z.enum(['started', 'step', 'stdout', 'stderr', 'progress', 'succeeded', 'failed',
    'cancelled', 'timed_out', 'interrupted', 'truncated']),
  data: z.record(z.string(), z.unknown()),
})
type Entry = z.infer<typeof ExecutionEvent>
type Record = { taskId: string; attempt: number; events: Entry[]; acked: number[]; truncated?: boolean }
export type ExecutionReporter = {
  step(message: string, data?: { [key: string]: unknown }): void
  stdout(text: string): void
  stderr(text: string): void
  progress(done: number, total: number): void
}
export const ExecutionAck = z.object({ taskId: z.string(), attempt: z.number().int(),
  sequences: z.array(z.number().int()), rejected: z.boolean().optional() })

/** Scope journals to both server and identity; re-pairing cannot leak old task output. */
export class ExecutionJournal {
  readonly directory: string
  private records = new Map<string, Record>()
  private inFlight = new Map<string, number>()
  private sizes = new Map<string, number>()
  private dirty = new Set<string>()
  private flushTimer: ReturnType<typeof setTimeout> | undefined
  constructor(server: string, workerId: string, root = join(AGENT_HOME, 'executions')) {
    this.directory = join(root, createHash('sha256').update(`${server}\n${workerId}`).digest('hex'))
    try {
      mkdirSync(this.directory, { recursive: true, mode: 0o700 })
      let bytes = 0
      const files = readdirSync(this.directory).filter(f => f.endsWith('.json'))
        .map(name => ({ name, stat: statSync(join(this.directory, name)) }))
        .sort((a, b) => b.stat.mtimeMs - a.stat.mtimeMs)
      for (const { name, stat } of files) {
        bytes += stat.size
        if (Date.now() - stat.mtimeMs > 7 * 86400_000 || bytes > 32 * 1024 * 1024 || stat.size > 2 * 1024 * 1024) {
          unlinkSync(join(this.directory, name)); continue
        }
        try {
          const text = readFileSync(join(this.directory, name), 'utf8')
          const record = JSON.parse(text) as Record
          if (!Array.isArray(record.events) || !Array.isArray(record.acked) ||
              !record.events.every(e => ExecutionEvent.safeParse(e).success)) continue
          this.records.set(this.key(record.taskId, record.attempt), record)
          this.sizes.set(this.key(record.taskId, record.attempt), Buffer.byteLength(text))
          const last = record.events.at(-1)
          if (last && !['succeeded', 'failed', 'cancelled', 'timed_out', 'interrupted'].includes(last.kind)) {
            this.emit(record.taskId, record.attempt, 'interrupted', { message: 'Runner restarted before recording completion' })
          }
        } catch { /* A damaged optional journal cannot prevent execution. */ }
      }
    } catch { /* Continue with a bounded in-memory journal if storage is unavailable. */ }
  }
  private key(taskId: string, attempt: number): string { return `${taskId}:${attempt}` }
  private persist(record: Record): void {
    this.dirty.delete(this.key(record.taskId, record.attempt))
    try {
      const name = createHash('sha256').update(this.key(record.taskId, record.attempt)).digest('hex')
      const path = join(this.directory, `${name}.json`)
      writeFileSync(`${path}.tmp`, JSON.stringify(record), { mode: 0o600 })
      renameSync(`${path}.tmp`, path)
    } catch { /* Diagnostics must not turn a successful task into a failed task. */ }
  }
  /** Rewriting the whole journal per event is quadratic in a chatty task; coalesce ordinary output. */
  private schedulePersist(key: string): void {
    this.dirty.add(key)
    if (this.flushTimer) return
    this.flushTimer = setTimeout(() => {
      this.flushTimer = undefined
      for (const pending of this.dirty) {
        const record = this.records.get(pending)
        if (record) this.persist(record)
      }
      this.dirty.clear()
    }, 250)
    this.flushTimer.unref?.()
  }
  emit(taskId: string, attempt: number, kind: Entry['kind'], data: { [key: string]: unknown } = {}): void {
    const key = this.key(taskId, attempt)
    let record = this.records.get(key)
    if (!record) {
      // Bound memory and disk during long-lived processes too, not only at startup.
      if (this.records.size >= 32) {
        const oldest = [...this.records.entries()].sort((a, b) =>
          (a[1].events[0]?.at ?? '').localeCompare(b[1].events[0]?.at ?? ''))[0]![0]
        this.records.delete(oldest)
        this.inFlight.delete(oldest)
        this.sizes.delete(oldest)
        this.dirty.delete(oldest)
        try { unlinkSync(join(this.directory, `${createHash('sha256').update(oldest).digest('hex')}.json`)) } catch {}
      }
      record = { taskId, attempt, events: [], acked: [] }
      this.records.set(key, record)
    }
    const terminal = ['succeeded', 'failed', 'cancelled', 'timed_out', 'interrupted'].includes(kind)
    if (record.truncated && !terminal) return
    if (record.events.length >= 999 || (record.events.length >= 997 && !terminal)) return
    let clean = scrubTelemetry(data)
    let size = Buffer.byteLength(JSON.stringify(clean))
    if (size > 8192) {
      clean = { message: 'Event exceeded 8 KiB', truncated: true }
      size = Buffer.byteLength(JSON.stringify(clean))
    }
    const bytes = this.sizes.get(key) ?? 0
    if ((record.events.length === 996 || bytes + size > 1000 * 1024) && !terminal) {
      record.truncated = true
      kind = 'truncated'; clean = { message: 'Execution output limit reached; completion is still recorded' }
    }
    const entry: Entry = { sequence: record.events.length + 1, at: new Date().toISOString(), kind, data: clean }
    record.events.push(entry)
    this.sizes.set(key, bytes + Buffer.byteLength(JSON.stringify(entry)))
    // Durability matters most at the boundaries: the first event proves the attempt
    // started and terminal events settle it. Everything in between is coalesced.
    if (terminal || record.truncated || record.events.length === 1) this.persist(record)
    else this.schedulePersist(key)
  }
  reporter(taskId: string, attempt: number): ExecutionReporter {
    let lastProgress = -1
    return {
      step: (message, data = {}) => this.emit(taskId, attempt, 'step', { ...data, message: message.slice(0, 2048) }),
      stdout: text => this.emit(taskId, attempt, 'stdout', { text: text.slice(0, 2048), truncated: text.length > 2048 }),
      stderr: text => this.emit(taskId, attempt, 'stderr', { text: text.slice(0, 2048), truncated: text.length > 2048 }),
      progress: (done, total) => {
        if (!Number.isFinite(done) || !Number.isFinite(total) || total <= 0) return
        const percent = Math.max(0, Math.min(100, Math.floor(done / total * 100)))
        if (percent === lastProgress) return
        lastProgress = percent
        this.emit(taskId, attempt, 'progress', { done, total, percent })
      },
    }
  }
  flush(send: (batch: { taskId: string; attempt: number; events: Entry[] }) => void): void {
    for (const [key, record] of this.records) {
      if (Date.now() - (this.inFlight.get(key) ?? 0) < 5000) continue
      const events = record.events.filter(e => !record.acked.includes(e.sequence)).slice(0, 8)
      if (!events.length) continue
      send({ taskId: record.taskId, attempt: record.attempt, events })
      this.inFlight.set(key, Date.now())
      break // Limit traffic per tick so output cannot starve leases or task results.
    }
  }
  acknowledge(value: unknown): void {
    const parsed = ExecutionAck.safeParse(value)
    if (!parsed.success) return
    const { taskId, attempt, sequences } = parsed.data
    const key = this.key(taskId, attempt), record = this.records.get(key)
    if (!record) return
    record.acked = [...new Set([...record.acked, ...sequences.filter(n => record.events.some(e => e.sequence === n))])]
    this.inFlight.delete(key)
    this.persist(record)
  }
}
