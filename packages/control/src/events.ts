import { pool } from './db.ts'

type Event = {
  jobId?: string | null
  taskId?: string | null
  hostId?: string | null
  actor: 'control' | 'agent' | 'user'
  category: 'auth' | 'lifecycle' | 'dispatch' | 'result' | 'presence' | 'security'
  type: string
  payload?: Record<string, unknown>
}

/**
 * The append-only run log. Page content and secrets never land here — the
 * `security` category records that something was rejected, not what it contained.
 */
export async function record(e: Event): Promise<void> {
  await pool.query(
    `insert into run_events(job_id, task_id, host_id, actor, category, type, payload)
     values ($1,$2,$3,$4,$5,$6,$7)`,
    [e.jobId ?? null, e.taskId ?? null, e.hostId ?? null, e.actor, e.category, e.type, e.payload ?? {}],
  )
}
