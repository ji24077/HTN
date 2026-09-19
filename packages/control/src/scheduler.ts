import { pool, tx } from './db.ts'
import { config } from './config.ts'
import { record } from './events.ts'

export type ClaimedTask = {
  id: string
  job_id: string
  adapter: string
  input: unknown
  lease_id: string
  attempts: number
}

/**
 * Atomically move up to `limit` eligible tasks to `offered` for one host.
 *
 * SKIP LOCKED is what makes concurrent dispatch safe: two dispatch passes can run
 * at once and will never hand the same task to two hosts.
 */
export async function claimFor(hostId: string, limit: number): Promise<ClaimedTask[]> {
  if (limit <= 0) return []
  const { rows } = await pool.query<ClaimedTask>(
    `with picked as (
       select t.id
         from tasks t
         join jobs j on j.id = t.job_id
        where t.state = 'pending'
          and j.status = 'running'
          and j.cancel_requested_at is null
          and (t.pin_host_id is null or t.pin_host_id = $1)
          -- Only work this host advertised it can run. Offering anything else produces
          -- a decline, a requeue, and an immediate re-offer to the same host.
          and j.adapter = any(select unnest(adapters) from hosts where id = $1)
        order by t.seq
        for update of t skip locked
        limit $2
     )
     update tasks t
        set state            = 'offered',
            assigned_host_id = $1,
            lease_id         = gen_random_uuid(),
            lease_expires_at = now() + ($3 || ' seconds')::interval,
            attempts         = t.attempts + 1
       from picked p
      where t.id = p.id
     returning t.id,
               t.job_id,
               (select adapter from jobs where id = t.job_id) as adapter,
               t.input,
               t.lease_id,
               t.attempts`,
    [hostId, limit, String(config.leaseSeconds)],
  )
  return rows
}

export async function markLeased(taskId: string, leaseId: string): Promise<boolean> {
  const { rowCount } = await pool.query(
    `update tasks set state = 'leased', started_at = coalesce(started_at, now()),
            lease_expires_at = now() + ($3 || ' seconds')::interval
      where id = $1 and lease_id = $2 and state = 'offered'`,
    [taskId, leaseId, String(config.leaseSeconds)],
  )
  return Boolean(rowCount)
}

export async function renewLease(taskId: string, leaseId: string): Promise<boolean> {
  const { rowCount } = await pool.query(
    `update tasks set lease_expires_at = now() + ($3 || ' seconds')::interval, state = 'running'
      where id = $1 and lease_id = $2 and state in ('leased','running')`,
    [taskId, leaseId, String(config.leaseSeconds)],
  )
  return Boolean(rowCount)
}

/** Return an offered task to the queue without burning another attempt. */
export async function releaseOffer(taskId: string, leaseId: string): Promise<void> {
  await pool.query(
    `update tasks set state = 'pending', assigned_host_id = null, lease_id = null,
            lease_expires_at = null, attempts = greatest(attempts - 1, 0)
      where id = $1 and lease_id = $2 and state = 'offered'`,
    [taskId, leaseId],
  )
}

/**
 * Reclaim leases whose holder went quiet.
 *
 * A task that has burned its attempts fails rather than looping forever; one that
 * has not goes back to `pending` for any eligible host.
 */
export async function sweepExpiredLeases(): Promise<number> {
  return tx(async client => {
    const { rows } = await client.query<{ id: string; attempts: number; assigned_host_id: string | null; job_id: string }>(
      `select id, attempts, assigned_host_id, job_id from tasks
        where state in ('offered','leased','running') and lease_expires_at < now()
        for update skip locked`,
    )
    for (const t of rows) {
      const exhausted = t.attempts >= config.maxAttempts
      await client.query(
        `update tasks set state = $2, assigned_host_id = null, lease_id = null, lease_expires_at = null,
                error_class = case when $2 = 'failed' then 'lease_expired' else error_class end,
                finished_at = case when $2 = 'failed' then now() else finished_at end
          where id = $1`,
        [t.id, exhausted ? 'failed' : 'pending'],
      )
      await client.query(
        `insert into task_attempts(task_id, attempt, host_id, outcome) values ($1,$2,$3,'lease_expired')
         on conflict (task_id, attempt) do nothing`,
        [t.id, t.attempts, t.assigned_host_id],
      )
      await client.query(
        `insert into run_events(job_id, task_id, host_id, actor, category, type, payload)
         values ($1,$2,$3,'control','dispatch','lease.expired',$4)`,
        [t.job_id, t.id, t.assigned_host_id, JSON.stringify({ attempt: t.attempts, requeued: !exhausted })],
      )
    }
    return rows.length
  })
}

/** Close out a job once no task can still change state. */
export async function settleJob(jobId: string): Promise<void> {
  const { rows } = await pool.query<{ open: string; failed: string; succeeded: string; cancelled: string }>(
    `select count(*) filter (where state in ('pending','offered','leased','running')) as open,
            count(*) filter (where state = 'failed')    as failed,
            count(*) filter (where state = 'succeeded') as succeeded,
            count(*) filter (where state = 'cancelled') as cancelled
       from tasks where job_id = $1`,
    [jobId],
  )
  const c = rows[0]
  if (!c || Number(c.open) > 0) return

  const cancelled = Number(c.cancelled)
  const failed = Number(c.failed)
  const status = cancelled > 0 ? (Number(c.succeeded) > 0 ? 'cancelled_partial' : 'cancelled')
    : failed > 0 ? 'failed' : 'succeeded'

  const { rowCount } = await pool.query(
    `update jobs set status = $2, ended_at = now() where id = $1 and status = 'running'`, [jobId, status],
  )
  if (rowCount) await record({ jobId, actor: 'control', category: 'lifecycle', type: `job.${status}` })
}
