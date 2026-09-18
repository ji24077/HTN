import Fastify, { type FastifyInstance, type FastifyRequest } from 'fastify'
import { z } from 'zod'
import { pool } from './db.ts'
import { config } from './config.ts'
import { login, userForToken, type User } from './auth.ts'
import { issuePairCode, redeemPairCode, revokeHost } from './pairing.ts'
import { record } from './events.ts'
import { dispatchAll, disconnectHost } from './hub.ts'
import { settleJob } from './scheduler.ts'

const SESSION_COOKIE = 'dwp_session'

function cookie(req: FastifyRequest, name: string): string | undefined {
  return req.headers.cookie?.split(';')
    .map(c => c.trim().split('='))
    .find(([k]) => k === name)?.[1]
}

/** Observed source address, trusting one hop of proxy (the tunnel or load balancer). */
function clientIp(req: FastifyRequest): string {
  const fwd = req.headers['x-forwarded-for']
  const first = Array.isArray(fwd) ? fwd[0] : fwd?.split(',')[0]
  return (first ?? req.ip).trim()
}

export function buildServer(): FastifyInstance {
  const app = Fastify({ logger: false, trustProxy: true })

  async function requireUser(req: FastifyRequest): Promise<User> {
    const user = await userForToken(cookie(req, SESSION_COOKIE))
    if (!user) throw Object.assign(new Error('unauthorized'), { statusCode: 401 })
    return user
  }

  app.get('/health', async () => ({ ok: true, publicOrigin: config.publicOrigin }))

  /**
   * The team-operated demo endpoint for the browser-egress gate.
   *
   * Whatever address this reports is the address that actually fetched the page —
   * which is the whole point of running the browser on the remote host.
   */
  app.get('/whoami', async req => ({
    observedIp: clientIp(req),
    userAgent: req.headers['user-agent'] ?? null,
    serverTime: new Date().toISOString(),
    note: 'The address above is what this server observed. On a remote browser session it is the HOST egress, not yours.',
  }))

  // ------------------------------------------------------------------- auth

  app.post('/auth/login', async (req, reply) => {
    const body = z.object({ email: z.string(), password: z.string() }).parse(req.body)
    const token = await login(body.email, body.password)
    if (!token) return reply.code(401).send({ error: 'invalid-credentials' })
    reply.header('set-cookie',
      `${SESSION_COOKIE}=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200${config.publicOrigin.startsWith('https') ? '; Secure' : ''}`)
    return { ok: true }
  })

  app.get('/me', async req => ({ user: await requireUser(req) }))

  // ------------------------------------------------------------------ hosts

  app.post('/hosts/pair-code', async req => {
    const user = await requireUser(req)
    const body = z.object({ label: z.string().min(1).max(60).default('New computer') }).parse(req.body ?? {})
    const { code, expiresAt } = await issuePairCode(user.id, body.label)
    return { code, expiresAt, connectUrl: config.publicOrigin }
  })

  // Authenticated by the pairing code itself, not by a session: the agent runs on a
  // machine that has never seen the user's browser.
  app.post('/hosts/pair', async (req, reply) => {
    const body = z.object({
      code: z.string().min(4),
      publicKey: z.string().min(16),
      label: z.string().max(60).optional(),
    }).parse(req.body)
    const result = await redeemPairCode(body.code.trim().toUpperCase(), body.publicKey, body.label)
    if (!result.ok) {
      await record({ actor: 'control', category: 'security', type: 'pair.rejected', payload: { reason: result.reason } })
      return reply.code(400).send({ error: result.reason })
    }
    return { hostId: result.hostId, label: result.label, wsUrl: `${config.publicOrigin.replace(/^http/, 'ws')}/agent/connect` }
  })

  app.get('/hosts', async req => {
    const user = await requireUser(req)
    const { rows } = await pool.query(
      `select id, label, trust_tier, allow_compute, allow_browser, paused, os, arch, cpu_model,
              logical_cores, total_ram_mb, free_ram_mb, agent_version, max_concurrency,
              online, last_heartbeat_at, created_at, revoked_at
         from hosts where owner_id = $1 order by created_at`, [user.id])
    return { hosts: rows }
  })

  app.post('/hosts/:id/revoke', async (req, reply) => {
    const user = await requireUser(req)
    const { id } = z.object({ id: z.uuid() }).parse(req.params)
    if (!(await revokeHost(id, user.id))) return reply.code(404).send({ error: 'not-found' })
    await disconnectHost(id, 'revoked')
    return { ok: true }
  })

  // ------------------------------------------------------------------- jobs

  app.post('/jobs', async (req, reply) => {
    const user = await requireUser(req)
    const body = z.object({
      adapter: z.literal('echo'),
      // 'each' pins one task per online host — the shape that proves each host ran its own work.
      mode: z.enum(['each', 'queue']).default('each'),
      hostId: z.uuid().optional(),
      count: z.number().int().positive().max(10_000).default(1),
      sleepMs: z.number().int().nonnegative().max(60_000).default(0),
    }).parse(req.body)

    const { rows: hostRows } = await pool.query<{ id: string }>(
      `select id from hosts
        where owner_id = $1 and revoked_at is null and online and not paused and allow_compute
          and ($2::uuid is null or id = $2)
        order by created_at`, [user.id, body.hostId ?? null])
    if (hostRows.length === 0) return reply.code(409).send({ error: 'no-eligible-hosts' })

    const inputs = body.mode === 'each'
      ? hostRows.flatMap(h => Array.from({ length: body.count }, () => ({ pin: h.id })))
      : Array.from({ length: body.count }, () => ({ pin: null as string | null }))

    const job = await pool.query<{ id: string }>(
      `insert into jobs(owner_id, adapter, total_items, started_at, constraints)
       values ($1,$2,$3,now(),$4) returning id`,
      [user.id, body.adapter, inputs.length, JSON.stringify({ mode: body.mode, sleepMs: body.sleepMs })])
    const jobId = job.rows[0]!.id

    for (const [seq, item] of inputs.entries()) {
      await pool.query(`insert into tasks(job_id, seq, input, pin_host_id) values ($1,$2,$3,$4)`,
        [jobId, seq, JSON.stringify({ nonce: crypto.randomUUID(), sleepMs: body.sleepMs }), item.pin])
    }
    await record({ jobId, actor: 'user', category: 'lifecycle', type: 'job.created',
      payload: { adapter: body.adapter, items: inputs.length, mode: body.mode } })

    void dispatchAll()
    return reply.code(201).send({ jobId, totalItems: inputs.length, hosts: hostRows.length })
  })

  app.get('/jobs/:id', async req => {
    const user = await requireUser(req)
    const { id } = z.object({ id: z.uuid() }).parse(req.params)
    const { rows: jobRows } = await pool.query(
      `select * from jobs where id = $1 and owner_id = $2`, [id, user.id])
    if (!jobRows[0]) throw Object.assign(new Error('not-found'), { statusCode: 404 })
    const { rows: tasks } = await pool.query(
      `select t.id, t.seq, t.state, t.attempts, t.assigned_host_id, h.label as host_label,
              t.output, t.output_hash, t.error_class, t.queued_at, t.started_at, t.finished_at
         from tasks t left join hosts h on h.id = t.assigned_host_id
        where t.job_id = $1 order by t.seq`, [id])
    const { rows: rollup } = await pool.query(
      `select state, count(*)::int as n from tasks where job_id = $1 group by state`, [id])
    return { job: jobRows[0], rollup, tasks }
  })

  app.post('/jobs/:id/cancel', async req => {
    const user = await requireUser(req)
    const { id } = z.object({ id: z.uuid() }).parse(req.params)
    await pool.query(
      `update jobs set cancel_requested_at = now() where id = $1 and owner_id = $2 and status = 'running'`,
      [id, user.id])
    await pool.query(
      `update tasks set state = 'cancelled', finished_at = now() where job_id = $1 and state = 'pending'`, [id])
    await record({ jobId: id, actor: 'user', category: 'lifecycle', type: 'job.cancel_requested' })
    await settleJob(id)
    return { ok: true }
  })

  app.get('/jobs/:id/events', async req => {
    const user = await requireUser(req)
    const { id } = z.object({ id: z.uuid() }).parse(req.params)
    const { rows } = await pool.query(
      `select e.seq, e.actor, e.category, e.type, e.payload, e.server_ts, e.host_id, e.task_id
         from run_events e join jobs j on j.id = e.job_id
        where e.job_id = $1 and j.owner_id = $2 order by e.seq`, [id, user.id])
    return { events: rows }
  })

  app.setErrorHandler((err: unknown, _req, reply) => {
    const status = (err as { statusCode?: number }).statusCode ?? 500
    const message = err instanceof Error ? err.message : 'internal-error'
    if (status >= 500) console.error('[http]', err)
    reply.code(status).send({ error: message })
  })

  return app
}
