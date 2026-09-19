import Fastify, { type FastifyInstance, type FastifyRequest } from 'fastify'
import { z } from 'zod'
import { pool } from './db.ts'
import { config } from './config.ts'
import { login, userForToken, type User } from './auth.ts'
import { issuePairCode, redeemPairCode, revokeHost } from './pairing.ts'
import { record } from './events.ts'
import { dispatchAll, disconnectHost } from './hub.ts'
import { settleJob } from './scheduler.ts'
import { RateLimiter } from './ratelimit.ts'
import { joinPage } from './joinpage.ts'
import { snapshot } from './diagnostics.ts'
import { readArtifact, manifest } from './artifacts.ts'
import { latestRelease, releaseBundle } from './releases.ts'
import { binaryIndex, binaryFile, installShell, installPowerShell } from './installer.ts'
import { dashboardHtml } from './dashboard.ts'
import { loginHtml } from './loginpage.ts'
import { walkerHtml } from './walkerpage.ts'
import { verifyAssertion } from '@dwp/protocol'
import { log } from './logger.ts'

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

// Both of these are reachable without a session, so both are reachable by anyone on
// the internet once the tunnel is up. Login is guessing a password; pair is guessing
// an 8-character code.
const loginLimiter = new RateLimiter(10, 5 * 60_000)
const pairLimiter = new RateLimiter(20, 5 * 60_000)

export function buildServer(): FastifyInstance {
  const app = Fastify({ logger: false, trustProxy: true })

  async function requireUser(req: FastifyRequest): Promise<User> {
    const user = await userForToken(cookie(req, SESSION_COOKIE))
    if (!user) throw Object.assign(new Error('unauthorized'), { statusCode: 401 })
    return user
  }

  /**
   * Pasting the bare address into a browser is the first thing anyone does, and it
   * previously returned a bare 404 that looks like the server is broken. Send it
   * somewhere useful instead.
   */
  app.get('/', async (_req, reply) => reply.redirect('/join', 302))

  app.get('/health', async () => ({ ok: true, publicOrigin: config.publicOrigin }))

  /** What model and inputs this server is currently offering, for the dashboard. */
  /**
   * Experiment state: a scratchpad for work that spans many jobs.
   *
   * The driver script owns the algorithm; the platform just stores where it has got to,
   * so the dashboard can render progress without knowing anything about evolution.
   */
  app.get('/experiments/:name', async (req, reply) => {
    const { name } = z.object({ name: z.string().max(64) }).parse(req.params)
    const { rows } = await pool.query<{ state: unknown; updated_at: string }>(
      `select state, updated_at from experiments where name = $1`, [name])
    if (!rows[0]) return reply.code(404).send({ error: 'not-found' })
    return { name, ...rows[0] }
  })

  app.post('/experiments/:name', async req => {
    const user = await requireUser(req)
    const { name } = z.object({ name: z.string().max(64) }).parse(req.params)
    const state = z.record(z.string(), z.unknown()).parse(req.body)
    await pool.query(
      `insert into experiments(name, owner_id, state) values ($1,$2,$3)
       on conflict (name) do update set state = $3, updated_at = now()`,
      [name, user.id, JSON.stringify(state)])
    return { ok: true }
  })

  /**
   * The current agent release. Public because it is signed and contains no secret — an
   * agent needs to see it before it has any reason to authenticate.
   */
  /**
   * The one-line installers.
   *
   * Public by necessity — they are what someone runs before they have anything. They
   * carry no secret: the pairing code is supplied by the person running them, and the
   * binary hashes they check against arrive over the same TLS connection as the script.
   */
  app.get('/install', async (_req, reply) =>
    reply.type('text/x-shellscript; charset=utf-8').send(installShell(config.publicOrigin)))

  app.get('/install.ps1', async (_req, reply) =>
    reply.type('text/plain; charset=utf-8').send(installPowerShell(config.publicOrigin)))

  /** What binaries exist, with hashes, signed as a set. */
  app.get('/release/binaries', async (_req, reply) => {
    const index = binaryIndex()
    return index ? index : reply.code(404).send({ error: 'no-binaries' })
  })

  /**
   * The executables themselves.
   *
   * Unauthenticated on purpose: a machine has no identity until it has the agent, so
   * requiring one would be circular. The bytes are public, verifiable against a signed
   * hash, and contain nothing specific to this network.
   */
  app.get('/download/:file', async (req, reply) => {
    const { file } = z.object({ file: z.string().max(80) }).parse(req.params)
    const bytes = binaryFile(file)
    if (!bytes) return reply.code(404).send({ error: 'not-found' })
    return reply
      .type('application/octet-stream')
      .header('content-length', String(bytes.length))
      .header('content-disposition', `attachment; filename="${file}"`)
      .header('cache-control', 'public, max-age=31536000, immutable')
      .send(bytes)
  })

  app.get('/release/latest', async (_req, reply) => {
    const release = latestRelease()
    return release ? release : reply.code(404).send({ error: 'no-release' })
  })

  /** The bundle itself, to enrolled hosts only. */
  app.get('/release/:sha256', async (req, reply) => {
    const auth = req.headers.authorization
    const token = auth?.startsWith('Bearer ') ? auth.slice(7) : undefined
    if (!token) return reply.code(401).send({ error: 'missing-assertion' })

    const claimed = (() => {
      try { return JSON.parse(Buffer.from(token.split('.')[1] ?? '', 'base64url').toString('utf8')).iss as string }
      catch { return undefined }
    })()
    if (!claimed) return reply.code(401).send({ error: 'malformed' })

    const { rows } = await pool.query<{ public_key: string; revoked_at: string | null }>(
      `select public_key, revoked_at from hosts where id = $1`, [claimed])
    const host = rows[0]
    const result = verifyAssertion(token, () => (host?.revoked_at ? undefined : host?.public_key), () => false)
    if (!result.ok) return reply.code(401).send({ error: result.reason })

    const { sha256 } = z.object({ sha256: z.string().length(64) }).parse(req.params)
    const bytes = releaseBundle(sha256)
    if (!bytes) return reply.code(404).send({ error: 'not-found' })

    await record({ hostId: result.hostId, actor: 'agent', category: 'lifecycle',
      type: 'release.downloaded', payload: { sha256: sha256.slice(0, 12) } })

    return reply.type('application/gzip')
      .header('content-length', String(bytes.length))
      .header('cache-control', 'public, max-age=31536000, immutable')
      .send(bytes)
  })

  app.get('/manifest', async (_req, reply) => {
    const m = manifest()
    return m ? m : reply.code(404).send({ error: 'no-fixtures' })
  })

  /**
   * Serve a pinned artifact to an enrolled host.
   *
   * Authenticated with the same short-lived assertion the WebSocket uses, so a public
   * address does not mean a public file server. Hosts fetch each artifact once and cache
   * it by hash.
   */
  app.get('/artifacts/:sha256', async (req, reply) => {
    const auth = req.headers.authorization
    const token = auth?.startsWith('Bearer ') ? auth.slice(7) : undefined
    if (!token) return reply.code(401).send({ error: 'missing-assertion' })

    const claimed = (() => {
      try { return JSON.parse(Buffer.from(token.split('.')[1] ?? '', 'base64url').toString('utf8')).iss as string }
      catch { return undefined }
    })()
    if (!claimed) return reply.code(401).send({ error: 'malformed' })

    const { rows } = await pool.query<{ public_key: string; revoked_at: string | null }>(
      `select public_key, revoked_at from hosts where id = $1`, [claimed])
    const host = rows[0]
    // Artifact fetches are frequent, so they do not share the WebSocket's replay cache —
    // a host legitimately mints several assertions in quick succession here.
    const result = verifyAssertion(token, () => (host?.revoked_at ? undefined : host?.public_key), () => false)
    if (!result.ok) {
      await record({ actor: 'control', category: 'security', type: 'artifact.auth_rejected',
        payload: { reason: result.reason } })
      return reply.code(401).send({ error: result.reason })
    }

    const { sha256 } = z.object({ sha256: z.string().length(64) }).parse(req.params)
    const artifact = readArtifact(sha256)
    if (!artifact) return reply.code(404).send({ error: 'not-found' })

    return reply
      .type('application/octet-stream')
      .header('content-length', String(artifact.bytes.length))
      // Content-addressed, so it can never go stale.
      .header('cache-control', 'public, max-age=31536000, immutable')
      .send(artifact.bytes)
  })

  /**
   * The page a friend opens to join. Public by design: it carries no secret of its own,
   * only the server address and instructions. The pairing code arrives in the link the
   * owner sends, and is single-use and short-lived.
   */
  app.get('/join', async (req, reply) => {
    const { code } = z.object({ code: z.string().max(32).optional() }).parse(req.query ?? {})
    return reply.type('text/html; charset=utf-8').send(joinPage(config.publicOrigin, code))
  })

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
    const retryAfter = loginLimiter.check(clientIp(req))
    if (retryAfter !== null) {
      await record({ actor: 'control', category: 'security', type: 'login.rate_limited' })
      return reply.code(429).header('retry-after', retryAfter).send({ error: 'too-many-attempts', retryAfter })
    }
    const body = z.object({ email: z.string(), password: z.string() }).parse(req.body)
    const token = await login(body.email, body.password)
    if (!token) return reply.code(401).send({ error: 'invalid-credentials' })
    reply.header('set-cookie',
      `${SESSION_COOKIE}=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200${config.publicOrigin.startsWith('https') ? '; Secure' : ''}`)
    return { ok: true }
  })

  app.get('/me', async req => ({ user: await requireUser(req) }))

  /**
   * The same physics the agents ran, served to the browser.
   *
   * One source of truth: a replay drawn by a second implementation would diverge from
   * the run that was actually scored.
   */
  app.get('/walker.js', async (_req, reply) => {
    const path = new URL('../../protocol/src/walker.js', import.meta.url)
    const { readFile } = await import('node:fs/promises')
    return reply.type('application/javascript; charset=utf-8')
      .header('cache-control', 'no-cache')
      .send(await readFile(path, 'utf8'))
  })

  app.get('/walker', async (req, reply) => {
    const user = await userForToken(cookie(req, SESSION_COOKIE))
    if (!user) return reply.redirect('/login', 302)
    return reply.type('text/html; charset=utf-8').send(walkerHtml())
  })

  app.get('/login', async (_req, reply) =>
    reply.type('text/html; charset=utf-8').send(loginHtml()))

  /** Send a signed-out browser to the sign-in form rather than a bare 401. */
  app.get('/dashboard', async (req, reply) => {
    const user = await userForToken(cookie(req, SESSION_COOKIE))
    if (!user) return reply.redirect('/login', 302)
    return reply.type('text/html; charset=utf-8').send(dashboardHtml())
  })

  /** Everything the dashboard needs, in one round trip. */
  app.get('/dashboard/data', async req => {
    const user = await requireUser(req)

    const { rows: hosts } = await pool.query(
      `select id, label, online, paused, revoked_at, os, arch, logical_cores, total_ram_mb,
              last_heartbeat_at, agent_version
         from hosts where owner_id = $1 order by online desc, label`, [user.id])

    const { rows: jobs } = await pool.query<{
      id: string; adapter: string; status: string; total_items: number
      created_at: string; done: string; in_flight: string
    }>(
      `select j.id, j.adapter, j.status, j.total_items, j.created_at,
              (select count(*) from tasks t where t.job_id = j.id and t.state = 'succeeded') as done,
              (select count(*) from tasks t where t.job_id = j.id
                 and t.state in ('offered','leased','running')) as in_flight
         from jobs j where j.owner_id = $1 order by j.created_at desc limit 8`, [user.id])

    // Which computer did how much of each job — the point of the whole system.
    const { rows: splits } = await pool.query<{ job_id: string; label: string; n: string }>(
      `select t.job_id, h.label, count(*)::text as n
         from tasks t join hosts h on h.id = t.assigned_host_id
        where t.state = 'succeeded' and t.job_id = any($1::uuid[])
        group by t.job_id, h.label order by count(*) desc`,
      [jobs.map(j => j.id)])

    const { rows: events } = await pool.query(
      `select e.type, e.server_ts, h.label,
              coalesce(e.payload->>'reason', e.payload->>'errorClass', e.payload->>'closeCode',
                       case when e.payload ? 'suspendedForSeconds'
                            then 'asleep for ' || (e.payload->>'suspendedForSeconds') || 's' end,
                       '') as detail
         from run_events e left join hosts h on h.id = e.host_id
        where e.category in ('presence','security','lifecycle','result')
        order by e.seq desc limit 25`)

    // Accuracy comes straight out of the stored inference outputs.
    const { rows: acc } = await pool.query<{ items: string; correct: string }>(
      `select coalesce(sum((output->>'count')::int),0)::text as items,
              coalesce(sum((output->>'correct')::int),0)::text as correct
         from tasks t join jobs j on j.id = t.job_id
        where j.owner_id = $1 and j.adapter = 'cpu_inference_batch' and t.state = 'succeeded'`, [user.id])

    const { rows: totals } = await pool.query<{ done: string }>(
      `select count(*)::text as done from tasks t join jobs j on j.id = t.job_id
        where j.owner_id = $1 and t.state = 'succeeded'`, [user.id])

    const items = Number(acc[0]?.items ?? 0)
    const correct = Number(acc[0]?.correct ?? 0)

    return {
      publicOrigin: config.publicOrigin,
      summary: {
        online: hosts.filter(h => h.online).length,
        enrolled: hosts.filter(h => !h.revoked_at).length,
        tasksDone: Number(totals[0]?.done ?? 0),
        jobs: jobs.length,
        inferred: items,
        accuracy: items > 0 ? Number(((100 * correct) / items).toFixed(1)) : null,
      },
      hosts,
      jobs: jobs.map(j => ({
        ...j,
        done: Number(j.done),
        inFlight: Number(j.in_flight),
        split: splits.filter(s => s.job_id === j.id).map(s => ({ label: s.label, n: Number(s.n) })),
      })),
      events,
    }
  })

  /**
   * Live connection health: who is connected, who keeps dropping, and why auth failed.
   *
   * This is the endpoint to open while a friend is trying to join — it answers
   * "did their request even reach me?", which the database alone cannot.
   */
  app.get('/diagnostics', async req => {
    await requireUser(req)
    const { rows: hosts } = await pool.query(
      `select id, label, online, paused, revoked_at, last_heartbeat_at, agent_version, os, arch
         from hosts order by created_at`)
    const { rows: recentSecurity } = await pool.query(
      `select type, count(*)::int as n, max(server_ts) as last_at
         from run_events where category = 'security' and server_ts > now() - interval '24 hours'
        group by type order by n desc`)
    return { ...snapshot(), hosts, recentSecurity, publicOrigin: config.publicOrigin }
  })

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
    const retryAfter = pairLimiter.check(clientIp(req))
    if (retryAfter !== null) {
      await record({ actor: 'control', category: 'security', type: 'pair.rate_limited' })
      return reply.code(429).header('retry-after', retryAfter).send({ error: 'too-many-attempts', retryAfter })
    }
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
    // Pin the release key at pairing. Trusting a key that arrives with a later release
    // would make the signature meaningless — this is the one moment trust is established.
    const release = latestRelease()
    return {
      hostId: result.hostId,
      label: result.label,
      wsUrl: `${config.publicOrigin.replace(/^http/, 'ws')}/agent/connect`,
      releaseKey: release?.publicKey ?? null,
      releaseVersion: release?.manifest.version ?? null,
    }
  })

  app.get('/hosts', async req => {
    const user = await requireUser(req)
    const { rows } = await pool.query(
      `select id, label, trust_tier, allow_compute, allow_browser, paused, os, arch, cpu_model,
              logical_cores, total_ram_mb, free_ram_mb, agent_version, max_concurrency,
              adapters, online, last_heartbeat_at, created_at, revoked_at
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
      adapter: z.enum(['echo', 'cpu_inference_batch', 'walker_evolution']),
      // 'each' pins one task per online host — the shape that proves each host ran its own work.
      mode: z.enum(['each', 'queue']).default('each'),
      count: z.number().int().positive().max(10_000).default(1),
      sleepMs: z.number().int().nonnegative().max(60_000).default(0),
      hostId: z.uuid().optional(),
      /** cpu_inference_batch: how many digits each task covers. */
      batchSize: z.number().int().positive().max(2_000).default(100),
      /** Run the whole set this many times, for a job long enough to watch. */
      repeat: z.number().int().positive().max(200).default(1),
      /** walker_evolution: the slices to evaluate, built by the driver script. */
      tasks: z.array(z.record(z.string(), z.unknown())).max(500).optional(),
    }).parse(req.body)

    const { rows: hostRows } = await pool.query<{ id: string }>(
      `select id from hosts
        where owner_id = $1 and revoked_at is null and online and not paused and allow_compute
          and ($2::uuid is null or id = $2)
          and $3 = any(adapters)
        order by created_at`, [user.id, body.hostId ?? null, body.adapter])
    if (hostRows.length === 0) {
      // Say which of the two reasons it is: "nobody is online" and "nobody can run this"
      // need completely different responses from whoever submitted it.
      const { rows: anyOnline } = await pool.query<{ n: string }>(
        `select count(*)::text as n from hosts
          where owner_id = $1 and revoked_at is null and online and not paused`, [user.id])
      return reply.code(409).send({
        error: 'no-eligible-hosts',
        reason: Number(anyOnline[0]?.n ?? 0) === 0
          ? 'no computers are connected'
          : `no connected computer can run "${body.adapter}" — they may need: pnpm agent enable`,
      })
    }

    type Item = { pin: string | null; input: Record<string, unknown> }
    let items: Item[]
    let manifestHash: string | null = null

    if (body.adapter === 'walker_evolution') {
      if (!body.tasks?.length) return reply.code(400).send({ error: 'no-tasks' })
      items = body.tasks.map(input => ({ pin: body.hostId ?? null, input }))
    } else if (body.adapter === 'cpu_inference_batch') {
      const m = manifest() as {
        model?: { hash?: string; inputName?: string; outputName?: string }
        inputs?: { hash?: string; count?: number }
      } | null
      if (!m?.model?.hash || !m.inputs?.hash) {
        return reply.code(409).send({ error: 'no-fixtures', hint: 'run: node scripts/build-fixtures.ts' })
      }

      const available = m.inputs.count ?? 0
      const wanted = Math.min(body.count, available)
      if (wanted === 0) return reply.code(400).send({ error: 'no-inputs' })

      // Split the batch into independent slices. Never split one inference across
      // machines — parallelism is across items, which is the only safe kind.
      // Honour an explicit host for every adapter. It was previously accepted and
      // silently ignored here, so targeting one machine quietly ran the work on another.
      const pin = body.hostId ?? null
      items = []
      for (let pass = 0; pass < body.repeat; pass++) {
        for (let from = 0; from < wanted; from += body.batchSize) {
          items.push({
            pin,
            input: {
              modelHash: m.model.hash,
              inputsHash: m.inputs.hash,
              inputName: m.model.inputName ?? 'Input3',
              outputName: m.model.outputName ?? 'Plus214_Output_0',
              from,
              count: Math.min(body.batchSize, wanted - from),
              preprocessing: 'v1',
            },
          })
        }
      }
      manifestHash = m.inputs.hash
    } else {
      items = body.mode === 'each'
        ? hostRows.flatMap(h => Array.from({ length: body.count }, () => ({
            pin: h.id, input: { nonce: crypto.randomUUID(), sleepMs: body.sleepMs },
          })))
        : Array.from({ length: body.count }, () => ({
            pin: null, input: { nonce: crypto.randomUUID(), sleepMs: body.sleepMs },
          }))
    }

    const job = await pool.query<{ id: string }>(
      `insert into jobs(owner_id, adapter, total_items, started_at, constraints, manifest_hash)
       values ($1,$2,$3,now(),$4,$5) returning id`,
      [user.id, body.adapter, items.length,
       JSON.stringify({ mode: body.mode, sleepMs: body.sleepMs, batchSize: body.batchSize }),
       manifestHash])
    const jobId = job.rows[0]!.id

    for (const [seq, item] of items.entries()) {
      await pool.query(`insert into tasks(job_id, seq, input, pin_host_id) values ($1,$2,$3,$4)`,
        [jobId, seq, JSON.stringify(item.input), item.pin])
    }
    await record({ jobId, actor: 'user', category: 'lifecycle', type: 'job.created',
      payload: { adapter: body.adapter, items: items.length, mode: body.mode } })

    void dispatchAll()
    return reply.code(201).send({ jobId, totalItems: items.length, hosts: hostRows.length })
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

  app.setNotFoundHandler(async (req, reply) => {
    const wantsHtml = (req.headers.accept ?? '').includes('text/html')
    if (!wantsHtml) return reply.code(404).send({ error: 'not-found' })
    return reply.code(404).type('text/html; charset=utf-8').send(
      `<!doctype html><meta charset="utf-8">` +
      `<meta name="viewport" content="width=device-width,initial-scale=1">` +
      `<title>Not found</title>` +
      `<body style="margin:0;font:16px/1.6 ui-sans-serif,system-ui,sans-serif;` +
      `display:grid;place-items:center;min-height:100vh;padding:24px;text-align:center">` +
      `<div><h1 style="font-size:20px;margin:0 0 8px">Nothing here</h1>` +
      `<p style="color:#667;margin:0 0 20px">There is no page at <code>${req.url.replace(/[<>&"]/g, '')}</code>.</p>` +
      `<a href="/join" style="color:#0b6e8c">Go to the join page</a></div></body>`)
  })

  app.setErrorHandler((err: unknown, req, reply) => {
    // A rejected request body is the caller's mistake, not ours. Returning 500 for it
    // sends a friend chasing a server fault that does not exist, and hides real 500s
    // among the noise.
    const isValidation = err instanceof z.ZodError ||
      (err as { code?: string }).code === 'FST_ERR_VALIDATION'
    const status = isValidation ? 400 : ((err as { statusCode?: number }).statusCode ?? 500)
    const message = isValidation
      ? 'invalid-request'
      : (err instanceof Error ? err.message : 'internal-error')
    if (status >= 500) log.error('http.error', { method: req.method, url: req.url, err })
    else log.debug('http.rejected', { method: req.method, url: req.url, status, message })
    // Never hand an internal message to an unauthenticated caller on the open internet.
    reply.code(status).send({ error: status >= 500 ? 'internal-error' : message })
  })

  return app
}
