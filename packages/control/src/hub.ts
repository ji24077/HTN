import type { IncomingMessage } from 'node:http'
import { WebSocketServer, type WebSocket } from 'ws'
import {
  decode, envelope, verifyAssertion, verifyAttestation,
  Hello, Heartbeat, ConsentUpdate, TaskAccept, TaskDecline, LeaseRenew, TaskResult, TaskError,
} from '@dwp/protocol'
import { pool } from './db.ts'
import { config } from './config.ts'
import { record } from './events.ts'
import { claimFor, markLeased, renewLease, releaseOffer, settleJob } from './scheduler.ts'

type Conn = {
  ws: WebSocket
  hostId: string
  maxConcurrency: number
  allowCompute: boolean
  paused: boolean
  running: Set<string>
  lastSeen: number
}

const connections = new Map<string, Conn>()
const seenJti = new Map<string, number>()

export const onlineHostIds = () => [...connections.keys()]

function rememberJti(jti: string): boolean {
  const now = Date.now()
  for (const [k, t] of seenJti) if (now - t > 10 * 60_000) seenJti.delete(k)
  if (seenJti.has(jti)) return true
  seenJti.set(jti, now)
  return false
}

function send(conn: Conn, type: string, payload: unknown, replyTo?: string): void {
  if (conn.ws.readyState === conn.ws.OPEN) conn.ws.send(JSON.stringify(envelope(type, payload, replyTo)))
}

// ------------------------------------------------------------------- dispatch

// One dispatch pass per host at a time. Without this, two passes that overlap each
// read free capacity before either has claimed, and a host with a cap of 2 is handed
// a dozen tasks — observed, not theoretical.
const dispatching = new Set<string>()
const redispatch = new Set<string>()

/** Offer as much eligible work as this host has room for. */
async function dispatchTo(conn: Conn): Promise<void> {
  if (dispatching.has(conn.hostId)) {
    redispatch.add(conn.hostId)
    return
  }
  dispatching.add(conn.hostId)
  try {
    let again = true
    while (again) {
      again = false
      if (conn.paused || !conn.allowCompute || conn.ws.readyState !== conn.ws.OPEN) break

      const room = conn.maxConcurrency - conn.running.size
      if (room > 0) {
        const tasks = await claimFor(conn.hostId, room)
        // Reserve the slots before yielding again, so the next pass sees true capacity.
        for (const t of tasks) {
          conn.running.add(t.id)
          send(conn, 'task.offer', {
            taskId: t.id, jobId: t.job_id, adapter: t.adapter, attempt: t.attempts,
            input: t.input, leaseId: t.lease_id, leaseSeconds: config.leaseSeconds, wallClockMs: 60_000,
          })
        }
        for (const t of tasks) {
          await record({ jobId: t.job_id, taskId: t.id, hostId: conn.hostId, actor: 'control',
            category: 'dispatch', type: 'task.offered', payload: { attempt: t.attempts } })
        }
      }
      if (redispatch.delete(conn.hostId)) again = true
    }
  } finally {
    dispatching.delete(conn.hostId)
  }
}

export async function dispatchAll(): Promise<void> {
  for (const conn of connections.values()) {
    try { await dispatchTo(conn) } catch (err) { console.error('[hub] dispatch failed', err) }
  }
}

// ------------------------------------------------------------ result handling

async function acceptResult(conn: Conn, raw: unknown): Promise<void> {
  const parsed = TaskResult.safeParse(raw)
  if (!parsed.success) return
  const r = parsed.data

  const { rows } = await pool.query<{ public_key: string; job_id: string }>(
    `select h.public_key, t.job_id from tasks t join hosts h on h.id = $2 where t.id = $1`,
    [r.taskId, conn.hostId],
  )
  const ctx = rows[0]
  if (!ctx) return

  // Verify provenance BEFORE accepting. An unverifiable result is a security event,
  // not a result — this is what the `distribution` gate rests on.
  const signatureOk = verifyAttestation(ctx.public_key, r.signature, {
    taskId: r.taskId, attempt: r.attempt, hostId: conn.hostId,
    outputHash: r.outputHash, startedAt: r.startedAt, finishedAt: r.finishedAt,
  })
  if (!signatureOk) {
    conn.running.delete(r.taskId)
    await record({ jobId: ctx.job_id, taskId: r.taskId, hostId: conn.hostId, actor: 'control',
      category: 'security', type: 'result.signature_invalid', payload: { attempt: r.attempt } })
    await releaseOffer(r.taskId, r.leaseId)
    return
  }

  const accepted = await pool.query(
    `update tasks set state = 'succeeded', output = $3, output_hash = $4, finished_at = now()
      where id = $1 and lease_id = $2 and state <> 'succeeded'`,
    [r.taskId, r.leaseId, JSON.stringify(r.output ?? null), r.outputHash],
  )
  const isDuplicate = accepted.rowCount === 0
  conn.running.delete(r.taskId)

  await pool.query(
    `insert into task_attempts(task_id, attempt, host_id, outcome, host_signature, signature_ok,
                               host_reported_ms, started_at, finished_at)
     values ($1,$2,$3,$4,$5,true,$6,$7,$8) on conflict (task_id, attempt) do nothing`,
    [r.taskId, r.attempt, conn.hostId, isDuplicate ? 'duplicate' : 'succeeded',
     r.signature, r.hostReportedMs, r.startedAt, r.finishedAt],
  )
  await record({
    jobId: ctx.job_id, taskId: r.taskId, hostId: conn.hostId, actor: 'agent', category: 'result',
    type: isDuplicate ? 'duplicate_result' : 'task.succeeded',
    payload: { attempt: r.attempt, outputHash: r.outputHash, hostReportedMs: r.hostReportedMs },
  })
  if (!isDuplicate) await settleJob(ctx.job_id)
  await dispatchTo(conn)
}

async function handleError(conn: Conn, raw: unknown): Promise<void> {
  const parsed = TaskError.safeParse(raw)
  if (!parsed.success) return
  const e = parsed.data
  conn.running.delete(e.taskId)

  const { rows } = await pool.query<{ attempts: number; job_id: string }>(
    `select attempts, job_id from tasks where id = $1`, [e.taskId],
  )
  const t = rows[0]
  if (!t) return
  const exhausted = t.attempts >= config.maxAttempts

  await pool.query(
    `update tasks set state = $2, assigned_host_id = null, lease_id = null, lease_expires_at = null,
            error_class = $3, finished_at = case when $2 = 'failed' then now() else finished_at end
      where id = $1 and lease_id = $4 and state not in ('succeeded','cancelled')`,
    [e.taskId, exhausted ? 'failed' : 'pending', e.errorClass, e.leaseId],
  )
  await pool.query(
    `insert into task_attempts(task_id, attempt, host_id, outcome) values ($1,$2,$3,'failed')
     on conflict (task_id, attempt) do nothing`,
    [e.taskId, t.attempts, conn.hostId],
  )
  await record({ jobId: t.job_id, taskId: e.taskId, hostId: conn.hostId, actor: 'agent', category: 'result',
    type: 'task.failed', payload: { errorClass: e.errorClass, requeued: !exhausted } })
  if (exhausted) await settleJob(t.job_id)
  await dispatchTo(conn)
}

// ------------------------------------------------------------------ lifecycle

async function setPresence(hostId: string, online: boolean): Promise<void> {
  await pool.query(`update hosts set online = $2, last_heartbeat_at = now() where id = $1`, [hostId, online])
  await record({ hostId, actor: 'control', category: 'presence', type: online ? 'host.online' : 'host.offline' })
}

async function onMessage(conn: Conn, raw: Buffer): Promise<void> {
  const msg = decode(raw)
  if (!msg) return
  conn.lastSeen = Date.now()

  switch (msg.type) {
    case 'hello': {
      const p = Hello.safeParse(msg.payload)
      if (!p.success) return
      const { capability: c, consent } = p.data
      conn.maxConcurrency = consent.maxConcurrency
      conn.allowCompute = consent.allowCompute
      conn.paused = consent.paused
      await pool.query(
        `update hosts set os=$2, arch=$3, cpu_model=$4, logical_cores=$5, total_ram_mb=$6, free_ram_mb=$7,
                agent_version=$8, allow_compute=$9, allow_browser=$10, paused=$11, max_concurrency=$12,
                online=true, last_heartbeat_at=now()
          where id = $1`,
        [conn.hostId, c.os, c.arch, c.cpuModel, c.logicalCores, c.totalRamMb, c.freeRamMb, c.agentVersion,
         consent.allowCompute, consent.allowBrowser, consent.paused, consent.maxConcurrency],
      )
      await record({ hostId: conn.hostId, actor: 'agent', category: 'presence', type: 'host.hello',
        payload: { os: c.os, arch: c.arch, cores: c.logicalCores, adapters: c.adapters } })
      send(conn, 'hello.ack', { hostId: conn.hostId, serverTime: new Date().toISOString(),
        heartbeatSeconds: config.heartbeatSeconds }, msg.id)
      await dispatchTo(conn)
      return
    }
    case 'heartbeat': {
      const p = Heartbeat.safeParse(msg.payload)
      if (!p.success) return
      await pool.query(`update hosts set last_heartbeat_at = now(), online = true, free_ram_mb = $2 where id = $1`,
        [conn.hostId, p.data.freeRamMb])
      await dispatchTo(conn)
      return
    }
    case 'consent.update': {
      const p = ConsentUpdate.safeParse(msg.payload)
      if (!p.success) return
      conn.paused = p.data.paused
      conn.allowCompute = p.data.allowCompute
      conn.maxConcurrency = p.data.maxConcurrency
      await pool.query(
        `update hosts set paused=$2, allow_compute=$3, allow_browser=$4, max_concurrency=$5 where id = $1`,
        [conn.hostId, p.data.paused, p.data.allowCompute, p.data.allowBrowser, p.data.maxConcurrency],
      )
      await record({ hostId: conn.hostId, actor: 'agent', category: 'lifecycle', type: 'consent.updated',
        payload: { paused: p.data.paused } })
      if (!conn.paused) await dispatchTo(conn)
      return
    }
    case 'task.accept': {
      const p = TaskAccept.safeParse(msg.payload)
      if (p.success) await markLeased(p.data.taskId, p.data.leaseId)
      return
    }
    case 'task.decline': {
      const p = TaskDecline.safeParse(msg.payload)
      if (!p.success) return
      conn.running.delete(p.data.taskId)
      await releaseOffer(p.data.taskId, p.data.leaseId)
      return
    }
    case 'lease.renew': {
      const p = LeaseRenew.safeParse(msg.payload)
      if (p.success) await renewLease(p.data.taskId, p.data.leaseId)
      return
    }
    case 'task.result': return void (await acceptResult(conn, msg.payload))
    case 'task.error': return void (await handleError(conn, msg.payload))
    default: return
  }
}

export function attachAgentHub(server: import('node:http').Server): void {
  const wss = new WebSocketServer({ noServer: true })

  server.on('upgrade', async (req: IncomingMessage, socket, head) => {
    if (!req.url?.startsWith('/agent/connect')) return
    const deny = (reason: string) => {
      socket.write(`HTTP/1.1 401 Unauthorized\r\nx-dwp-reason: ${reason}\r\n\r\n`)
      socket.destroy()
    }

    const auth = req.headers.authorization
    const token = auth?.startsWith('Bearer ') ? auth.slice(7) : undefined
    if (!token) return deny('missing-assertion')

    const claimedHost = (() => {
      try { return JSON.parse(Buffer.from(token.split('.')[1] ?? '', 'base64url').toString('utf8')).iss as string }
      catch { return undefined }
    })()
    if (!claimedHost) return deny('malformed')

    const { rows } = await pool.query<{ id: string; public_key: string; revoked_at: string | null; max_concurrency: number; allow_compute: boolean; paused: boolean }>(
      `select id, public_key, revoked_at, max_concurrency, allow_compute, paused from hosts where id = $1`,
      [claimedHost],
    )
    const host = rows[0]
    const result = verifyAssertion(token, () => (host?.revoked_at ? undefined : host?.public_key), rememberJti)
    if (!result.ok) {
      await record({ hostId: host?.id ?? null, actor: 'control', category: 'security',
        type: 'agent.auth_rejected', payload: { reason: result.reason } })
      return deny(result.reason)
    }

    wss.handleUpgrade(req, socket, head, ws => {
      const existing = connections.get(result.hostId)
      if (existing) existing.ws.close(4000, 'superseded')

      const conn: Conn = {
        ws, hostId: result.hostId, maxConcurrency: host!.max_concurrency,
        allowCompute: host!.allow_compute, paused: host!.paused, running: new Set(), lastSeen: Date.now(),
      }
      connections.set(conn.hostId, conn)
      void setPresence(conn.hostId, true)
      console.log(`[hub] host ${conn.hostId} connected`)

      ws.on('message', (data: Buffer) => { void onMessage(conn, data).catch(err => console.error('[hub]', err)) })
      ws.on('pong', () => { conn.lastSeen = Date.now() })
      ws.on('close', () => {
        if (connections.get(conn.hostId) === conn) {
          connections.delete(conn.hostId)
          void setPresence(conn.hostId, false)
          console.log(`[hub] host ${conn.hostId} disconnected`)
        }
      })
      ws.on('error', () => ws.close())
    })
  })

  // Presence: three missed pings and the host is gone. Its leases expire on their own clock.
  setInterval(() => {
    const cutoff = Date.now() - config.offlineAfterSeconds * 1000
    for (const conn of connections.values()) {
      if (conn.lastSeen < cutoff) conn.ws.close(4008, 'heartbeat-timeout')
      else if (conn.ws.readyState === conn.ws.OPEN) conn.ws.ping()
    }
  }, config.heartbeatSeconds * 1000).unref()
}

export async function disconnectHost(hostId: string, reason: string): Promise<void> {
  const conn = connections.get(hostId)
  if (!conn) return
  send(conn, 'revoked', { reason })
  conn.ws.close(4003, reason)
}
