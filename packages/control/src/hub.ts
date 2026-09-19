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
import { log } from './logger.ts'
import { latestRelease } from './releases.ts'
import { noteConnection, noteDisconnection, noteAuthFailure } from './diagnostics.ts'

type Conn = {
  ws: WebSocket
  hostId: string
  label: string
  maxConcurrency: number
  allowCompute: boolean
  paused: boolean
  running: Set<string>
  lastSeen: number
  connectedAt: number
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
    try { await dispatchTo(conn) } catch (err) { log.error('dispatch.failed', { hostId: conn.hostId, err }) }
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
      const { capability: c, consent, afterSuspensionMs } = p.data
      conn.maxConcurrency = consent.maxConcurrency
      conn.allowCompute = consent.allowCompute
      conn.paused = consent.paused
      await pool.query(
        `update hosts set os=$2, arch=$3, cpu_model=$4, logical_cores=$5, total_ram_mb=$6, free_ram_mb=$7,
                agent_version=$8, allow_compute=$9, allow_browser=$10, paused=$11, max_concurrency=$12,
                adapters=$13, online=true, last_heartbeat_at=now()
          where id = $1`,
        [conn.hostId, c.os, c.arch, c.cpuModel, c.logicalCores, c.totalRamMb, c.freeRamMb, c.agentVersion,
         consent.allowCompute, consent.allowBrowser, consent.paused, consent.maxConcurrency, c.adapters],
      )
      await record({ hostId: conn.hostId, actor: 'agent', category: 'presence', type: 'host.hello',
        payload: { os: c.os, arch: c.arch, cores: c.logicalCores, adapters: c.adapters } })

      // Say why it came back, when it knows. "Asleep for 12 minutes" is an answer;
      // a gap between two timestamps is only a question.
      if (afterSuspensionMs && afterSuspensionMs > 0) {
        const seconds = Math.round(afterSuspensionMs / 1000)
        log.info('agent.woke_from_sleep', { hostId: conn.hostId, label: conn.label, suspendedForSeconds: seconds })
        await record({ hostId: conn.hostId, actor: 'agent', category: 'presence',
          type: 'host.woke_from_sleep', payload: { suspendedForSeconds: seconds } })
      }
      // Now it is genuinely usable: we know what it is and what it can run.
      await setPresence(conn.hostId, true)
      send(conn, 'hello.ack', {
        hostId: conn.hostId,
        serverTime: new Date().toISOString(),
        heartbeatSeconds: config.heartbeatSeconds,
        releaseVersion: latestRelease()?.manifest.version ?? null,
      }, msg.id)
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
    // Behind a tunnel or proxy the socket address is the proxy's, so prefer the
    // forwarded address when one is present.
    const fwd = req.headers['x-forwarded-for']
    const remoteAddr = ((Array.isArray(fwd) ? fwd[0] : fwd?.split(',')[0]) ?? req.socket.remoteAddress ?? 'unknown').trim()

    const deny = (reason: string) => {
      log.warn('agent.upgrade_denied', { reason, remoteAddr })
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

    const { rows } = await pool.query<{ id: string; label: string; public_key: string; revoked_at: string | null; max_concurrency: number; allow_compute: boolean; paused: boolean }>(
      `select id, label, public_key, revoked_at, max_concurrency, allow_compute, paused from hosts where id = $1`,
      [claimedHost],
    )
    const host = rows[0]
    const result = verifyAssertion(token, () => (host?.revoked_at ? undefined : host?.public_key), rememberJti)
    if (!result.ok) {
      await record({ hostId: host?.id ?? null, actor: 'control', category: 'security',
        type: 'agent.auth_rejected', payload: { reason: result.reason } })
      noteAuthFailure(claimedHost, result.reason, remoteAddr)
      log.warn('agent.auth_rejected', { hostId: claimedHost, reason: result.reason, remoteAddr })
      return deny(result.reason)
    }

    wss.handleUpgrade(req, socket, head, ws => {
      const existing = connections.get(result.hostId)
      if (existing) existing.ws.close(4000, 'superseded')

      const conn: Conn = {
        ws, hostId: result.hostId, label: host!.label, maxConcurrency: host!.max_concurrency,
        allowCompute: host!.allow_compute, paused: host!.paused, running: new Set(),
        lastSeen: Date.now(), connectedAt: Date.now(),
      }
      connections.set(conn.hostId, conn)
      noteConnection(conn.hostId, host!.label ?? conn.hostId, remoteAddr)
      // Presence is set once the host has introduced itself, not when the socket opens.
      // Between the two it is connected but has not yet said what it can run, and a job
      // submitted in that window was rejected as "no computer can run this" — a window
      // wide enough to hit on a slow link.
      log.info('agent.connected', {
        hostId: conn.hostId,
        // Carry the name through so logs read as names, not UUIDs. Debugging a real
        // connection means asking "is Sam's laptop on?", not reciting an identifier.
        label: host!.label,
        remoteAddr,
        supersededPrevious: Boolean(existing),
        userAgent: req.headers['user-agent'] ?? null,
      })

      ws.on('message', (data: Buffer) => {
        void onMessage(conn, data).catch((err: unknown) =>
          log.error('agent.message_failed', { hostId: conn.hostId, err }))
      })
      ws.on('pong', () => { conn.lastSeen = Date.now() })
      ws.on('close', (code: number, reason: Buffer) => {
        const heldTasks = [...conn.running]
        if (connections.get(conn.hostId) === conn) {
          connections.delete(conn.hostId)
          void setPresence(conn.hostId, false)
        }
        noteDisconnection(conn.hostId, code)
        log.info('agent.disconnected', {
          hostId: conn.hostId,
          label: conn.label,
          closeCode: code,
          closeReason: reason.toString('utf8').slice(0, 120) || null,
          connectedMs: Date.now() - conn.connectedAt,
          // Tasks still held at disconnect are exactly what the lease sweeper
          // will have to reclaim; naming them here makes that traceable.
          abandonedTasks: heldTasks.length,
          taskIds: heldTasks.slice(0, 10),
        })
      })
      ws.on('error', () => ws.close())
    })
  })

  // Presence: three missed pings and the host is gone. Its leases expire on their own clock.
  setInterval(() => {
    const cutoff = Date.now() - config.offlineAfterSeconds * 1000
    for (const conn of connections.values()) {
      if (conn.lastSeen < cutoff) {
        // A silent socket is the normal symptom of a laptop that slept, lost wifi, or
        // died. It looks identical to a healthy one until we time it out.
        log.warn('agent.heartbeat_timeout', {
          hostId: conn.hostId,
          label: conn.label,
          silentMs: Date.now() - conn.lastSeen,
          heldTasks: conn.running.size,
        })

        // terminate(), not close(). A close frame asks the peer to reply, and a peer
        // that has gone silent never will — the connection would sit in CLOSING
        // forever, this timer would fire on it again every tick, and the host would
        // never be marked offline. Drop it from the registry here rather than waiting
        // for a 'close' event that may not arrive.
        connections.delete(conn.hostId)
        void setPresence(conn.hostId, false)
        noteDisconnection(conn.hostId, 4008)
        conn.ws.terminate()
      } else if (conn.ws.readyState === conn.ws.OPEN) {
        conn.ws.ping()
      }
    }
  }, config.heartbeatSeconds * 1000).unref()
}

/**
 * Close every agent connection, for shutdown.
 *
 * Without this the process cannot exit: the HTTP server's graceful close waits for open
 * connections, and an agent's WebSocket is an open connection that will not close until
 * the server tells it to. The two wait for each other forever, the port stays occupied,
 * and the next start fails with EADDRINUSE.
 *
 * terminate() rather than close(): we are on our way out and have no time to wait for a
 * closing handshake. Agents treat an abrupt close as a normal disconnect and reconnect.
 */
export function closeAllConnections(reason: string): number {
  const n = connections.size
  for (const conn of connections.values()) {
    try { conn.ws.terminate() } catch {}
  }
  connections.clear()
  if (n > 0) log.info('shutdown.connections_closed', { count: n, reason })
  return n
}

export async function disconnectHost(hostId: string, reason: string): Promise<void> {
  const conn = connections.get(hostId)
  if (!conn) return
  send(conn, 'revoked', { reason })
  conn.ws.close(4003, reason)
}
