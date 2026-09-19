import { freemem } from 'node:os'
import WebSocket from 'ws'
import {
  decode, envelope, mintAssertion, signAttestation, hashOutput,
  TaskOffer, HelloAck, Revoked,
} from '@dwp/protocol'
import type { KeyObject } from 'node:crypto'
import { createLogger } from '@dwp/protocol'
import { diagnoseOrigin } from '@dwp/protocol'
import { fallbackLookup, dnsFallbackEnabled, installDnsFallback } from './resolver.ts'
import { applyUpdate, restartIntoNewVersion } from './update.ts'
import { probe } from './capability.ts'
import { isPaused, type AgentConfig } from './config.ts'
import { runEcho } from './adapters/echo.ts'
import { runInference } from './adapters/inference.ts'
import { runWalker } from './adapters/walker.ts'

const log = createLogger({ component: 'agent' })

const ADAPTERS = ['echo', 'cpu_inference_batch', 'walker_evolution']
// Fallbacks only. The server announces the real cadence at handshake, and a lease
// duration with every offer; a hardcoded agent-side interval would silently drift out
// of agreement with the server the moment either is tuned.
const DEFAULT_HEARTBEAT_MS = 15_000
/**
 * A connection that survives this many heartbeat intervals counts as healthy and resets
 * the backoff. Relative rather than absolute so it scales with however the server is
 * tuned — the simulator runs a 2s heartbeat, production 15s.
 */
const STABLE_HEARTBEATS = 2
/** Long enough to stop hammering, short enough that recovery is not glacial. */
const MAX_BACKOFF_MS = 30_000
/** A dial that has not completed by now is not going to. */
const HANDSHAKE_TIMEOUT_MS = 15_000

type Running = { controller: AbortController; leaseId: string; renew: NodeJS.Timeout }

export function connect(cfg: AgentConfig, privateKey: KeyObject): void {
  installDnsFallback()
  let backoffMs = 1_000
  let stopped = false
  let attempt = 0
  let connectedSince = 0
  let everConnected = false
  let explained = false
  const hostLog = log.child({ hostId: cfg.hostId, label: cfg.label })
  const running = new Map<string, Running>()

  const open = (): void => {
    if (stopped) return
    attempt += 1
    const dialStartedAt = Date.now()
    hostLog.info('connect.attempt', { attempt, url: cfg.wsUrl })
    const ws = new WebSocket(cfg.wsUrl, {
      headers: { authorization: `Bearer ${mintAssertion(cfg.hostId, privateKey)}` },
      // Without this a dial into a black hole never returns: the upgrade request is
      // swallowed, no response ever comes, and the socket sits in CONNECTING forever —
      // a laptop that wakes from sleep and then simply does nothing, with no error to
      // show for it. Fail the attempt instead, so the normal retry path takes over.
      handshakeTimeout: HANDSHAKE_TIMEOUT_MS,
      // Same resolver fallback as pairing, so a machine that could enrol can also stay
      // connected rather than failing on the very next dial.
      ...(dnsFallbackEnabled() ? { lookup: fallbackLookup } : {}),
    })
    let heartbeat: NodeJS.Timeout | undefined
    let heartbeatMs = DEFAULT_HEARTBEAT_MS
    let pauseWas = isPaused()
    // Last time we heard ANYTHING from the server. A link can fail in one direction
    // only — our writes disappear into it while the socket still looks open — so the
    // absence of inbound traffic is the only reliable signal we have.
    let lastInbound = Date.now()

    const send = (type: string, payload: unknown, replyTo?: string): void => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(envelope(type, payload, replyTo)))
    }

    const consent = () => ({
      paused: isPaused(),
      allowCompute: cfg.allowCompute,
      allowBrowser: cfg.allowBrowser,
      maxConcurrency: cfg.maxConcurrency,
    })

    // The server keeps the link warm with ping frames. The `ws` client answers them
    // automatically and emits 'ping' — NOT 'pong', which is only for pings we send.
    // Listening for the wrong one made a healthy idle agent believe the server had gone
    // silent and reconnect every few seconds.
    ws.on('ping', () => { lastInbound = Date.now() })
    ws.on('pong', () => { lastInbound = Date.now() })

    ws.on('open', () => {
      connectedSince = Date.now()
      lastInbound = Date.now()
      everConnected = true
      explained = false
      hostLog.info('connect.established', { attempt, dialMs: Date.now() - dialStartedAt, url: cfg.wsUrl })
      send('hello', { capability: probe(ADAPTERS), consent: consent() })

      startHeartbeat()
    })

    const startHeartbeat = (): void => {
      if (heartbeat) clearInterval(heartbeat)
      heartbeat = setInterval(() => {
        // If the server has said nothing for three intervals, stop believing in this
        // socket and dial again. Without this an agent whose network died silently
        // waits forever, looking healthy to itself and absent to everyone else.
        const silentMs = Date.now() - lastInbound
        if (silentMs > heartbeatMs * 3) {
          hostLog.warn('server.went_silent', { silentMs, thresholdMs: heartbeatMs * 3 })
          ws.terminate()      // not close(): the peer is not answering
          teardown('server went silent')
          return
        }

        // The local pause flag is authoritative and is pushed the moment it changes.
        const paused = isPaused()
        if (paused !== pauseWas) {
          pauseWas = paused
          send('consent.update', consent())
          if (paused) {
            for (const [taskId, r] of running) { r.controller.abort(); clearInterval(r.renew); running.delete(taskId) }
          }
        }
        send('heartbeat', { freeRamMb: Math.round(freemem() / 1024 / 1024), running: running.size })
        // Our own ping, so a server that stops responding is detectable even when it
        // has nothing to say to us.
        if (ws.readyState === WebSocket.OPEN) { try { ws.ping() } catch {} }
      }, heartbeatMs)
    }

    ws.on('message', (raw: Buffer) => {
      lastInbound = Date.now()
      const msg = decode(raw)
      if (!msg) return

      if (msg.type === 'hello.ack') {
        const ack = HelloAck.safeParse(msg.payload)
        if (ack.success) {
          // Adopt the server's cadence rather than our own guess.
          if (ack.data.heartbeatSeconds > 0) {
            heartbeatMs = ack.data.heartbeatSeconds * 1000
            startHeartbeat()
          }
          /**
           * The server announces what it is offering; the agent decides whether to take
           * it, and verifies the signature itself. Updating while holding work would
           * abandon it mid-flight, so wait until the machine is idle.
           */
          const offered = ack.data.releaseVersion
          if (offered && cfg.autoUpdate !== false && offered !== cfg.installedRelease && running.size === 0) {
            hostLog.info('update.available', { have: cfg.installedRelease ?? null, offered })
            void applyUpdate(cfg, privateKey).then(result => {
              if (result.status === 'updated') {
                console.log(`\n  Updated to ${result.to}. Restarting.\n`)
                stopped = true
                try { ws.close(1000, 'updating') } catch {}
                setTimeout(restartIntoNewVersion, 500)
              } else if (result.status === 'refused') {
                hostLog.error('update.refused', { reason: result.reason })
                console.error(`\n  Refused an update: ${result.reason}\n`)
              } else if (result.status === 'unavailable') {
                hostLog.warn('update.unavailable', { reason: result.reason })
              }
            }).catch((err: unknown) => hostLog.warn('update.failed', { err }))
          }

          // Clock skew shows up here first, and misattributed timings later.
          const skewMs = Date.now() - Date.parse(ack.data.serverTime)
          hostLog.info('handshake.complete', { serverSkewMs: skewMs })
          if (Math.abs(skewMs) > 60_000) {
            hostLog.warn('clock.skewed', {
              serverSkewMs: skewMs,
              note: 'this machine disagrees with the server by over a minute',
            })
          }
        }
        return
      }

      if (msg.type === 'revoked') {
        const r = Revoked.safeParse(msg.payload)
        hostLog.error('host.revoked', { reason: r.success ? r.data.reason : 'unknown' })
        stopped = true
        ws.close()
        process.exitCode = 1
        return
      }

      if (msg.type === 'task.cancel') {
        const taskId = (msg.payload as { taskId?: string }).taskId
        const r = taskId ? running.get(taskId) : undefined
        if (r) { r.controller.abort(); clearInterval(r.renew); running.delete(taskId!) }
        return
      }

      if (msg.type === 'task.offer') {
        const parsed = TaskOffer.safeParse(msg.payload)
        if (!parsed.success) return
        const offer = parsed.data

        if (isPaused() || !cfg.allowCompute || !ADAPTERS.includes(offer.adapter)) {
          send('task.decline', { taskId: offer.taskId, leaseId: offer.leaseId, reason: 'not-eligible' })
          return
        }

        send('task.accept', { taskId: offer.taskId, leaseId: offer.leaseId })
        const controller = new AbortController()
        const renewMs = Math.max(1_000, Math.floor((offer.leaseSeconds * 1000) / 3))
        const renew = setInterval(() => send('lease.renew', { taskId: offer.taskId, leaseId: offer.leaseId }), renewMs)
        running.set(offer.taskId, { controller, leaseId: offer.leaseId, renew })

        const startedAt = new Date().toISOString()
        const t0 = performance.now()
        const wallClock = setTimeout(() => controller.abort(), offer.wallClockMs)
        void wallClock

        const work =
          offer.adapter === 'cpu_inference_batch'
            ? runInference(offer.input, {
                hostId: cfg.hostId, server: cfg.server, privateKey, signal: controller.signal,
              })
          : offer.adapter === 'walker_evolution'
            ? runWalker(offer.input, cfg.hostId, controller.signal)
          : runEcho(offer.input, cfg.hostId, controller.signal)

        void work
          .then(output => {
            const finishedAt = new Date().toISOString()
            const outputHash = hashOutput(output)
            send('task.result', {
              taskId: offer.taskId,
              leaseId: offer.leaseId,
              attempt: offer.attempt,
              output,
              outputHash,
              startedAt,
              finishedAt,
              hostReportedMs: Number((performance.now() - t0).toFixed(3)),
              // Signed here, on this machine, with a key the server has never seen.
              signature: signAttestation(privateKey, {
                taskId: offer.taskId, attempt: offer.attempt, hostId: cfg.hostId,
                outputHash, startedAt, finishedAt,
              }),
            })
          })
          .catch((err: unknown) => {
            send('task.error', {
              taskId: offer.taskId,
              leaseId: offer.leaseId,
              errorClass: controller.signal.aborted ? 'aborted' : 'adapter_error',
              message: err instanceof Error ? err.message : String(err),
            })
          })
          .finally(() => {
            clearTimeout(wallClock)
            clearInterval(renew)
            running.delete(offer.taskId)
          })
      }
    })

    let tornDown = false
    const teardown = (why: string, code?: number): void => {
      if (tornDown) return
      tornDown = true
      if (heartbeat) clearInterval(heartbeat)
      const abandoned = [...running.keys()]
      for (const [, r] of running) { r.controller.abort(); clearInterval(r.renew) }
      running.clear()

      const heldMs = connectedSince ? Date.now() - connectedSince : 0
      hostLog.warn('connect.lost', {
        why,
        closeCode: code ?? null,
        connectedMs: heldMs,
        abandonedTasks: abandoned.length,
      })
      connectedSince = 0
      if (stopped) return

      // Reset the backoff only for a connection that actually held. Resetting on every
      // 'open' turns a flapping link into a hot reconnect loop: connect, drop, retry in
      // 1s, forever, hammering the server. A link that cannot stay up for STABLE_MS is
      // not working, and we should back away from it like any other failure.
      if (heldMs >= heartbeatMs * STABLE_HEARTBEATS) {
        backoffMs = 1_000
        attempt = 0
      }

      // Full jitter: a fleet reconnecting after an outage must not arrive in lockstep.
      const delay = Math.random() * backoffMs
      backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS)
      hostLog.info('connect.retry_scheduled', { inMs: Math.round(delay), nextBackoffCapMs: backoffMs })

      // After a few failures in a row, stop emitting the same terse line and say what is
      // actually wrong. The overwhelmingly likely cause is that the host restarted and
      // their temporary address changed — which no amount of retrying will fix.
      // Re-explain periodically rather than once: whoever is going to read this probably
      // is not watching at the moment it first fails.
      if (attempt >= 3 && (!explained || attempt % 10 === 0)) {
        explained = true
        void diagnoseOrigin(cfg.server).then(d => {
          hostLog.error('connect.giving_advice', { cause: d.cause, attempts: attempt })
          console.error(`\n  Cannot reach ${cfg.server} after ${attempt} attempts.\n\n  ${d.message}\n`)
          // 'not-a-server' covers the common case where a stale tunnel hostname still
          // resolves but has nothing behind it any more.
          const looksStale = ['dns-nowhere', 'timeout', 'refused', 'not-a-server'].includes(d.cause)
          if (looksStale && everConnected) {
            console.error(
              `  This address used to work, so the most likely explanation is that the host\n` +
              `  restarted and their temporary address changed.\n\n` +
              `  Ask them for a new invite link, then point this computer at it —\n` +
              `  no need to pair again:\n\n` +
              `      pnpm agent set-server --server https://their-new-address\n`)
          }
          if (d.cause === 'dns-local-only' && !dnsFallbackEnabled()) {
            console.error(`  Or retry with:  DWP_DNS_FALLBACK=1 pnpm agent run\n`)
          }
        }).catch(() => {})
      }

      setTimeout(open, delay)
    }

    ws.on('close', (code: number, reason: Buffer) => {
      teardown(reason.toString('utf8') || `close ${code}`, code)
    })
    ws.on('error', (err: Error) => {
      // 'error' is usually followed by 'close'; teardown is idempotent so whichever
      // arrives first wins and the other is ignored.
      hostLog.warn('connect.error', { message: err.message })
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        try { ws.close() } catch { teardown(err.message) }
      } else {
        teardown(err.message)
      }
    })
  }

  open()
  for (const sig of ['SIGINT', 'SIGTERM'] as const) {
    process.on(sig, () => { stopped = true; process.exit(0) })
  }
}
