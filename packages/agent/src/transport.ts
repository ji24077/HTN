import { freemem } from 'node:os'
import WebSocket from 'ws'
import {
  decode, envelope, mintAssertion, signAttestation, hashOutput,
  TaskOffer, HelloAck, Revoked,
} from '@dwp/protocol'
import type { KeyObject } from 'node:crypto'
import { createLogger } from '@dwp/protocol'
import { diagnoseOrigin } from '@dwp/protocol'
import { fallbackLookup, dnsFallbackEnabled, installDnsFallback, directDial } from './resolver.ts'
import { applyUpdate, restartIntoNewVersion } from './update.ts'
import { isCompiledBinary } from './paths.ts'
import { probe } from './capability.ts'
import { isPaused, type AgentConfig } from './config.ts'
import { runEcho } from './adapters/echo.ts'
import { runInference } from './adapters/inference.ts'
import { runWalker } from './adapters/walker.ts'

const log = createLogger({ component: 'agent' })

import { availableAdapters } from './workloads.ts'

/**
 * What this machine can actually run, decided when it connects rather than at import.
 *
 * Computing this at module load resolved the optional packages before `agent enable` had
 * installed them, and Node caches that failed lookup — so the very command that installs
 * a runtime could not then see it. Advertising an adapter whose runtime is missing would
 * also have the scheduler send work the host can only fail, which reads as a broken
 * machine rather than an absent option.
 */
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
/** How often a connected agent asks whether a newer release exists. */
const UPDATE_CHECK_MS = 10 * 60 * 1000

type Running = { controller: AbortController; leaseId: string; renew: NodeJS.Timeout }

export function connect(cfg: AgentConfig, privateKey: KeyObject): void {
  installDnsFallback()
  let backoffMs = 1_000
  let stopped = false
  let attempt = 0
  let connectedSince = 0
  let everConnected = false
  let explained = false
  /**
   * Wall-clock reading from the previous heartbeat tick, and how long a suspension
   * lasted. Both belong to the agent rather than to one connection: the whole point is
   * that they are read by the *next* connection, to explain why it exists. Declaring
   * them inside the per-connection scope meant the reconnect always saw zero and the
   * server never learned the machine had been asleep.
   */
  let lastTick = Date.now()
  let sleptForMs = 0
  const hostLog = log.child({ hostId: cfg.hostId, label: cfg.label })
  const running = new Map<string, Running>()

  const open = (): void => {
    if (stopped) return
    attempt += 1
    hostLog.info('connect.attempt', { attempt, url: cfg.wsUrl })
    // Resolve before dialling when this machine's own resolver cannot. The `lookup` hook
    // below covers Node, but a compiled binary uses its own WebSocket and ignores it —
    // so without this a machine with broken DNS could pair and then never connect.
    void directDial(cfg.wsUrl).catch(() => null).then(openWith)
  }

  const openWith = (direct: Awaited<ReturnType<typeof directDial>>): void => {
    if (stopped) return
    const dialStartedAt = Date.now()
    if (direct) {
      hostLog.info('connect.direct_address', {
        address: direct.url,
        reason: 'this machine cannot resolve the hostname; used a public resolver',
      })
    }
    const ws = new WebSocket(direct?.url ?? cfg.wsUrl, {
      ...(direct ? { servername: direct.options.servername } : {}),
      headers: {
        ...(direct?.options.headers ?? {}),
        authorization: `Bearer ${mintAssertion(cfg.hostId, privateKey)}`,
      },
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
    let updating = false
    let lastUpdateCheck = Date.now()

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
      hostLog.info('connect.established', {
        attempt,
        dialMs: Date.now() - dialStartedAt,
        url: cfg.wsUrl,
        // Carried into the first log line after waking, so the reconnect explains itself.
        ...(sleptForMs > 0 ? { afterSuspensionMs: Math.round(sleptForMs) } : {}),
      })
      send('hello', {
        capability: probe(availableAdapters()),
        consent: consent(),
        ...(sleptForMs > 0 ? { afterSuspensionMs: Math.round(sleptForMs) } : {}),
      })

      startHeartbeat()
    })

    /**
     * Take an update if one is offered, we can verify it, and no work is in flight.
     *
     * Updating while holding a task would abandon it mid-flight; the queue would recover
     * it, but throwing work away to install a patch is rude and unnecessary when waiting
     * costs nothing.
     */
    const maybeUpdate = (offered: string | null): void => {
      if (cfg.autoUpdate === false) return

      /**
       * A compiled binary ignores what the handshake announces.
       *
       * `hello.ack` carries the *source* release version, so a binary agent saw a
       * mismatch on every handshake and every poll — firing a needless fetch of the
       * binaries index and logging a line that read like an available update when there
       * was none. It was harmless, because applyUpdate routes to the binary path and
       * compares against the right index, but a permanent false positive is exactly the
       * sort of noise that hides a real one later.
       */
      if (isCompiledBinary()) {
        if (updating || running.size > 0) return
        updating = true
        void applyUpdate(cfg, privateKey).then(onUpdateResult).catch((err: unknown) => {
          hostLog.warn('update.failed', { err })
          updating = false
        })
        return
      }

      if (!offered) return
      if (offered === cfg.installedRelease) return
      if (running.size > 0) return
      if (updating) return
      updating = true

      hostLog.info('update.available', { have: cfg.installedRelease ?? null, offered })
      void applyUpdate(cfg, privateKey).then(onUpdateResult).catch((err: unknown) => {
        hostLog.warn('update.failed', { err })
        updating = false
      })
    }

    const onUpdateResult = (result: Awaited<ReturnType<typeof applyUpdate>>): void => {
      if (result.status === 'updated') {
        console.log(`\n  Updated to ${result.to}. Restarting.\n`)
        stopped = true
        try { ws.close(1000, 'updating') } catch {}
        setTimeout(restartIntoNewVersion, 500)
        return
      }
      if (result.status === 'refused') {
        hostLog.error('update.refused', { reason: result.reason })
        console.error(`\n  Refused an update: ${result.reason}\n`)
      } else if (result.status === 'unavailable') {
        hostLog.warn('update.unavailable', { reason: result.reason })
      }
      // 'current' is the ordinary outcome for a binary: nothing to say about it.
      updating = false
    }

    const startHeartbeat = (): void => {
      if (heartbeat) clearInterval(heartbeat)
      /**
       * Restart the suspension clock with the interval that reads it.
       *
       * Without this, the gap while reconnecting counts as drift: after an eight-second
       * backoff the first tick of the new interval looks exactly like eight seconds of
       * suspended time, and the agent reports a sleep that never happened. The flaky-wifi
       * scenario produced three such false reports in one run.
       */
      lastTick = Date.now()
      heartbeat = setInterval(() => {
        /**
         * Detect suspension.
         *
         * A timer set for 15 seconds that fires 900 seconds later did not run late — the
         * machine was suspended in between, and every socket it held is stale even though
         * nothing reported an error. Without this, a sleeping laptop is indistinguishable
         * from a network outage in the logs, and the two need different explanations.
         */
        const drift = Date.now() - lastTick - heartbeatMs
        if (drift > heartbeatMs * 2) {
          sleptForMs = drift
          hostLog.warn('machine.woke', {
            suspendedForMs: Math.round(drift),
            note: 'timers did not fire — this machine was asleep or suspended',
          })
          lastTick = Date.now()
          // Every socket held across a suspension is stale. Do not wait to discover that.
          ws.terminate()
          teardown('woke from sleep')
          return
        }
        lastTick = Date.now()

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

        // An agent that stays connected for days would otherwise only ever learn about a
        // release at connect time, which for a stable machine means never.
        if (cfg.autoUpdate !== false && Date.now() - lastUpdateCheck > UPDATE_CHECK_MS) {
          lastUpdateCheck = Date.now()
          void fetch(`${cfg.server}/release/latest`, { signal: AbortSignal.timeout(15_000) })
            .then(r => (r.ok ? r.json() : null))
            .then((r: { manifest?: { version?: string } } | null) => maybeUpdate(r?.manifest?.version ?? null))
            .catch(() => {})
        }
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
          maybeUpdate(ack.data.releaseVersion)

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

        if (isPaused() || !cfg.allowCompute || !availableAdapters().includes(offer.adapter)) {
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
