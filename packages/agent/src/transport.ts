import { freemem } from 'node:os'
import WebSocket from 'ws'
import {
  decode, envelope, mintAssertion, signAttestation, hashOutput, resultTooLarge, JSON_LIMIT,
  TaskOffer, HelloAck, Revoked,
} from '@dwp/protocol'
import type { KeyObject } from 'node:crypto'
import { createLogger } from '@dwp/protocol'
import { diagnoseOrigin } from '@dwp/protocol'
import { fallbackLookup, dnsFallbackEnabled, installDnsFallback, directDial } from './resolver.ts'
import { applyUpdate, restartIntoNewVersion } from './update.ts'
import { AGENT_VERSION, isCompiledBinary } from './paths.ts'
import { probe } from './capability.ts'
import { isPaused, type AgentConfig } from './config.ts'
import { runEcho } from './adapters/echo.ts'
import { runInference } from './adapters/inference.ts'
import { runWalker } from './adapters/walker.ts'
import { SupersessionPolicy } from './supersession.ts'
import * as history from './history.ts'
import type { RunRecord } from './history.ts'
import { captureWorkloadFailure } from './telemetry.ts'
import { applyManagedTelemetry } from './managed-telemetry.ts'
import { ExecutionJournal } from './execution.ts'

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

type Running = {
  controller: AbortController
  leaseId: string
  renew: NodeJS.Timeout
  /** Carried so a window can say "running walker_evolution" rather than a bare task id. */
  adapter: string
  startedAt: number
  finishTracking: (kind: 'interrupted' | 'cancelled') => void
}

/**
 * Everything a window needs to describe this agent, and nothing it does not.
 *
 * The desktop app is the same process as the agent, so this is passed by reference
 * rather than scraped from a log — there is no second source of truth to drift. It is a
 * plain snapshot on purpose: the window polls, so a missed notification costs one second
 * of staleness rather than a wrong display that never corrects itself.
 */
export type AgentState = {
  connection: 'offline' | 'connecting' | 'online'
  attempt: number
  connectedSince: number | null
  running: { taskId: string; adapter: string; startedAt: number }[]
  lastLostReason: string | null
  /**
   * The plain-language diagnosis produced after repeated failures.
   *
   * This is the single most useful thing a non-technical owner can be shown — usually
   * "the address changed, ask for a new invite" — and until now it only ever reached a
   * console that an app user never sees.
   */
  advice: string | null
  /** Another agent holds this host's identity, so this one has stood down. */
  stoodDown: boolean
  updating: boolean
  /**
   * When this machine last finished a task, so a window can say "last run 4 min ago"
   * without reading the history file. Everything else about past runs comes from the
   * history module directly.
   */
  lastRunAt: number | null
}

/** Enough of a handle for a window to steer the connection. */
export type AgentHandle = {
  /** Try now rather than waiting out the stand-down delay. */
  retryNow(): void
  /** Stop for good: no more work, no reconnection. Used when leaving a network. */
  stop(): void
}

export function connect(
  cfg: AgentConfig,
  privateKey: KeyObject,
  observe?: (state: AgentState) => void,
): AgentHandle {
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
  const supersession = new SupersessionPolicy()
  const hostLog = log.child({ hostId: cfg.hostId, label: cfg.label })
  const running = new Map<string, Running>()
  const journal = new ExecutionJournal(cfg.server, cfg.hostId)

  const state: AgentState = {
    connection: 'offline', attempt: 0, connectedSince: null, running: [],
    lastLostReason: null, advice: null, stoodDown: false, updating: false, lastRunAt: null,
  }
  /** Derive the task list from the live map rather than maintaining a second copy. */
  const notify = (patch: Partial<AgentState> = {}): void => {
    Object.assign(state, patch)
    state.running = [...running].map(([taskId, r]) =>
      ({ taskId, adapter: r.adapter, startedAt: r.startedAt }))
    observe?.({ ...state, running: [...state.running] })
  }

  /** The socket currently in hand, so the caller can close it deliberately. */
  let current: WebSocket | null = null

  const open = (): void => {
    if (stopped) return
    attempt += 1
    notify({ connection: 'connecting', attempt })
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
    current = ws
    let heartbeat: NodeJS.Timeout | undefined
    let heartbeatMs = DEFAULT_HEARTBEAT_MS
    let pauseWas = isPaused()
    // Last time we heard ANYTHING from the server. A link can fail in one direction
    // only — our writes disappear into it while the socket still looks open — so the
    // absence of inbound traffic is the only reliable signal we have.
    let lastInbound = Date.now()
    let updating = false
    let lastUpdateCheck = Date.now()
    let trackExecutions = false
    const eventTimer = setInterval(() => {
      if (trackExecutions && ws.readyState === WebSocket.OPEN && ws.bufferedAmount < 128 * 1024) {
        journal.flush(batch => send('task.events', batch))
      }
    }, 250)

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
      notify({ connection: 'online', connectedSince, lastLostReason: null, advice: null, stoodDown: false })
      hostLog.info('connect.established', {
        attempt,
        dialMs: Date.now() - dialStartedAt,
        url: cfg.wsUrl,
        // Carried into the first log line after waking, so the reconnect explains itself.
        ...(sleptForMs > 0 ? { afterSuspensionMs: Math.round(sleptForMs) } : {}),
      })
      send('hello', {
        capability: probe(availableAdapters(), cfg.installedRelease),
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
        notify({ updating: true })
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

      notify({ updating: true })
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
        supersession.cancelPending()
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
      notify({ updating: false })
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
          applyManagedTelemetry(cfg, ack.data.telemetry)
          trackExecutions = ack.data.executionEvents === true
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

      if (msg.type === 'task.events.ack') {
        journal.acknowledge(msg.payload)
        return
      }

      if (msg.type === 'revoked') {
        const r = Revoked.safeParse(msg.payload)
        hostLog.error('host.revoked', { reason: r.success ? r.data.reason : 'unknown' })
        stopped = true
        supersession.cancelPending()
        ws.close()
        process.exitCode = 1
        return
      }

      if (msg.type === 'task.cancel') {
        const taskId = (msg.payload as { taskId?: string }).taskId
        const r = taskId ? running.get(taskId) : undefined
        if (r) { r.finishTracking('cancelled'); r.controller.abort(); clearInterval(r.renew); running.delete(taskId!) }
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
        const report = journal.reporter(offer.taskId, offer.attempt)
        let trackedFinished = false
        const finishTracking = (kind: 'succeeded' | 'failed' | 'cancelled' | 'timed_out' | 'interrupted', data = {}) => {
          if (trackedFinished) return
          trackedFinished = true
          journal.emit(offer.taskId, offer.attempt, kind, { duration_ms: performance.now() - t0, ...data })
        }
        const renewMs = Math.max(1_000, Math.floor((offer.leaseSeconds * 1000) / 3))
        const renew = setInterval(() => send('lease.renew', { taskId: offer.taskId, leaseId: offer.leaseId }), renewMs)
        running.set(offer.taskId, {
          controller, leaseId: offer.leaseId, renew, adapter: offer.adapter, startedAt: Date.now(),
          finishTracking,
        })
        notify()

        const startedAt = new Date().toISOString()
        const t0 = performance.now()
        const cpu0 = history.cpuStart()
        /**
         * Whether anything else was already running when this started.
         *
         * `process.cpuUsage()` is process-wide, so with two tasks in flight the figure
         * recorded against each is really both of them. Noting it costs one boolean and
         * lets the window say "(shared)" rather than quietly overstate one task's cost.
         */
        const overlapped = running.size > 1
        let outcome: RunRecord['outcome'] = 'error'
        let errorClass: string | undefined
        let message: string | undefined
        let outputBytes: number | undefined
        journal.emit(offer.taskId, offer.attempt, 'started', {
          adapter: offer.adapter, job_id: offer.jobId, agent_version: AGENT_VERSION,
          runtime: process.versions.bun ? `bun ${process.versions.bun}` : `node ${process.version}`,
          os: process.platform, arch: process.arch, input_hash: hashOutput(offer.input),
        })
        report.progress(0, 1)
        const wallClock = setTimeout(() => { finishTracking('timed_out'); controller.abort() }, offer.wallClockMs)
        void wallClock

        const work =
          offer.adapter === 'cpu_inference_batch'
            ? runInference(offer.input, {
                hostId: cfg.hostId, server: cfg.server, privateKey, signal: controller.signal, report,
              })
          : offer.adapter === 'walker_evolution'
            ? runWalker(offer.input, cfg.hostId, controller.signal, report)
          : runEcho(offer.input, cfg.hostId, controller.signal, report)

        void work
          .then(output => {
            if (controller.signal.aborted) return
            const finishedAt = new Date().toISOString()
            /**
             * Say so here rather than shipping a result the server will not store.
             *
             * The server refuses an oversized result, and used to refuse it by closing
             * the connection — which marked this host unhealthy, re-queued the slice, and
             * handed the same too-big slice to the next machine to fail the same way.
             * Reporting it as this task's error keeps the failure where it belongs and
             * puts the real reason on the task instead of `worker_reconnected`.
             */
            if (resultTooLarge(output)) {
              hostLog.error('task.result_too_large', {
                taskId: offer.taskId, adapter: offer.adapter, limitBytes: JSON_LIMIT,
                note: 'this slice produced more output than one result may carry — split it',
              })
              outcome = 'result_too_large'
              errorClass = 'result_too_large'
              message = `result exceeds the ${JSON_LIMIT}-byte limit; submit this work in smaller slices`
              send('task.error', {
                taskId: offer.taskId,
                leaseId: offer.leaseId,
                errorClass: 'result_too_large',
                message,
              })
              return
            }
            const outputHash = hashOutput(output)
            outcome = 'ok'
            try { outputBytes = Buffer.byteLength(JSON.stringify(output) ?? '') } catch { /* unmeasurable */ }
            report.progress(1, 1)
            finishTracking('succeeded', { output_hash: outputHash, result_url: `/v1/tasks/${offer.taskId}` })
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
            finishTracking(controller.signal.aborted ? 'cancelled' : 'failed', {
              // Bounded so a deep async stack cannot push the one event operators need past the 8 KiB cap.
              error: err instanceof Error
                ? { name: err.name, message: err.message.slice(0, 2048), stack: err.stack?.slice(0, 4096) }
                : String(err).slice(0, 2048),
            })
            if (!controller.signal.aborted) {
              captureWorkloadFailure(err, {
                workerId: cfg.hostId, taskId: offer.taskId, adapter: offer.adapter, attempt: offer.attempt,
                jobId: offer.jobId,
              })
            }
            outcome = controller.signal.aborted ? 'aborted' : 'error'
            errorClass = controller.signal.aborted ? 'aborted' : 'adapter_error'
            message = (err instanceof Error ? err.message : String(err)).slice(0, 200)
            send('task.error', {
              taskId: offer.taskId,
              leaseId: offer.leaseId,
              errorClass,
              message: err instanceof Error ? err.message : String(err),
            })
          })
          .finally(() => {
            clearTimeout(wallClock)
            clearInterval(renew)
            running.delete(offer.taskId)
            // Before notify(), so the window's very next poll already sees this run.
            // record() swallows its own failures; nothing here can throw.
            const finishedAt = Date.now()
            history.record({
              taskId: offer.taskId,
              jobId: offer.jobId,
              adapter: offer.adapter,
              attempt: offer.attempt,
              startedAt,
              finishedAt: new Date(finishedAt).toISOString(),
              durationMs: Number((performance.now() - t0).toFixed(3)),
              outcome,
              ...(errorClass === undefined ? {} : { errorClass }),
              ...(message === undefined ? {} : { message: message.slice(0, 200) }),
              cpuMs: history.cpuMsSince(cpu0),
              rssMb: history.rssMb(),
              shared: overlapped || running.size > 0,
              ...(outputBytes === undefined ? {} : { outputBytes }),
              agentVersion: cfg.installedRelease ?? AGENT_VERSION,
            })
            notify({ lastRunAt: finishedAt })
          })
      }
    })

    let tornDown = false
    const teardown = (why: string, code?: number): void => {
      if (tornDown) return
      tornDown = true
      if (heartbeat) clearInterval(heartbeat)
      clearInterval(eventTimer)
      const abandoned = [...running.keys()]
      for (const [, r] of running) { r.finishTracking('interrupted'); r.controller.abort(); clearInterval(r.renew) }
      running.clear()

      const heldMs = connectedSince ? Date.now() - connectedSince : 0
      hostLog.warn('connect.lost', {
        why,
        closeCode: code ?? null,
        connectedMs: heldMs,
        abandonedTasks: abandoned.length,
      })
      connectedSince = 0
      notify({ connection: 'offline', connectedSince: null, lastLostReason: why })
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
          notify({ advice: d.message })
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
      /**
       * 4000 means the server handed this host's identity to a newer connection.
       *
       * Retrying would take it straight back and supersede the other one, which would
       * retry in turn — two agents on one machine trading the connection forever, each
       * abandoning the other's work. Stand down and say so instead. This is the case the
       * desktop app makes easy to hit: launching the window while a login-started agent
       * is already running.
       */
      if (code === 4000) {
        const { count, giveUp, retryInMs } = supersession.standDown(() => {
          stopped = false
          open()
        })
        hostLog.warn('host.superseded', { count, giveUp })

        // Stop teardown from scheduling the ordinary retry; this case has its own pace.
        stopped = true
        teardown('superseded by another agent on this host', code)
        notify({ stoodDown: true })

        console.error(
          `\n  Another copy of the agent is already connected as "${cfg.label}".\n` +
          `  This one has stood down rather than fight it for the connection.\n\n` +
          `  That is usually an agent installed from a terminal and started at login.\n` +
          `  Stop that one if you want this one to take over.\n`)

        if (giveUp) {
          console.error(`  Tried ${count} times; not trying again.\n`)
          return
        }
        /**
         * Look again shortly. Standing down must not be permanent.
         *
         * The first version set `stopped` and never retried, so an app that lost the
         * race sat showing "stopped" forever — including after the other copy had been
         * stopped, which is exactly when its owner expects it to come back. Nothing was
         * connected, and nothing would be until someone relaunched it by hand.
         */
        hostLog.info('host.superseded_retry_scheduled', { inMs: Math.round(retryInMs!) })
        return
      }
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
    process.on(sig, () => {
      stopped = true; supersession.cancelPending()
      for (const r of running.values()) r.finishTracking('interrupted')
      process.exit(0)
    })
  }

  return {
    stop: () => {
      stopped = true
      notify({ connection: 'offline', connectedSince: null })
      try { current?.close(1000, 'left the network') } catch {}
    },
    retryNow: () => {
      if (!stopped || !state.stoodDown) return
      supersession.reset()
      stopped = false
      notify({ stoodDown: false })
      open()
    },
  }
}
