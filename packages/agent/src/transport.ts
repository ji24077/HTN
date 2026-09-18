import { freemem } from 'node:os'
import WebSocket from 'ws'
import {
  decode, envelope, mintAssertion, signAttestation, hashOutput,
  TaskOffer, HelloAck, Revoked,
} from '@dwp/protocol'
import type { KeyObject } from 'node:crypto'
import { probe } from './capability.ts'
import { isPaused, type AgentConfig } from './config.ts'
import { runEcho } from './adapters/echo.ts'

const ADAPTERS = ['echo']
const HEARTBEAT_MS = 15_000
const RENEW_MS = 10_000

type Running = { controller: AbortController; leaseId: string; renew: NodeJS.Timeout }

export function connect(cfg: AgentConfig, privateKey: KeyObject): void {
  let backoffMs = 1_000
  let stopped = false
  const running = new Map<string, Running>()

  const open = (): void => {
    if (stopped) return
    const ws = new WebSocket(cfg.wsUrl, {
      headers: { authorization: `Bearer ${mintAssertion(cfg.hostId, privateKey)}` },
    })
    let heartbeat: NodeJS.Timeout | undefined
    let pauseWas = isPaused()

    const send = (type: string, payload: unknown, replyTo?: string): void => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(envelope(type, payload, replyTo)))
    }

    const consent = () => ({
      paused: isPaused(),
      allowCompute: cfg.allowCompute,
      allowBrowser: cfg.allowBrowser,
      maxConcurrency: cfg.maxConcurrency,
    })

    ws.on('open', () => {
      console.log(`[agent] connected to ${cfg.wsUrl}`)
      backoffMs = 1_000
      send('hello', { capability: probe(ADAPTERS), consent: consent() })

      heartbeat = setInterval(() => {
        // The local pause flag is authoritative and is pushed the moment it changes.
        const paused = isPaused()
        if (paused !== pauseWas) {
          pauseWas = paused
          send('consent.update', consent())
          if (paused) for (const [taskId, r] of running) { r.controller.abort(); clearInterval(r.renew); running.delete(taskId) }
        }
        send('heartbeat', { freeRamMb: Math.round(freemem() / 1024 / 1024), running: running.size })
      }, HEARTBEAT_MS)
    })

    ws.on('message', (raw: Buffer) => {
      const msg = decode(raw)
      if (!msg) return

      if (msg.type === 'hello.ack') {
        const ack = HelloAck.safeParse(msg.payload)
        if (ack.success) console.log(`[agent] host ${ack.data.hostId} online`)
        return
      }

      if (msg.type === 'revoked') {
        const r = Revoked.safeParse(msg.payload)
        console.error(`[agent] this host was revoked (${r.success ? r.data.reason : 'unknown'}). Stopping.`)
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
        const renew = setInterval(() => send('lease.renew', { taskId: offer.taskId, leaseId: offer.leaseId }), RENEW_MS)
        running.set(offer.taskId, { controller, leaseId: offer.leaseId, renew })

        const startedAt = new Date().toISOString()
        const t0 = performance.now()
        const wallClock = setTimeout(() => controller.abort(), offer.wallClockMs)

        void runEcho(offer.input, cfg.hostId, controller.signal)
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

    const teardown = (why: string): void => {
      if (heartbeat) clearInterval(heartbeat)
      for (const [, r] of running) { r.controller.abort(); clearInterval(r.renew) }
      running.clear()
      if (stopped) return
      // Full jitter: a fleet reconnecting after an outage must not arrive in lockstep.
      const delay = Math.random() * backoffMs
      backoffMs = Math.min(backoffMs * 2, 60_000)
      console.warn(`[agent] disconnected (${why}); retrying in ${Math.round(delay)}ms`)
      setTimeout(open, delay)
    }

    ws.on('close', (code: number) => teardown(`close ${code}`))
    ws.on('error', (err: Error) => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close()
      else teardown(err.message)
    })
  }

  open()
  for (const sig of ['SIGINT', 'SIGTERM'] as const) {
    process.on(sig, () => { stopped = true; process.exit(0) })
  }
}
