import { createServer, connect as tcpConnect, type Server, type Socket } from 'node:net'
import { once } from 'node:events'

/**
 * A fault-injecting TCP proxy that sits between agents and the control service.
 *
 * Everything an agent does travels through here, so this is where we reproduce the
 * conditions that matter and cannot be staged by hand: a laptop that sleeps, a router
 * that silently drops an idle mapping, a café network that blocks WebSocket upgrades,
 * a link with 400 ms of latency.
 *
 * It proxies raw bytes, so HTTP and WebSocket both pass through unchanged and the agent
 * needs no knowledge that it exists.
 */

export type Impairments = {
  /** One-way delay applied to every chunk in both directions. */
  latencyMs: number
  /** Random extra delay, uniform in [0, jitterMs]. */
  jitterMs: number
  /** Throughput ceiling per connection, bytes/sec. 0 means unlimited. */
  bandwidthBps: number
  /** Refuse new connections outright — the server looks unreachable. */
  refuseConnections: boolean
  /**
   * Accept and hold connections but forward nothing, in either direction.
   *
   * This is the sleeping-laptop case, and the important one: there is no FIN, so both
   * sides believe the connection is fine. Only heartbeats reveal it.
   */
  blackhole: boolean
  /** Destroy each connection this long after it opens. 0 disables. */
  killAfterMs: number
  /** Per-chunk chance of destroying the connection, 0..1. */
  resetProbability: number
  /** Answer WebSocket upgrades with 403 — the blocking-proxy case. */
  rejectUpgrade: boolean
  /** Drop connections idle this long, like a NAT reclaiming a mapping. 0 disables. */
  idleTimeoutMs: number
}

export const CLEAN: Impairments = {
  latencyMs: 0,
  jitterMs: 0,
  bandwidthBps: 0,
  refuseConnections: false,
  blackhole: false,
  killAfterMs: 0,
  resetProbability: 0,
  rejectUpgrade: false,
  idleTimeoutMs: 0,
}

export type NetSimStats = {
  accepted: number
  refused: number
  upgradesRejected: number
  killed: number
  bytesUp: number
  bytesDown: number
  openNow: number
}

export type NetSim = {
  readonly port: number
  readonly stats: NetSimStats
  /** Change conditions while the system is running. */
  set(patch: Partial<Impairments>): void
  /** Restore a clean network. */
  clear(): void
  /** Destroy every live connection right now, with no close frame. */
  cutAll(): number
  close(): Promise<void>
}

const wait = (ms: number): Promise<void> => new Promise(r => setTimeout(r, ms))

export async function startNetSim(targetPort: number, listenPort = 0): Promise<NetSim> {
  let impair: Impairments = { ...CLEAN }
  const live = new Set<Socket>()
  const stats: NetSimStats = {
    accepted: 0, refused: 0, upgradesRejected: 0, killed: 0,
    bytesUp: 0, bytesDown: 0, openNow: 0,
  }

  /**
   * Forward one direction with the configured impairments applied.
   *
   * Chunks are queued and drained in order: a naive per-chunk setTimeout would reorder
   * them under jitter, and TCP does not reorder. Getting this wrong would make the
   * simulator test something no real network does.
   */
  function pump(from: Socket, to: Socket, dir: 'up' | 'down'): void {
    const queue: Buffer[] = []
    let draining = false

    const drain = async (): Promise<void> => {
      if (draining) return
      draining = true
      while (queue.length > 0) {
        const chunk = queue.shift()!
        const delay = impair.latencyMs + (impair.jitterMs > 0 ? Math.random() * impair.jitterMs : 0)
        if (delay > 0) await wait(delay)

        if (impair.bandwidthBps > 0) {
          await wait((chunk.length / impair.bandwidthBps) * 1000)
        }
        if (impair.blackhole) continue          // swallow it; no FIN, no error
        if (impair.resetProbability > 0 && Math.random() < impair.resetProbability) {
          stats.killed += 1
          from.destroy()
          to.destroy()
          return
        }
        if (to.destroyed) return
        to.write(chunk)
        if (dir === 'up') stats.bytesUp += chunk.length
        else stats.bytesDown += chunk.length
      }
      draining = false
    }

    from.on('data', (chunk: Buffer) => {
      // Inspect the first client bytes only — enough to spot a WebSocket upgrade.
      if (dir === 'up' && impair.rejectUpgrade && /upgrade:\s*websocket/i.test(chunk.subarray(0, 1024).toString('latin1'))) {
        stats.upgradesRejected += 1
        from.write('HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
        from.end()
        to.destroy()
        return
      }
      queue.push(chunk)
      void drain()
    })
  }

  const server: Server = createServer(client => {
    if (impair.refuseConnections) {
      stats.refused += 1
      client.destroy()
      return
    }

    stats.accepted += 1
    stats.openNow += 1
    const upstream = tcpConnect({ port: targetPort, host: '127.0.0.1' })
    live.add(client)
    live.add(upstream)

    let killTimer: NodeJS.Timeout | undefined
    if (impair.killAfterMs > 0) {
      killTimer = setTimeout(() => {
        stats.killed += 1
        client.destroy()
        upstream.destroy()
      }, impair.killAfterMs)
    }

    if (impair.idleTimeoutMs > 0) {
      client.setTimeout(impair.idleTimeoutMs, () => { stats.killed += 1; client.destroy(); upstream.destroy() })
      upstream.setTimeout(impair.idleTimeoutMs, () => { stats.killed += 1; client.destroy(); upstream.destroy() })
    }

    const teardown = (): void => {
      if (killTimer) clearTimeout(killTimer)
      if (live.delete(client)) stats.openNow -= 1
      live.delete(upstream)
      client.destroy()
      upstream.destroy()
    }

    pump(client, upstream, 'up')
    pump(upstream, client, 'down')

    client.on('close', teardown)
    upstream.on('close', teardown)
    client.on('error', teardown)
    upstream.on('error', teardown)
  })

  server.listen(listenPort, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  const port = typeof address === 'object' && address ? address.port : listenPort

  return {
    port,
    stats,
    set: patch => { impair = { ...impair, ...patch } },
    clear: () => { impair = { ...CLEAN } },
    cutAll: () => {
      const n = live.size
      for (const s of live) s.destroy()
      live.clear()
      stats.openNow = 0
      stats.killed += n
      return n
    },
    close: async () => {
      for (const s of live) s.destroy()
      live.clear()
      server.close()
      await once(server, 'close').catch(() => {})
    },
  }
}
