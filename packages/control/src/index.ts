import { migrate, pool } from './db.ts'
import { config, assertSafeToExpose } from './config.ts'
import { bootstrapUser } from './auth.ts'
import { buildServer } from './http.ts'
import { attachAgentHub, dispatchAll } from './hub.ts'
import { sweepExpiredLeases } from './scheduler.ts'
import { log } from './logger.ts'

assertSafeToExpose()

await migrate()
await bootstrapUser()

// Nobody is connected to a process that has just started. Presence lives in memory, so
// without this the fleet keeps showing whoever was online when the server was last
// killed — a dashboard that confidently lists dead machines is worse than an empty one.
const { rowCount: staleHosts } = await pool.query(`update hosts set online = false where online`)
if (staleHosts) log.info('presence.reset_on_boot', { hostsMarkedOffline: staleHosts })

const app = buildServer()
await app.listen({ port: config.port, host: config.bindHost })
attachAgentHub(app.server)

// Reclaim abandoned leases, then offer whatever came back to whoever is connected.
setInterval(() => {
  void sweepExpiredLeases()
    .then(n => (n > 0 ? dispatchAll() : undefined))
    .catch((err: unknown) => log.error('sweep.failed', { err }))
}, 5_000).unref()

// A steady low-rate dispatch pass covers hosts that connected between events.
setInterval(() => void dispatchAll().catch(() => {}), 2_000).unref()

log.info('control.started', {
  bind: `${config.bindHost}:${config.port}`,
  publicOrigin: config.publicOrigin,
  reachableFromInternet: config.isPublic,
  heartbeatSeconds: config.heartbeatSeconds,
  leaseSeconds: config.leaseSeconds,
})

for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    void app.close().then(() => pool.end()).then(() => process.exit(0))
  })
}
