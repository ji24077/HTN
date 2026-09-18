import { migrate, pool } from './db.ts'
import { config } from './config.ts'
import { bootstrapUser } from './auth.ts'
import { buildServer } from './http.ts'
import { attachAgentHub, dispatchAll } from './hub.ts'
import { sweepExpiredLeases } from './scheduler.ts'

await migrate()
await bootstrapUser()

const app = buildServer()
await app.listen({ port: config.port, host: '0.0.0.0' })
attachAgentHub(app.server)

// Reclaim abandoned leases, then offer whatever came back to whoever is connected.
setInterval(() => {
  void sweepExpiredLeases()
    .then(n => (n > 0 ? dispatchAll() : undefined))
    .catch(err => console.error('[sweep]', err))
}, 5_000).unref()

// A steady low-rate dispatch pass covers hosts that connected between events.
setInterval(() => void dispatchAll().catch(() => {}), 2_000).unref()

console.log(`[control] listening on :${config.port}  public origin ${config.publicOrigin}`)

for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    void app.close().then(() => pool.end()).then(() => process.exit(0))
  })
}
