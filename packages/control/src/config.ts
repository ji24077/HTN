const port = Number(process.env.PORT ?? 8787)
const publicOrigin = process.env.PUBLIC_ORIGIN ?? `http://localhost:${port}`

/** True when the control service is reachable from outside this machine. */
const isPublic = !/^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|$)/.test(publicOrigin)

export const config = {
  databaseUrl: process.env.DATABASE_URL ?? 'postgres://dwp:dwp@localhost:5433/dwp',
  port,
  publicOrigin,
  isPublic,

  /**
   * Loopback by default.
   *
   * The tunnel runs on this same machine and connects over loopback, so there is no
   * reason to listen on the LAN. Binding 0.0.0.0 would hand every device on the
   * coffee-shop wifi an unauthenticated path to the port.
   */
  bindHost: process.env.BIND_HOST ?? '127.0.0.1',

  bootstrapEmail: process.env.BOOTSTRAP_EMAIL ?? 'operator@local',
  bootstrapPassword: process.env.BOOTSTRAP_PASSWORD ?? 'dwp-dev',

  /**
   * Timings live here and nowhere else.
   *
   * The server announces the heartbeat cadence to agents at handshake and the lease
   * duration with every offer, so changing these values changes the whole system's
   * behaviour without touching the agent. That is what lets the simulator compress
   * hours of real-world flakiness into seconds while exercising the same code paths.
   */
  heartbeatSeconds: Number(process.env.DWP_HEARTBEAT_SECONDS ?? 15),
  offlineAfterSeconds: Number(process.env.DWP_OFFLINE_AFTER_SECONDS ?? 45),
  leaseSeconds: Number(process.env.DWP_LEASE_SECONDS ?? 30),
  maxAttempts: Number(process.env.DWP_MAX_ATTEMPTS ?? 3),
  pairCodeTtlMs: Number(process.env.DWP_PAIR_CODE_TTL_MS ?? 10 * 60 * 1000),
}

const WEAK = new Set(['dwp-dev', 'change-me', 'password', 'admin', ''])

/**
 * Refuse to expose a laptop to the internet behind a default password.
 *
 * This is a startup failure rather than a warning on purpose: a warning scrolls past
 * and the service keeps running, which is exactly the outcome to prevent here.
 */
export function assertSafeToExpose(): void {
  if (!config.isPublic) return
  const problems: string[] = []

  if (WEAK.has(config.bootstrapPassword) || config.bootstrapPassword.length < 12) {
    problems.push('BOOTSTRAP_PASSWORD is a default or shorter than 12 characters')
  }
  if (!config.publicOrigin.startsWith('https://')) {
    problems.push(`PUBLIC_ORIGIN is not https (${config.publicOrigin}) — credentials would cross the internet in clear text`)
  }
  if (config.bindHost !== '127.0.0.1' && config.bindHost !== 'localhost') {
    problems.push(`BIND_HOST is ${config.bindHost}; the tunnel connects over loopback, so this exposes the port to your local network too`)
  }

  if (problems.length > 0) {
    console.error('\n  Refusing to start: this instance is reachable from the internet.\n')
    for (const p of problems) console.error(`    - ${p}`)
    console.error('\n  Fix these in .env, then start again. `pnpm share` sets them up for you.\n')
    process.exit(1)
  }
}
