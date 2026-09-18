export const config = {
  databaseUrl: process.env.DATABASE_URL ?? 'postgres://dwp:dwp@localhost:5433/dwp',
  port: Number(process.env.PORT ?? 8787),
  publicOrigin: process.env.PUBLIC_ORIGIN ?? `http://localhost:${process.env.PORT ?? 8787}`,
  bootstrapEmail: process.env.BOOTSTRAP_EMAIL ?? 'operator@local',
  bootstrapPassword: process.env.BOOTSTRAP_PASSWORD ?? 'dwp-dev',
  /** A host is offline after three missed heartbeats. */
  heartbeatSeconds: 15,
  offlineAfterSeconds: 45,
  leaseSeconds: 30,
  maxAttempts: 3,
  pairCodeTtlMs: 10 * 60 * 1000,
}
