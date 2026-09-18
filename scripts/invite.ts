/**
 * Create another invite link without restarting anything.
 *
 *   node scripts/invite.ts [label]
 *
 * Talks to the already-running server over loopback, so it works even when this machine
 * cannot resolve its own public address.
 */
import { readFileSync, existsSync } from 'node:fs'

const PORT = Number(process.env.PORT ?? 8787)
const LOCAL = `http://127.0.0.1:${PORT}`

function env(key: string): string | undefined {
  if (process.env[key]) return process.env[key]
  if (!existsSync('.env')) return undefined
  for (const line of readFileSync('.env', 'utf8').split('\n')) {
    const m = new RegExp(`^${key}=(.*)$`).exec(line.trim())
    if (m) return m[1]
  }
  return undefined
}

const email = env('BOOTSTRAP_EMAIL') ?? 'operator@local'
const password = env('BOOTSTRAP_PASSWORD') ?? ''
const label = process.argv[2] ?? `friend-${new Date().toISOString().slice(11, 16).replace(':', '')}`

let health: { publicOrigin?: string }
try {
  health = await (await fetch(`${LOCAL}/health`, { signal: AbortSignal.timeout(4000) })).json() as typeof health
} catch {
  console.error(`\n  No server is running on port ${PORT}.\n\n  Start one first:\n      pnpm share\n`)
  process.exit(1)
}

const loginRes = await fetch(`${LOCAL}/auth/login`, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email, password }),
})
if (!loginRes.ok) {
  console.error(`\n  Could not sign in as ${email} (HTTP ${loginRes.status}).\n` +
    `  Check BOOTSTRAP_EMAIL and BOOTSTRAP_PASSWORD in .env.\n`)
  process.exit(1)
}
const cookie = (loginRes.headers.getSetCookie?.()[0] ?? '').split(';')[0]!

const res = await fetch(`${LOCAL}/hosts/pair-code`, {
  method: 'POST', headers: { 'content-type': 'application/json', cookie },
  body: JSON.stringify({ label }),
})
const { code } = await res.json() as { code: string }
const origin = health.publicOrigin ?? LOCAL

console.log(`
  Invite for "${label}" — expires in 10 minutes, works once:

      ${origin}/join?code=${code}
`)
