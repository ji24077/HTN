/**
 * Put this machine's control service on the internet, and print how to join it.
 *
 *   node scripts/share.ts
 *
 * Order matters: the tunnel has to exist before the control service starts, because the
 * public URL is what agents record at pairing time and it is not known until the tunnel
 * says so. Starting the service first and patching the URL afterwards is how you end up
 * with agents pointed at localhost.
 *
 * Note the tunnel targets 127.0.0.1 rather than "localhost": on macOS that name resolves
 * to ::1 first, and the control service listens on IPv4 loopback, so the tunnel would
 * connect to nothing and every public request would fail.
 */
import { spawn, type ChildProcess } from 'node:child_process'
import { createWriteStream, existsSync, readFileSync, writeFileSync } from 'node:fs'
import { randomBytes } from 'node:crypto'
import { createInterface } from 'node:readline'
import { createServer } from 'node:net'
import { probeUrl } from './lib/net-probe.ts'
import { cloudflaredInstallHint, envPrefix, portHolderHint, resolveCommand, stopChild, stopServerHint } from './lib/platform.ts'

const ENV_PATH = '.env'
const PORT = Number(process.env.PORT ?? 8787)
const children: ChildProcess[] = []
let stopping = false

/**
 * --url lets you bring your own public address instead of a throwaway tunnel.
 *
 * This matters more than it looks. A quick tunnel gets a new random hostname every time
 * it restarts, and agents remember the address they paired with — so every restart means
 * every friend has to re-point their agent. A stable address (a named Cloudflare tunnel,
 * an ngrok reserved domain, a Tailscale Funnel hostname) makes the network survive
 * restarts, and is the only sensible option once more than one person uses it.
 */
const argv = process.argv.slice(2)
const manualUrl = (() => {
  const i = argv.indexOf('--url')
  return i >= 0 ? argv[i + 1]?.replace(/\/+$/, '') : undefined
})()

function shutdown(code = 0): never {
  stopping = true
  // POSIX gets the same SIGTERM as before; Windows has none to deliver and leaves
  // grandchildren running, so stopChild takes the tree there instead.
  for (const c of children) stopChild(c)
  process.exit(code)
}
for (const sig of ['SIGINT', 'SIGTERM'] as const) process.on(sig, () => shutdown(0))

// --------------------------------------------------------------- credentials

type Env = Record<string, string>

function readEnv(): Env {
  if (!existsSync(ENV_PATH)) return {}
  const out: Env = {}
  for (const line of readFileSync(ENV_PATH, 'utf8').split('\n')) {
    const m = /^([A-Z0-9_]+)=(.*)$/.exec(line.trim())
    if (m) out[m[1]!] = m[2]!
  }
  return out
}

function writeEnvKey(key: string, value: string): void {
  const lines = existsSync(ENV_PATH) ? readFileSync(ENV_PATH, 'utf8').split('\n') : []
  const i = lines.findIndex(l => l.startsWith(`${key}=`))
  if (i >= 0) lines[i] = `${key}=${value}`
  else lines.push(`${key}=${value}`)
  writeFileSync(ENV_PATH, lines.join('\n').replace(/\n+$/, '') + '\n')
}

const env = readEnv()
const WEAK = new Set(['dwp-dev', 'change-me', 'password', 'admin', ''])

// A laptop about to accept connections from the internet does not get a default password.
if (WEAK.has(env.BOOTSTRAP_PASSWORD ?? '') || (env.BOOTSTRAP_PASSWORD ?? '').length < 12) {
  const generated = randomBytes(12).toString('base64url')
  writeEnvKey('BOOTSTRAP_PASSWORD', generated)
  env.BOOTSTRAP_PASSWORD = generated
  console.log(`\n  Generated a strong operator password and saved it to .env`)
  console.log(`    ${generated}\n`)
  console.log(`  This replaces the placeholder. It is yours alone — do not send it to anyone joining.\n`)
}
if (!env.BOOTSTRAP_EMAIL) {
  writeEnvKey('BOOTSTRAP_EMAIL', 'operator@local')
  env.BOOTSTRAP_EMAIL = 'operator@local'
}

// ------------------------------------------------------------------ preflight

/**
 * Fail early and in plain language if the port is taken.
 *
 * This script gets started and stopped constantly, and a leftover server from the last
 * run is the most likely reason it will not start. An EADDRINUSE stack trace does not
 * tell anyone what to do about that.
 */
async function assertPortFree(port: number): Promise<void> {
  const inUse = await new Promise<boolean>(resolve => {
    const probe = createServer()
    probe.once('error', (err: NodeJS.ErrnoException) => resolve(err.code === 'EADDRINUSE'))
    probe.once('listening', () => probe.close(() => resolve(false)))
    probe.listen(port, '127.0.0.1')
  })
  if (inUse) {
    console.error(`
  Port ${port} is already in use — most likely a server from a previous run.

  Find and stop it:
      ${portHolderHint(port)}
      ${stopServerHint()}

  Or use a different port:
      ${envPrefix('PORT', '8788', 'node scripts/share.ts')}
`)
    process.exit(1)
  }
}

await assertPortFree(PORT)

// -------------------------------------------------------------------- tunnel

const TUNNEL_LOG = '.dwp-tunnel.log'
async function openQuickTunnel(): Promise<string> {
  console.log('  Opening a public tunnel to this machine...')
  const tunnelLog = createWriteStream(TUNNEL_LOG, { flags: 'w' })

  // resolveCommand is an identity everywhere but Windows, where a bare name reaches
  // neither a .cmd shim nor a PATH-resolved .exe.
  const tunnel = spawn(resolveCommand('cloudflared'), ['tunnel', '--no-autoupdate', '--url', `http://127.0.0.1:${PORT}`], {
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  children.push(tunnel)
  tunnel.stdout?.pipe(tunnelLog)
  tunnel.stderr?.pipe(tunnelLog)

  tunnel.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'ENOENT') {
      console.error(`\n  cloudflared is not installed.\n\n    ${cloudflaredInstallHint()}\n`)
    } else {
      console.error('\n  Could not start the tunnel:', err.message, '\n')
    }
    shutdown(1)
  })

  return new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('the tunnel did not report a URL within 60s')), 60_000)
    const scan = (chunk: Buffer): void => {
      const m = /https:\/\/[a-z0-9-]+\.trycloudflare\.com/.exec(chunk.toString('utf8'))
      if (m) { clearTimeout(timer); resolve(m[0]) }
    }
    tunnel.stdout?.on('data', scan)
    tunnel.stderr?.on('data', scan)   // cloudflared prints its banner to stderr
  })
}

const publicOrigin: string = manualUrl
  ? manualUrl
  : await openQuickTunnel().catch((err: Error) => { console.error(`\n  ${err.message}\n`); shutdown(1) })

if (manualUrl) {
  console.log(`  Using the address you supplied: ${manualUrl}`)
  console.log(`  ${'(make sure your own tunnel is already forwarding it to 127.0.0.1:' + PORT + ')'}\n`)
}

// ------------------------------------------------------------------- control
//
// Start the service as soon as the URL is known. It binds loopback only, so it is not
// reachable by anyone until Cloudflare finishes registering the route — which is the
// slow part, and is what the wait below is for.

/**
 * Supervise the server rather than dying with it.
 *
 * The tunnel owns the public address, and agents remember the address they paired with.
 * Tearing the tunnel down because the server stopped means every friend's agent breaks
 * over something as ordinary as a restart or a crash — which is exactly what happened
 * the first time the server was restarted with a real second machine connected.
 *
 * So: keep the tunnel, restart the server underneath it, and the address survives.
 */
let control: ChildProcess
/**
 * Restart times, for spotting a crash loop.
 *
 * A lifetime counter is the wrong shape: a server restarted once a day is healthy and
 * would eventually hit any fixed total, stopping for no reason. What matters is whether
 * it is failing *repeatedly and quickly*, so only restarts inside a short window count.
 */
const recentRestarts: number[] = []
const LOOP_WINDOW_MS = 60_000
const LOOP_LIMIT = 5

function startControl(): void {
  control = spawn(process.execPath, ['--env-file-if-exists=.env', 'packages/control/src/index.ts'], {
    stdio: ['ignore', 'inherit', 'inherit'],
    env: { ...process.env, PUBLIC_ORIGIN: publicOrigin, PORT: String(PORT), BIND_HOST: '127.0.0.1' },
  })
  children.push(control)

  control.on('exit', code => {
    if (stopping) return

    const now = Date.now()
    recentRestarts.push(now)
    while (recentRestarts.length > 0 && now - recentRestarts[0]! > LOOP_WINDOW_MS) recentRestarts.shift()

    if (recentRestarts.length > LOOP_LIMIT) {
      console.error(
        `\n  The server has exited ${recentRestarts.length} times in the last minute.\n` +
        `  That is a crash loop, not a restart — stopping so the cause is visible.\n`)
      shutdown(code ?? 1)
    }

    console.log(`\n  Server exited (code ${code}); restarting — the address stays the same.\n`)
    setTimeout(startControl, 1000)
  })
}
startControl()

// Wait for the whole path — DNS, Cloudflare edge, tunnel, service — to actually work.
// A quick tunnel's hostname routinely takes 30-60s to become resolvable, so this is
// generous on purpose and says what it is doing rather than appearing hung.
process.stdout.write('  Waiting for Cloudflare to route the address')
let live = false
let localDnsBroken = false
let lastError = 'no response'
for (let i = 0; i < 90; i++) {
  await new Promise(r => setTimeout(r, 2000))
  const probe = await probeUrl(`${publicOrigin}/health`, 5000)
  if (probe.ok) {
    live = true
    localDnsBroken = !probe.localDnsWorks
    break
  }
  lastError = probe.error ?? `HTTP ${probe.status}`
  if (!probe.localDnsWorks) localDnsBroken = true
  if (i % 5 === 4) process.stdout.write('.')
}
process.stdout.write('\n')

if (!live) {
  console.error(`
  The address never became reachable.  (${lastError})

  The tunnel log is in ${TUNNEL_LOG} — check it for a reason.
  This is usually a network that blocks outbound QUIC/UDP 7844. cloudflared falls back
  to HTTP/2 on its own, but a restrictive network can stop both.
`)
  shutdown(1)
}

if (localDnsBroken) {
  console.log(`
  Note: this machine's own DNS cannot resolve ${new URL(publicOrigin).hostname},
  but a public resolver can, and the address is serving correctly.

  That means the address works for everyone else and only looks broken from here.
  Opening the link in your own browser may fail while friends connect fine.
`)
}

// ---------------------------------------------------------------------- join

// This script talks to the server over loopback, never through the tunnel. Creating an
// invite has no reason to leave the machine, and routing it through Cloudflare makes it
// fail whenever local DNS cannot resolve the public hostname — which is exactly the
// situation on this machine.
const LOCAL_ORIGIN = `http://127.0.0.1:${PORT}`

async function login(): Promise<string> {
  const res = await fetch(`${LOCAL_ORIGIN}/auth/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email: env.BOOTSTRAP_EMAIL, password: env.BOOTSTRAP_PASSWORD }),
  })
  if (!res.ok) throw new Error(`login failed: ${res.status}`)
  return (res.headers.getSetCookie?.()[0] ?? '').split(';')[0]!
}

async function newCode(label: string): Promise<string> {
  const cookie = await login()
  const res = await fetch(`${LOCAL_ORIGIN}/hosts/pair-code`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', cookie },
    body: JSON.stringify({ label }),
  })
  const { code } = await res.json() as { code: string }
  return code
}

const first = await newCode('friend-1').catch((err: Error) => {
  console.error(`
  The server is running, but this script could not sign in to create an invite.
    ${err.message}

  Check BOOTSTRAP_EMAIL and BOOTSTRAP_PASSWORD in .env match the account you expect.
`)
  shutdown(1)
})

console.log(`
  ${'-'.repeat(68)}

  Your network is live at

      ${publicOrigin}

  Send a friend this link. It shows them exactly what to run:

      ${publicOrigin}/join?code=${first}

  The code expires in 10 minutes and works once. Press Enter here for another.
  Your own dashboard login is in .env — that is not for them.

  ${manualUrl
    ? 'This address is yours, so it survives restarts.'
    : 'Note: this address changes every restart. For something permanent see docs/03-connectivity.md,\n  then run:  node scripts/share.ts --url https://your-stable-address'}

  Ctrl-C stops the tunnel and the server.
  ${'-'.repeat(68)}
`)

// A fresh code per friend, on demand, without digging through curl commands.
const rl = createInterface({ input: process.stdin, output: process.stdout })
let n = 1
rl.on('line', () => {
  n += 1
  void newCode(`friend-${n}`)
    .then(code => console.log(`\n  Invite ${n}:  ${publicOrigin}/join?code=${code}\n`))
    .catch((err: Error) => console.error(`  could not create a code: ${err.message}`))
})
