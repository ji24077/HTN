/**
 * Check everything needed to host a network, and say exactly what to fix.
 *
 *   node scripts/doctor.ts
 *
 * Written to be run when something is wrong and it is not obvious what. Each check
 * reports a fix, not just a status.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { lookup } from 'node:dns/promises'
import { probeUrl } from './lib/net-probe.ts'

const GREEN = '\x1b[32m', RED = '\x1b[31m', YEL = '\x1b[33m', DIM = '\x1b[2m', BOLD = '\x1b[1m', RESET = '\x1b[0m'
const PORT = Number(process.env.PORT ?? 8787)

let problems = 0
const ok = (name: string, detail = ''): void => console.log(`  ${GREEN}ok${RESET}    ${name}${detail ? `  ${DIM}${detail}${RESET}` : ''}`)
const warn = (name: string, fix: string): void => console.log(`  ${YEL}warn${RESET}  ${name}\n        ${DIM}${fix}${RESET}`)
const bad = (name: string, fix: string): void => { problems += 1; console.log(`  ${RED}FAIL${RESET}  ${name}\n        ${fix}`) }

const has = (cmd: string, args: string[] = ['--version']): string | null => {
  try { return execFileSync(cmd, args, { stdio: 'pipe' }).toString().trim().split('\n')[0]! } catch { return null }
}

console.log(`\n${BOLD}Checking this machine${RESET}\n`)

// ---------------------------------------------------------------- toolchain
const major = Number(process.versions.node.split('.')[0])
major >= 24
  ? ok('Node.js', process.version)
  : bad(`Node.js ${process.version} is too old`, 'Install Node 24 or newer: https://nodejs.org')

const pnpm = has('pnpm')
pnpm ? ok('pnpm', pnpm) : bad('pnpm is not installed', 'npm i -g pnpm')

const cf = has('cloudflared', ['--version'])
cf ? ok('cloudflared', cf) : warn('cloudflared is not installed',
  'Needed only for `pnpm share`. Install: brew install cloudflared — or bring your own address with --url')

// ----------------------------------------------------------------- database
const docker = has('docker')
if (!docker) {
  bad('Docker is not available', 'Install Docker Desktop, or point DATABASE_URL at any PostgreSQL 16+')
} else {
  ok('Docker', docker)
  try {
    const status = execFileSync('docker', ['ps', '--filter', 'name=dwp-db', '--format', '{{.Status}}'], { stdio: 'pipe' }).toString().trim()
    if (status.startsWith('Up')) ok('database container', status)
    else bad('the database is not running', 'pnpm db:up')
  } catch {
    bad('could not query Docker', 'Is Docker Desktop running?')
  }
}

// --------------------------------------------------------------- credentials
if (!existsSync('.env')) {
  warn('no .env file yet', 'cp .env.example .env — or just run `pnpm share`, which creates one')
} else {
  const env = readFileSync('.env', 'utf8')
  const password = /^BOOTSTRAP_PASSWORD=(.*)$/m.exec(env)?.[1] ?? ''
  const weak = ['', 'change-me', 'dwp-dev', 'password', 'admin'].includes(password) || password.length < 12
  weak
    ? warn('the operator password is a placeholder',
        '`pnpm share` will generate a strong one automatically before going public')
    : ok('operator password', 'set and strong enough')
}

// ---------------------------------------------------------------------- port
const portFree = await new Promise<boolean>(resolve => {
  const probe = createServer()
  probe.once('error', () => resolve(false))
  probe.once('listening', () => probe.close(() => resolve(true)))
  probe.listen(PORT, '127.0.0.1')
})

type Health = { publicOrigin?: string }
let running: Health | null = null
if (portFree) {
  ok(`port ${PORT} is free`, 'no server running yet')
} else {
  try {
    running = await (await fetch(`http://127.0.0.1:${PORT}/health`, { signal: AbortSignal.timeout(3000) })).json() as Health
    ok(`a server is already running on ${PORT}`, running?.publicOrigin ?? '')
  } catch {
    bad(`port ${PORT} is taken by something else`,
      `lsof -nP -iTCP:${PORT} -sTCP:LISTEN   — or use PORT=8788 pnpm share`)
  }
}

// ------------------------------------------------------------------- network
console.log(`\n${BOLD}Checking the network${RESET}\n`)

try {
  const publicIp = (await (await fetch('https://api.ipify.org', { signal: AbortSignal.timeout(8000) })).text()).trim()
  ok('internet access', `this machine appears as ${publicIp}`)

  // A private address on the interface plus a different public address means NAT, and a
  // carrier-grade range means port forwarding can never work here.
  const localAddrs = Object.values(await import('node:os').then(m => m.networkInterfaces()))
    .flat().filter(Boolean)
    .filter(i => i!.family === 'IPv4' && !i!.internal).map(i => i!.address)
  const cgnat = localAddrs.find(a => /^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\./.test(a))
  const hotspot = localAddrs.find(a => /^172\.20\.10\./.test(a))
  if (cgnat) {
    console.log(`        ${DIM}${cgnat} is carrier-grade NAT — port forwarding is impossible here, a tunnel is required${RESET}`)
  } else if (hotspot) {
    console.log(`        ${DIM}${hotspot} looks like a phone hotspot — port forwarding is not possible, a tunnel is required${RESET}`)
  }
} catch {
  bad('no internet access', 'Check this machine is online.')
}

// DNS for freshly created names is the failure that makes a working address look broken.
try {
  await lookup('example.com')
  ok('DNS', 'this machine can resolve names')
} catch {
  bad('DNS is not working on this machine', 'Try setting DNS to 1.1.1.1')
}

// --------------------------------------------------- the live address, if any
if (running?.publicOrigin && !running.publicOrigin.includes('localhost')) {
  const probe = await probeUrl(`${running.publicOrigin}/health`, 10_000)
  if (probe.ok && probe.via === 'system') {
    ok('your public address', `${running.publicOrigin} is reachable`)
  } else if (probe.ok) {
    warn(`${running.publicOrigin} works, but this machine cannot resolve it`,
      'It is fine for everyone else. Agents here need DWP_DNS_FALLBACK=1; your browser may fail to open it.')
  } else {
    bad(`your public address is not answering`, `${probe.error ?? `HTTP ${probe.status}`}\n        Restart with: pnpm share`)
  }
}

console.log(`\n  ${problems === 0 ? GREEN + 'Ready to host.' + RESET : RED + `${problems} problem(s) to fix first.` + RESET}\n`)
process.exit(problems === 0 ? 0 : 1)
