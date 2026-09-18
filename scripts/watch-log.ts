/**
 * Live view of what the system is doing, for watching a real connection come in.
 *
 *   node scripts/watch-log.ts                      everything, as it happens
 *   node scripts/watch-log.ts --host <id|label>    one host only
 *   node scripts/watch-log.ts --grep connect       events matching a pattern
 *   node scripts/watch-log.ts --level warn         warnings and errors only
 *   node scripts/watch-log.ts --since 30m --no-follow    recent history, then exit
 *   node scripts/watch-log.ts --story              group by host and narrate each join
 *
 * Reads the JSONL files the control service and agents write. Works after the fact too:
 * when a friend says "it didn't work an hour ago", --since finds it.
 */
import { readdirSync, readFileSync, statSync, watch } from 'node:fs'
import { join } from 'node:path'

const args = process.argv.slice(2)
const flag = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 ? args[i + 1] : undefined
}
const has = (name: string): boolean => args.includes(`--${name}`)

const DIR = flag('dir') ?? process.env.DWP_LOG_DIR ?? '.dwp/logs'
const hostFilter = flag('host')?.toLowerCase()
const grep = flag('grep') ? new RegExp(flag('grep')!, 'i') : null
const minLevel = (flag('level') ?? 'debug') as 'debug' | 'info' | 'warn' | 'error'
const follow = !has('no-follow')
const story = has('story')

const ORDER = { debug: 10, info: 20, warn: 30, error: 40 }
const COLOR = { debug: '\x1b[90m', info: '\x1b[36m', warn: '\x1b[33m', error: '\x1b[31m' }
const RESET = '\x1b[0m'
const DIM = '\x1b[2m'
const BOLD = '\x1b[1m'

function parseSince(v: string | undefined): number {
  if (!v) return 0
  const m = /^(\d+)([smhd])$/.exec(v)
  if (!m) return 0
  const mult = { s: 1e3, m: 60e3, h: 3600e3, d: 86400e3 }[m[2]!]!
  return Date.now() - Number(m[1]) * mult
}
const sinceMs = parseSince(flag('since'))

type Rec = Record<string, unknown> & { ts: string; lvl: keyof typeof ORDER; comp: string; evt: string }

function keep(r: Rec): boolean {
  if (ORDER[r.lvl] < ORDER[minLevel]) return false
  if (sinceMs && Date.parse(r.ts) < sinceMs) return false
  if (hostFilter) {
    const hay = `${r.hostId ?? ''} ${r.label ?? ''}`.toLowerCase()
    if (!hay.includes(hostFilter)) return false
  }
  if (grep && !grep.test(JSON.stringify(r))) return false
  return true
}

/** Plain-language gloss for the events that matter during a join. */
const EXPLAIN: Record<string, string> = {
  'control.started': 'server is up',
  'agent.connected': 'a computer joined',
  'agent.disconnected': 'a computer dropped off',
  'agent.auth_rejected': 'someone was refused — see reason',
  'agent.upgrade_denied': 'a connection was refused before the handshake',
  'agent.heartbeat_timeout': 'went silent; treating as offline',
  'connect.attempt': 'dialling the server',
  'connect.established': 'socket open',
  'connect.lost': 'socket closed',
  'connect.retry_scheduled': 'will retry',
  'connect.error': 'dial failed',
  'handshake.complete': 'registered with the server',
  'clock.skewed': 'this machine disagrees with the server about the time',
  'host.revoked': 'access was withdrawn',
}

function render(r: Rec): string {
  const time = r.ts.slice(11, 23)
  const known = new Set(['ts', 'lvl', 'comp', 'runId', 'evt'])
  const extras = Object.entries(r)
    .filter(([k]) => !known.has(k))
    .map(([k, v]) => `${DIM}${k}=${RESET}${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
    .join(' ')
  const why = EXPLAIN[r.evt] ? `  ${DIM}// ${EXPLAIN[r.evt]}${RESET}` : ''
  return `${DIM}${time}${RESET} ${COLOR[r.lvl]}${r.lvl.toUpperCase().padEnd(5)}${RESET}` +
    `${BOLD}${r.comp}/${r.evt}${RESET}  ${extras}${why}`
}

function readAll(): Rec[] {
  let files: string[]
  try {
    files = readdirSync(DIR).filter(f => f.endsWith('.jsonl'))
  } catch {
    console.error(`No log directory at ${DIR}. Start the control service or an agent first.`)
    process.exit(1)
  }
  const out: Rec[] = []
  for (const f of files) {
    for (const line of readFileSync(join(DIR, f), 'utf8').split('\n')) {
      if (!line.trim()) continue
      try { out.push(JSON.parse(line) as Rec) } catch {}
    }
  }
  return out.sort((a, b) => a.ts.localeCompare(b.ts))
}

// ------------------------------------------------------------------ story mode

if (story) {
  const records = readAll().filter(r => !sinceMs || Date.parse(r.ts) >= sinceMs)
  const hosts = new Map<string, Rec[]>()
  for (const r of records) {
    const id = String(r.hostId ?? '')
    if (!id) continue
    if (!hosts.has(id)) hosts.set(id, [])
    hosts.get(id)!.push(r)
  }
  if (hosts.size === 0) {
    console.log('\nNo host activity in the log yet.\n')
    process.exit(0)
  }
  console.log(`\n${BOLD}Connection stories${RESET}  ${DIM}(${hosts.size} host${hosts.size === 1 ? '' : 's'})${RESET}\n`)
  for (const [id, events] of hosts) {
    const label = events.find(e => e.label)?.label ?? id.slice(0, 8)
    const connects = events.filter(e => e.evt === 'agent.connected').length
    const drops = events.filter(e => e.evt === 'agent.disconnected').length
    const rejects = events.filter(e => e.evt === 'agent.auth_rejected')
    const timeouts = events.filter(e => e.evt === 'agent.heartbeat_timeout').length

    console.log(`  ${BOLD}${label}${RESET} ${DIM}${id}${RESET}`)
    console.log(`    joined ${connects}x · dropped ${drops}x · silent-timeouts ${timeouts}` +
      (rejects.length ? ` · ${COLOR.warn}refused ${rejects.length}x${RESET}` : ''))
    if (rejects.length) {
      const reasons = [...new Set(rejects.map(r => String(r.reason)))]
      console.log(`    ${COLOR.warn}refusal reasons:${RESET} ${reasons.join(', ')}`)
    }
    if (connects > 1 && drops >= connects - 1) {
      console.log(`    ${COLOR.warn}unstable link — reconnecting repeatedly${RESET}`)
    }
    const last = events.at(-1)!
    console.log(`    ${DIM}last: ${last.evt} at ${last.ts.slice(11, 19)}${RESET}\n`)
  }
  process.exit(0)
}

// ------------------------------------------------------------------ live mode

const seen = new Map<string, number>()
for (const r of readAll()) if (keep(r)) console.log(render(r))

if (!follow) process.exit(0)

try {
  for (const f of readdirSync(DIR).filter(n => n.endsWith('.jsonl'))) {
    seen.set(f, statSync(join(DIR, f)).size)
  }
} catch {}

console.log(`${DIM}\n— watching ${DIR} — Ctrl-C to stop —${RESET}\n`)

// Poll rather than rely on fs.watch semantics, which differ across platforms and miss
// appends to files that did not exist when the watch started.
setInterval(() => {
  let files: string[] = []
  try { files = readdirSync(DIR).filter(n => n.endsWith('.jsonl')) } catch { return }
  for (const f of files) {
    const path = join(DIR, f)
    let size = 0
    try { size = statSync(path).size } catch { continue }
    const from = seen.get(f) ?? 0
    if (size <= from) { seen.set(f, size); continue }
    let chunk = ''
    try { chunk = readFileSync(path, 'utf8').slice(from) } catch { continue }
    seen.set(f, size)
    for (const line of chunk.split('\n')) {
      if (!line.trim()) continue
      try {
        const r = JSON.parse(line) as Rec
        if (keep(r)) console.log(render(r))
      } catch {}
    }
  }
}, 400)
