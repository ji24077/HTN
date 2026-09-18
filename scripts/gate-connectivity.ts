/**
 * Gate `connectivity` — can a computer elsewhere on the internet actually reach this one?
 *
 *   node scripts/gate-connectivity.ts <publicOrigin>
 *
 * The important check is #3. Everything else can pass on a machine that is talking to
 * itself; #3 asks an unrelated third-party server to fetch the URL and reports the
 * address that arrived. If that address is NOT this machine's egress, then a computer
 * on a different network genuinely reached this laptop through the tunnel.
 */
import WebSocket from 'ws'
import { generateKeyPairSync, createPublicKey } from 'node:crypto'
import { mintAssertion, envelope } from '@dwp/protocol'

const ORIGIN = process.argv[2] ?? process.env.PUBLIC_ORIGIN
if (!ORIGIN || !ORIGIN.startsWith('https://')) {
  console.error('usage: node scripts/gate-connectivity.ts https://<public-host>')
  process.exit(1)
}
const EMAIL = process.env.BOOTSTRAP_EMAIL ?? 'operator@local'
const PASSWORD = process.env.BOOTSTRAP_PASSWORD ?? ''

let passed = 0
let failed = 0
const check = (name: string, ok: boolean, detail = ''): void => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `\n          ${detail}` : ''}`)
  ok ? passed++ : failed++
}

console.log(`\ngate:connectivity against ${ORIGIN}\n`)

// 1 ------------------------------------------------- reachable over the internet
let health: { ok?: boolean; publicOrigin?: string } = {}
try {
  const res = await fetch(`${ORIGIN}/health`, { signal: AbortSignal.timeout(15_000) })
  health = await res.json() as typeof health
  check('the public URL serves the control service', res.ok && health.ok === true, `HTTP ${res.status}`)
} catch (err) {
  check('the public URL serves the control service', false, String(err))
}

// 2 ----------------------------------------- this machine's own public address
let myEgress = 'unknown'
try {
  myEgress = (await (await fetch('https://api.ipify.org', { signal: AbortSignal.timeout(10_000) })).text()).trim()
} catch {}
console.log(`\n  This machine's public address: ${myEgress}\n`)

// 3 --------------------------- a genuinely external client reaches this machine
//
// Third-party fetchers run on their own infrastructure, on networks unrelated to this
// one. If one of them can load the URL, the path from the open internet to this laptop
// is real — not an artefact of the request starting and ending here.
const FETCHERS = [
  { name: 'api.codetabs.com', url: (t: string) => `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(t)}` },
  { name: 'api.allorigins.win', url: (t: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(t)}` },
]

let externalOk = false
let externalDetail = 'no third-party fetcher succeeded'
for (const f of FETCHERS) {
  try {
    const res = await fetch(f.url(`${ORIGIN}/whoami`), { signal: AbortSignal.timeout(25_000) })
    if (!res.ok) { externalDetail = `${f.name}: HTTP ${res.status}`; continue }
    const seen = JSON.parse(await res.text()) as { observedIp?: string }
    if (!seen.observedIp) { externalDetail = `${f.name}: unexpected body`; continue }
    externalOk = true
    externalDetail = seen.observedIp === myEgress
      ? `${f.name} reached it, but arrived as ${seen.observedIp} — the same address as this machine, so this is not proof of an outside path`
      : `${f.name} fetched it from ${seen.observedIp}, a different network from this machine (${myEgress})`
    if (seen.observedIp === myEgress) externalOk = false
    break
  } catch (err) {
    externalDetail = `${f.name}: ${err instanceof Error ? err.message : String(err)}`
  }
}
check('a computer on another network can reach this machine', externalOk, externalDetail)

// 4 ------------------------------------------- an agent can connect over WSS
let wsReason = 'no result'
let wsOk = false
if (PASSWORD) {
  try {
    const loginRes = await fetch(`${ORIGIN}/auth/login`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: EMAIL, password: PASSWORD }),
    })
    const cookie = (loginRes.headers.getSetCookie?.()[0] ?? '').split(';')[0]!
    const codeRes = await fetch(`${ORIGIN}/hosts/pair-code`, {
      method: 'POST', headers: { 'content-type': 'application/json', cookie },
      body: JSON.stringify({ label: 'connectivity-gate' }),
    })
    const { code } = await codeRes.json() as { code: string }

    const { privateKey } = generateKeyPairSync('ed25519')
    const publicKey = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')
    const pairRes = await fetch(`${ORIGIN}/hosts/pair`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code, publicKey, label: 'connectivity-gate' }),
    })
    const { hostId, wsUrl } = await pairRes.json() as { hostId: string; wsUrl: string }

    wsReason = await new Promise<string>(resolve => {
      const ws = new WebSocket(wsUrl, { headers: { authorization: `Bearer ${mintAssertion(hostId, privateKey)}` } })
      ws.on('open', () => {
        ws.send(JSON.stringify(envelope('hello', {
          capability: { agentVersion: 'gate', os: 'linux', arch: 'x64', cpuModel: 'gate',
                        logicalCores: 1, totalRamMb: 1024, freeRamMb: 512, adapters: ['echo'] },
          consent: { paused: false, allowCompute: true, allowBrowser: false, maxConcurrency: 1 },
        })))
      })
      ws.on('message', (raw: Buffer) => {
        const msg = JSON.parse(raw.toString('utf8')) as { type: string }
        if (msg.type === 'hello.ack') { ws.close(); resolve(`handshake completed over ${new URL(wsUrl).protocol}`) }
      })
      ws.on('unexpected-response', (_r, res) => resolve(`rejected: HTTP ${res.statusCode}`))
      ws.on('error', (e: Error) => resolve(`error: ${e.message}`))
      setTimeout(() => { try { ws.close() } catch {} ; resolve('timeout') }, 20_000)
    })
    wsOk = wsReason.startsWith('handshake completed')
  } catch (err) {
    wsReason = err instanceof Error ? err.message : String(err)
  }
} else {
  wsReason = 'set BOOTSTRAP_PASSWORD to run this check'
}
check('an agent completes the WSS handshake through the public URL', wsOk, wsReason)

// 5 --------------------------------------------------- brute force is limited
let limited = false
try {
  let sawLimit = false
  for (let i = 0; i < 14; i++) {
    const res = await fetch(`${ORIGIN}/auth/login`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: EMAIL, password: `wrong-${i}` }),
      signal: AbortSignal.timeout(10_000),
    })
    if (res.status === 429) { sawLimit = true; break }
  }
  limited = sawLimit
} catch {}
check('repeated failed logins are rate limited', limited, limited ? 'HTTP 429 before 14 attempts' : 'no 429 seen')

// 6 ------------------------------------------------------- the join page loads
try {
  const res = await fetch(`${ORIGIN}/join`, { signal: AbortSignal.timeout(15_000) })
  const body = await res.text()
  check('the join page loads for an invited friend', res.ok && body.includes('pnpm agent pair'), `HTTP ${res.status}`)
} catch (err) {
  check('the join page loads for an invited friend', false, String(err))
}

console.log(`\n  gate:connectivity ${failed === 0 ? 'PASS' : 'FAIL'}  (${passed} passed, ${failed} failed)\n`)
process.exit(failed === 0 ? 0 : 1)
