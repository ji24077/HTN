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

const DIM = '\x1b[2m', RESET = '\x1b[0m'
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
  { name: 'r.jina.ai', url: (t: string) => `https://r.jina.ai/${t}` },
  { name: 'api.codetabs.com', url: (t: string) => `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(t)}` },
  { name: 'api.allorigins.win', url: (t: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(t)}` },
]

/**
 * Check the messenger before trusting the message.
 *
 * These are free public services and they break. An earlier version of this gate
 * reported that no outside machine could reach the tunnel; in fact two of the three
 * fetchers were returning errors for example.com as well. A test that cannot tell
 * "the thing under test is broken" from "my instrument is broken" is worse than no test.
 */
async function fetcherWorks(f: (typeof FETCHERS)[number]): Promise<boolean> {
  try {
    const res = await fetch(f.url('https://example.com'), { signal: AbortSignal.timeout(25_000) })
    if (!res.ok) return false
    return /example/i.test(await res.text())
  } catch {
    return false
  }
}

let externalOk = false
let externalDetail = 'no third-party fetcher was working, so this could not be checked'
let usable = 0

for (const f of FETCHERS) {
  if (!(await fetcherWorks(f))) {
    console.log(`  ${DIM}skip  ${f.name} is not working today (failed its own control test)${RESET}`)
    continue
  }
  usable += 1
  try {
    const res = await fetch(f.url(`${ORIGIN}/whoami`), { signal: AbortSignal.timeout(25_000) })
    const text = await res.text()
    const seen = /"observedIp"\s*:\s*"([^"]+)"/.exec(text)?.[1]
    if (!seen) { externalDetail = `${f.name}: reached, but the reply was not recognisable`; continue }

    if (seen === myEgress) {
      externalDetail = `${f.name} arrived as ${seen}, the same address as this machine — not proof of an outside path`
      continue
    }
    externalOk = true
    externalDetail = `${f.name} fetched it from ${seen}, a different network from this machine (${myEgress})`
    break
  } catch (err) {
    externalDetail = `${f.name}: ${err instanceof Error ? err.message : String(err)}`
  }
}

if (usable === 0) {
  console.log(`  ${DIM}note  every public fetcher was down; skipping the outside-reachability check${RESET}`)
}

if (usable === 0) {
  console.log(`  ${DIM}----${RESET} a computer on another network can reach this machine  ${DIM}${externalDetail}${RESET}`)
} else {
  check('a computer on another network can reach this machine', externalOk, externalDetail)
}

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
    const { hostId, wsUrl: serverSuppliedWsUrl } = await pairRes.json() as { hostId: string; wsUrl: string }

    // Dial the address under test, not whatever the server believes its own address to
    // be. An earlier version trusted the server's reply and happily reported "handshake
    // completed through the public URL" while actually connecting to ws://localhost —
    // a green check for a claim it never tested.
    const wsUrl = `${ORIGIN.replace(/^http/, 'ws')}/agent/connect`
    if (serverSuppliedWsUrl !== wsUrl) {
      console.log(`  ${DIM}note  the server reports its address as ${serverSuppliedWsUrl};` +
        ` testing ${wsUrl} instead${RESET}`)
    }

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
        if (msg.type === 'hello.ack') { ws.close(); resolve(`handshake completed over ${wsUrl}`) }
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
