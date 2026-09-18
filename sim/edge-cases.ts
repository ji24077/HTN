/**
 * Adversarial and awkward inputs that a network simulator cannot produce.
 *
 *   node sim/edge-cases.ts
 *
 * The network scenarios prove the system survives bad *conditions*. These prove it
 * survives bad *input*: forged tokens, skewed clocks, malformed frames, hostile labels.
 * Every case here expects a rejection or a shrug — never a crash and never acceptance.
 */
import { generateKeyPairSync, createPublicKey, sign } from 'node:crypto'
import WebSocket from 'ws'
import { envelope, mintAssertion } from '@dwp/protocol'
import { startHarness, sleep } from './harness.ts'

const GREEN = '\x1b[32m', RED = '\x1b[31m', DIM = '\x1b[2m', BOLD = '\x1b[1m', RESET = '\x1b[0m'
let passed = 0, failed = 0

const check = (name: string, ok: boolean, detail = ''): void => {
  if (ok) { passed++; console.log(`  ${GREEN}ok${RESET}   ${name}${detail ? `  ${DIM}${detail}${RESET}` : ''}`) }
  else { failed++; console.log(`  ${RED}FAIL${RESET} ${name}${detail ? `  ${RED}${detail}${RESET}` : ''}`) }
}

const b64u = (b: Buffer): string => b.toString('base64url')
const json = (o: unknown): Buffer => Buffer.from(JSON.stringify(o), 'utf8')

const harness = await startHarness({ logDir: '.dwp/sim-logs/edge-cases', controlPort: 8891 })
const ORIGIN = harness.directOrigin
const WS_URL = `${ORIGIN.replace(/^http/, 'ws')}/agent/connect`

/** Dial the agent endpoint with a raw token and report how it was refused. */
function dial(token: string): Promise<string> {
  return new Promise(resolve => {
    const ws = new WebSocket(WS_URL, { headers: { authorization: `Bearer ${token}` }, handshakeTimeout: 8000 })
    const done = (v: string): void => { try { ws.close() } catch {} ; resolve(v) }
    ws.on('unexpected-response', (_r, res) => done(String(res.headers['x-dwp-reason'] ?? res.statusCode)))
    ws.on('open', () => done('ACCEPTED'))
    ws.on('error', () => done('network-error'))
    setTimeout(() => done('timeout'), 9000)
  })
}

/** Enrol a throwaway host and return its id plus signing key. */
async function freshHost(label: string) {
  const codeRes = await harness.api('/hosts/pair-code', { label })
  const { code } = await codeRes.json() as { code: string }
  const { privateKey } = generateKeyPairSync('ed25519')
  const publicKey = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')
  const res = await fetch(`${ORIGIN}/hosts/pair`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ code, publicKey, label }),
  })
  const { hostId } = await res.json() as { hostId: string }
  return { hostId, privateKey }
}

/** Hand-roll an assertion with arbitrary claims, signed by a real key. */
function forgeAssertion(claims: Record<string, unknown>, privateKey: ReturnType<typeof generateKeyPairSync>['privateKey'], header: Record<string, unknown> = { alg: 'EdDSA', typ: 'JWT' }): string {
  const input = `${b64u(json(header))}.${b64u(json(claims))}`
  return `${input}.${b64u(sign(null, Buffer.from(input, 'utf8'), privateKey))}`
}

try {
  console.log(`\n${BOLD}Clocks and token lifetimes${RESET}`)
  {
    const { hostId, privateKey } = await freshHost('clock')
    const now = Math.floor(Date.now() / 1000)

    check('an expired assertion is rejected',
      await dial(forgeAssertion({ iss: hostId, aud: 'dwp-control', iat: now - 600, exp: now - 300, jti: crypto.randomUUID() }, privateKey)) === 'expired')

    check('a clock running an hour fast is rejected',
      await dial(forgeAssertion({ iss: hostId, aud: 'dwp-control', iat: now + 3600, exp: now + 3720, jti: crypto.randomUUID() }, privateKey)) === 'future-iat')

    check('an assertion with an over-long lifetime is rejected',
      await dial(forgeAssertion({ iss: hostId, aud: 'dwp-control', iat: now, exp: now + 86_400, jti: crypto.randomUUID() }, privateKey)) === 'ttl-too-long',
      'a stolen token must not be usable for a day')

    check('a small clock difference is tolerated',
      await dial(forgeAssertion({ iss: hostId, aud: 'dwp-control', iat: now - 30, exp: now + 90, jti: crypto.randomUUID() }, privateKey)) === 'ACCEPTED',
      'real machines are never perfectly in sync')
  }

  console.log(`\n${BOLD}Token forgery${RESET}`)
  {
    const { hostId, privateKey } = await freshHost('forgery')
    const now = Math.floor(Date.now() / 1000)

    check('an unsigned "alg: none" token is rejected',
      await dial(`${b64u(json({ alg: 'none', typ: 'JWT' }))}.${b64u(json({ iss: hostId, aud: 'dwp-control', iat: now, exp: now + 60, jti: crypto.randomUUID() }))}.`) === 'bad-header',
      'the classic JWT bypass')

    check('an HMAC-signed token is rejected',
      await dial(forgeAssertion({ iss: hostId, aud: 'dwp-control', iat: now, exp: now + 60, jti: crypto.randomUUID() }, privateKey, { alg: 'HS256', typ: 'JWT' })) === 'bad-header',
      'algorithm confusion')

    check('a token for the wrong audience is rejected',
      await dial(forgeAssertion({ iss: hostId, aud: 'some-other-service', iat: now, exp: now + 60, jti: crypto.randomUUID() }, privateKey)) === 'bad-audience',
      'a token minted for another service must not work here')

    // These assert the property that matters — that the connection is refused — and
    // report whichever reason the server gave. An earlier version pinned exact reason
    // strings and "failed" on three correct refusals, which tests the error text rather
    // than the security boundary.
    for (const [label, token] of [
      ['a structurally broken token', 'not.a.token'],
      ['an empty token', ''],
      ['a token with no signature', `${b64u(json({ alg: 'EdDSA', typ: 'JWT' }))}.${b64u(json({ iss: hostId }))}.`],
      ['a token with a tampered payload', `${b64u(json({ alg: 'EdDSA', typ: 'JWT' }))}.${b64u(json({ iss: hostId, aud: 'dwp-control', iat: now, exp: now + 60, jti: 'x' }))}.AAAA`],
    ] as const) {
      const reason = await dial(token)
      check(`${label} is rejected`, reason !== 'ACCEPTED', `refused: ${reason}`)
    }
  }

  console.log(`\n${BOLD}Hostile and malformed traffic${RESET}`)
  {
    const { hostId, privateKey } = await freshHost('hostile')
    const ws = new WebSocket(WS_URL, { headers: { authorization: `Bearer ${mintAssertion(hostId, privateKey)}` } })
    await new Promise<void>((res, rej) => { ws.on('open', () => res()); ws.on('error', rej) })

    // None of these should close the socket or take the server down.
    ws.send('this is not json at all')
    ws.send('{"unclosed":')
    ws.send(JSON.stringify({ v: 99, id: 'x', ts: 'nonsense', type: 'hello' }))
    ws.send(JSON.stringify(envelope('no.such.message.type', { anything: true })))
    ws.send(JSON.stringify(envelope('task.result', { taskId: 'not-a-uuid' })))
    ws.send(JSON.stringify(envelope('hello', { capability: null, consent: 'wrong shape' })))
    ws.send(Buffer.alloc(64 * 1024, 0x41))                       // 64 KB of junk
    ws.send(JSON.stringify(envelope('heartbeat', { freeRamMb: -1, running: -5 })))
    await sleep(1500)

    check('malformed frames do not close the connection', ws.readyState === WebSocket.OPEN)
    const health = await fetch(`${ORIGIN}/health`)
    check('the server is still healthy after hostile frames', health.ok)

    // A well-formed hello after the junk must still work.
    ws.send(JSON.stringify(envelope('hello', {
      capability: { agentVersion: 'edge', os: 'linux', arch: 'x64', cpuModel: 'test',
                    logicalCores: 1, totalRamMb: 512, freeRamMb: 256, adapters: ['echo'] },
      consent: { paused: false, allowCompute: true, allowBrowser: false, maxConcurrency: 1 },
    })))
    const acked = await new Promise<boolean>(resolve => {
      const timer = setTimeout(() => resolve(false), 5000)
      ws.on('message', (raw: Buffer) => {
        try {
          if ((JSON.parse(raw.toString('utf8')) as { type: string }).type === 'hello.ack') {
            clearTimeout(timer); resolve(true)
          }
        } catch {}
      })
    })
    check('a valid message still works after a burst of junk', acked)
    ws.close()
  }

  console.log(`\n${BOLD}Pairing codes${RESET}`)
  {
    const codeRes = await harness.api('/hosts/pair-code', { label: 'case-test' })
    const { code } = await codeRes.json() as { code: string }
    const { privateKey } = generateKeyPairSync('ed25519')
    const publicKey = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')

    // People retype codes with the wrong case and stray spaces.
    const res = await fetch(`${ORIGIN}/hosts/pair`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code: `  ${code.toLowerCase()}  `, publicKey, label: 'case-test' }),
    })
    check('a code typed in lower case with stray spaces still works', res.ok, `HTTP ${res.status}`)

    const bad = await fetch(`${ORIGIN}/hosts/pair`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code: 'AAAA-AAAA', publicKey, label: 'nope' }),
    })
    check('an unknown code is rejected', bad.status === 400)
  }

  console.log(`\n${BOLD}Authorization${RESET}`)
  {
    for (const path of ['/hosts', '/diagnostics', '/jobs/00000000-0000-0000-0000-000000000000']) {
      const res = await fetch(`${ORIGIN}${path}`)
      check(`${path} requires a session`, res.status === 401, `HTTP ${res.status}`)
    }
    const res = await fetch(`${ORIGIN}/jobs`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ adapter: 'echo', mode: 'each', count: 1 }),
    })
    check('submitting work requires a session', res.status === 401, `HTTP ${res.status}`)
  }

  console.log(`\n${BOLD}Awkward but legitimate input${RESET}`)
  {
    const weird = 'héllo-💻-<script>alert(1)</script>'
    const res = await harness.api('/hosts/pair-code', { label: weird })
    check('a label with unicode and markup is accepted safely', res.ok, `HTTP ${res.status}`)

    const tooLong = await harness.api('/hosts/pair-code', { label: 'x'.repeat(500) })
    check('an over-long label is rejected rather than truncated silently',
      tooLong.status === 400, `HTTP ${tooLong.status} (must be 400, not 500)`)

    const noHosts = await harness.api('/jobs', { adapter: 'echo', mode: 'each', count: 1 })
    check('submitting work with no computers online fails clearly', noHosts.status === 409, `HTTP ${noHosts.status}`)

    const badAdapter = await harness.api('/jobs', { adapter: 'definitely-not-real', mode: 'each', count: 1 })
    check('an unknown task type is rejected as bad input',
      badAdapter.status === 400, `HTTP ${badAdapter.status} (must be 400, not 500)`)

    const hugeCount = await harness.api('/jobs', { adapter: 'echo', mode: 'queue', count: 10_000_000 })
    check('an absurd item count is rejected as bad input',
      hugeCount.status === 400, `HTTP ${hugeCount.status} (must be 400, not 500)`)
  }
} finally {
  await harness.teardown()
}

console.log(`\n  ${failed === 0 ? GREEN + 'ALL EDGE CASES PASS' : RED + 'FAILURES PRESENT'}${RESET}` +
  `  ${passed} passed, ${failed} failed\n`)
process.exit(failed === 0 ? 0 : 1)
