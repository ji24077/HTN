/**
 * Gate `security` — the access-control checks that must hold before any broader trial.
 *
 * Every case here asserts a REJECTION. A pass means the platform refused something it
 * should refuse; a failure is a reason to stop, not a bug to file and move past.
 *
 *   node --env-file-if-exists=.env scripts/gate-security.ts [controlOrigin]
 */
import { generateKeyPairSync, createPublicKey, sign } from 'node:crypto'
import WebSocket from 'ws'
import { mintAssertion, envelope, hashOutput, signAttestation } from '@dwp/protocol'

const ORIGIN = process.argv[2] ?? process.env.PUBLIC_ORIGIN ?? 'http://localhost:8787'
const WS_URL = `${ORIGIN.replace(/^http/, 'ws')}/agent/connect`
const EMAIL = process.env.BOOTSTRAP_EMAIL ?? 'you@example.com'
const PASSWORD = process.env.BOOTSTRAP_PASSWORD ?? 'change-me'

let passed = 0
let failed = 0
let inconclusive = 0

/**
 * Three outcomes, not two.
 *
 * "The boundary held", "the boundary did not hold" and "the check could not run" are
 * different facts, and collapsing the third into the second is how a security gate
 * stops being believed. A dial that timed out under load previously reported as a
 * failed replay check — alarming, wrong, and unreproducible afterwards, which is the
 * worst combination. Report it as unverified and exit non-zero so it is never quietly
 * green either.
 */
function check(name: string, ok: boolean, detail = ''): void {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`)
  ok ? passed++ : failed++
}

function unverified(name: string, detail: string): void {
  console.log(`  ????  ${name}  — could not verify: ${detail}`)
  inconclusive++
}

/** Did this result come from the system under test, or from the test giving up? */
const isInconclusive = (r: string): boolean => r === 'timeout' || r === 'network-error'

/**
 * Being rate limited is not a security finding.
 *
 * Pairing is limited to 20 attempts per five minutes, keyed on client address — so every
 * terminal, script and session pairing from this machine draws on **one shared budget**,
 * and none of them can see the others' consumption. This gate spends 4 of the 20 per run.
 *
 * When that budget runs out, pairing returns 429, no host is created, and every check
 * that needs one fails on a malformed assertion — reading as several broken security
 * boundaries when not one of them was exercised, and passing again ten minutes later.
 *
 * The first explanation for this was "consecutive runs exhaust it", which the arithmetic
 * disproves: 4 per run against 20 is five runs per window. It was two sessions sharing
 * the bucket. Worth recording, because the wrong lesson — space your runs out — would
 * have people avoiding something that was never the cause.
 */
let rateLimited = false
function noteRateLimit(res: Response): boolean {
  if (res.status !== 429) return false
  rateLimited = true
  return true
}

/** Resolve to the rejection reason, or 'ACCEPTED' if the socket opened (always a failure). */
function tryConnect(token: string): Promise<string> {
  return new Promise(resolve => {
    const ws = new WebSocket(WS_URL, { headers: { authorization: `Bearer ${token}` } })
    const done = (v: string) => { try { ws.close() } catch {} ; resolve(v) }
    ws.on('unexpected-response', (_r, res) => done(String(res.headers['x-dwp-reason'] ?? res.statusCode)))
    ws.on('open', () => done('ACCEPTED'))
    ws.on('error', () => done('network-error'))
    // Generous: this competes with the simulator and the suites for the machine, and a
    // slow dial is not a security finding.
    setTimeout(() => done('timeout'), 20_000)
  })
}

// ------------------------------------------------------------------- session

const loginRes = await fetch(`${ORIGIN}/auth/login`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: EMAIL, password: PASSWORD }),
})
if (!loginRes.ok) throw new Error(`cannot log in as ${EMAIL}: ${loginRes.status}`)
const cookie = (loginRes.headers.getSetCookie?.()[0] ?? '').split(';')[0]!

async function api(path: string, body?: unknown): Promise<Response> {
  return fetch(`${ORIGIN}${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { cookie, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
}

console.log(`\ngate:security against ${ORIGIN}\n`)

// 1 --------------------------------------------------------- unenrolled host
{
  const { privateKey } = generateKeyPairSync('ed25519')
  const reason = await tryConnect(mintAssertion(crypto.randomUUID(), privateKey))
  if (isInconclusive(reason)) unverified('unenrolled host is rejected', reason)
  else check('unenrolled host is rejected', reason === 'unknown-host', reason)
}

// 2 ------------------------------------------------------- bad pairing codes
{
  const { privateKey } = generateKeyPairSync('ed25519')
  const pub = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')
  const res = await fetch(`${ORIGIN}/hosts/pair`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ code: 'ZZZZ-ZZZZ', publicKey: pub }),
  })
  if (noteRateLimit(res)) unverified('invalid pairing code is rejected', 'rate limited (HTTP 429)')
  else check('invalid pairing code is rejected', res.status === 400, `HTTP ${res.status}`)
}

// 3 -------------------------------------------- pairing code is single-use
{
  const codeRes = await api('/hosts/pair-code', { label: 'gate-single-use' })
  const { code } = await codeRes.json() as { code: string }
  const enroll = async () => {
    const { privateKey } = generateKeyPairSync('ed25519')
    const pub = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')
    const res = await fetch(`${ORIGIN}/hosts/pair`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code, publicKey: pub }),
    })
    return { status: res.status, body: await res.json() as { hostId?: string } }
  }
  const first = await enroll()
  const second = await enroll()

  if (first.status === 429 || second.status === 429) {
    rateLimited = true
    unverified('pairing code cannot be redeemed twice', `rate limited (${first.status}/${second.status})`)
    unverified('assertion signed by the wrong key is rejected', 'no host could be enrolled')
    unverified('revoked host is rejected', 'no host could be enrolled')
  } else {
    check('pairing code cannot be redeemed twice', first.status === 200 && second.status === 400,
      `first ${first.status}, second ${second.status}`)

    // 4 ---------------------------------- wrong key for a real, enrolled host
    const hostId = first.body.hostId!
    const { privateKey: wrongKey } = generateKeyPairSync('ed25519')
    const reason = await tryConnect(mintAssertion(hostId, wrongKey))
    if (isInconclusive(reason)) unverified('assertion signed by the wrong key is rejected', reason)
    else check('assertion signed by the wrong key is rejected', reason === 'bad-signature', reason)

    // 5 --------------------------------------------- revoked host is rejected
    await api(`/hosts/${hostId}/revoke`, {})
    const { privateKey: realKey } = generateKeyPairSync('ed25519')
    const revokedReason = await tryConnect(mintAssertion(hostId, realKey))
    if (isInconclusive(revokedReason)) unverified('revoked host is rejected', revokedReason)
    else check('revoked host is rejected', revokedReason === 'unknown-host', revokedReason)
  }
}

// 6 --------------------------------------------------- assertion replay
{
  const codeRes = await api('/hosts/pair-code', { label: 'gate-replay' })
  const { code } = await codeRes.json() as { code: string }
  const { privateKey } = generateKeyPairSync('ed25519')
  const pub = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')
  const res = await fetch(`${ORIGIN}/hosts/pair`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ code, publicKey: pub }),
  })
  if (noteRateLimit(res) || !res.ok) {
    // No host, so neither of the checks below can be exercised. Say that, rather than
    // reporting a malformed assertion as a replay-protection failure.
    unverified('a replayed assertion is rejected', `could not enrol a host (HTTP ${res.status})`)
    unverified('a result signed with an unrelated key is rejected on a real task',
      `could not enrol a host (HTTP ${res.status})`)
  } else {
  const { hostId } = await res.json() as { hostId: string }

  const token = mintAssertion(hostId, privateKey)
  const first = await tryConnect(token)
  const replay = await tryConnect(token)          // same jti, second time
  // The first dial must genuinely succeed for the replay to mean anything. If it timed
  // out, nothing was replayed and the check proved nothing either way.
  if (isInconclusive(first) || isInconclusive(replay)) {
    unverified('a replayed assertion is rejected', `first ${first}, replay ${replay}`)
  } else {
    check('a replayed assertion is rejected', first === 'ACCEPTED' && replay === 'replayed',
      `first ${first}, replay ${replay}`)
  }

  // 7 ------------------------------------------- forged result signature
  //
  // This must run against a task the host genuinely holds. An earlier version of this
  // check used a random task id and "passed" because the server bailed before ever
  // reaching the signature test — a green light that proved nothing.
  const forged = await new Promise<string>(resolve => {
    const ws = new WebSocket(WS_URL, { headers: { authorization: `Bearer ${mintAssertion(hostId, privateKey)}` } })
    let jobId: string | undefined

    ws.on('open', () => {
      ws.send(JSON.stringify(envelope('hello', {
        capability: {
          agentVersion: 'gate', os: 'linux', arch: 'x64', cpuModel: 'gate', logicalCores: 1,
          totalRamMb: 1024, freeRamMb: 512, adapters: ['echo'],
        },
        consent: { paused: false, allowCompute: true, allowBrowser: false, maxConcurrency: 1 },
      })))
      void api('/jobs', { adapter: 'echo', mode: 'each', count: 1, hostId })
        .then(r => r.json() as Promise<{ jobId?: string }>)
        .then(j => { jobId = j.jobId })
    })

    ws.on('message', (raw: Buffer) => {
      const msg = JSON.parse(raw.toString('utf8')) as { type: string; payload: Record<string, string> }
      if (msg.type !== 'task.offer') return
      const taskId = msg.payload.taskId as string
      const leaseId = msg.payload.leaseId as string
      const output = { nonce: 'forged', hostId, hostname: 'nowhere', os: 'linux', arch: 'x64', elapsedMs: 1 }
      const startedAt = new Date().toISOString()
      const finishedAt = new Date().toISOString()
      const outputHash = hashOutput(output)
      const { privateKey: attackerKey } = generateKeyPairSync('ed25519')

      ws.send(JSON.stringify(envelope('task.accept', { taskId, leaseId })))
      ws.send(JSON.stringify(envelope('task.result', {
        taskId, leaseId, attempt: 1, output, outputHash, startedAt, finishedAt, hostReportedMs: 1,
        // Signed with a key the control service has never associated with this host.
        signature: signAttestation(attackerKey, { taskId, attempt: 1, hostId, outputHash, startedAt, finishedAt }),
      })))

      setTimeout(() => {
        void api(`/jobs/${jobId}`)
          .then(r => r.json() as Promise<{ tasks: { id: string; state: string; output: unknown }[] }>)
          .then(j => {
            const task = j.tasks.find(t => t.id === taskId)
            ws.close()
            resolve(task && task.state !== 'succeeded' && task.output === null
              ? `rejected (task left ${task.state})`
              : `ACCEPTED FORGED RESULT (task ${task?.state})`)
          })
      }, 1200)
    })

    ws.on('error', () => resolve('network-error'))
    setTimeout(() => { try { ws.close() } catch {} ; resolve('timeout') }, 30_000)
  })
  if (forged === 'timeout' || forged === 'network-error') {
    unverified('a result signed with an unrelated key is rejected on a real task', forged)
  } else {
    check('a result signed with an unrelated key is rejected on a real task',
      forged.startsWith('rejected'), forged)
  }
  }
}

const verdict = failed > 0 ? 'FAIL' : inconclusive > 0 ? 'UNVERIFIED' : 'PASS'
console.log(`\n  gate:security ${verdict}  (${passed} passed, ${failed} failed` +
  `${inconclusive > 0 ? `, ${inconclusive} could not be checked` : ''})\n`)
if (rateLimited) {
  console.log(`  Pairing was rate limited — 4 attempts per run against a budget of 20 per`)
  console.log(`  five minutes, shared by everything that pairs from this machine: other`)
  console.log(`  terminals, other sessions, anything else enrolling hosts right now.`)
  console.log(`  Nothing here failed. Wait for the window and run it once.\n`)
} else if (inconclusive > 0 && failed === 0) {
  console.log(`  Nothing failed, but ${inconclusive} check(s) could not run — usually load.`)
  console.log(`  Re-run on a quiet machine before treating this as green.\n`)
}
// Non-zero for both, so an unverified gate is never mistaken for a passing one.
process.exit(failed === 0 && inconclusive === 0 ? 0 : 1)
