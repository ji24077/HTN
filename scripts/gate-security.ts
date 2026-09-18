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

function check(name: string, ok: boolean, detail = ''): void {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`)
  ok ? passed++ : failed++
}

/** Resolve to the rejection reason, or 'ACCEPTED' if the socket opened (always a failure). */
function tryConnect(token: string): Promise<string> {
  return new Promise(resolve => {
    const ws = new WebSocket(WS_URL, { headers: { authorization: `Bearer ${token}` } })
    const done = (v: string) => { try { ws.close() } catch {} ; resolve(v) }
    ws.on('unexpected-response', (_r, res) => done(String(res.headers['x-dwp-reason'] ?? res.statusCode)))
    ws.on('open', () => done('ACCEPTED'))
    ws.on('error', () => done('network-error'))
    setTimeout(() => done('timeout'), 5_000)
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
  check('unenrolled host is rejected', reason === 'unknown-host', reason)
}

// 2 ------------------------------------------------------- bad pairing codes
{
  const { privateKey } = generateKeyPairSync('ed25519')
  const pub = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')
  const res = await fetch(`${ORIGIN}/hosts/pair`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ code: 'ZZZZ-ZZZZ', publicKey: pub }),
  })
  check('invalid pairing code is rejected', res.status === 400, `HTTP ${res.status}`)
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
  check('pairing code cannot be redeemed twice', first.status === 200 && second.status === 400,
    `first ${first.status}, second ${second.status}`)

  // 4 ------------------------------------ wrong key for a real, enrolled host
  const hostId = first.body.hostId!
  const { privateKey: wrongKey } = generateKeyPairSync('ed25519')
  const reason = await tryConnect(mintAssertion(hostId, wrongKey))
  check('assertion signed by the wrong key is rejected', reason === 'bad-signature', reason)

  // 5 ----------------------------------------------- revoked host is rejected
  await api(`/hosts/${hostId}/revoke`, {})
  const { privateKey: realKey } = generateKeyPairSync('ed25519')
  const revokedReason = await tryConnect(mintAssertion(hostId, realKey))
  check('revoked host is rejected', revokedReason === 'unknown-host', revokedReason)
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
  const { hostId } = await res.json() as { hostId: string }

  const token = mintAssertion(hostId, privateKey)
  const first = await tryConnect(token)
  const replay = await tryConnect(token)          // same jti, second time
  check('a replayed assertion is rejected', first === 'ACCEPTED' && replay === 'replayed',
    `first ${first}, replay ${replay}`)

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
    setTimeout(() => { try { ws.close() } catch {} ; resolve('timeout — no offer received') }, 15_000)
  })
  check('a result signed with an unrelated key is rejected on a real task',
    forged.startsWith('rejected'), forged)
}

console.log(`\n  gate:security ${failed === 0 ? 'PASS' : 'FAIL'}  (${passed} passed, ${failed} failed)\n`)
process.exit(failed === 0 ? 0 : 1)
