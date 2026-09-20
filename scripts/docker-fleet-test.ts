/**
 * Bring up a control service, a fleet of containerised agents, and prove the whole path.
 *
 *   node scripts/docker-fleet-test.ts --agents 4 --tasks 24
 *
 * This is the check that the Docker migration actually migrated something. Every part of
 * it is a claim that was made about the container image and is worth disbelieving:
 *
 *   - a container can join a network it was only ever given an invite link for
 *   - N containers are N machines, not one machine connecting N times
 *   - work spreads across them and every task comes back
 *   - what comes back is *right*, checked by recomputing the deterministic walker
 *     locally and comparing numbers, not by trusting that a result arrived
 *   - each container's own window agrees with the server about what it ran
 *   - killing one mid-flight loses no work
 *
 * It runs against its own database schema and its own port, so it cannot disturb a
 * control service already running on this machine.
 */
import { execFile, spawn } from 'node:child_process'
import { randomBytes, randomUUID } from 'node:crypto'
import { promisify } from 'node:util'
import { setTimeout as sleep } from 'node:timers/promises'
import { existsSync } from 'node:fs'
import { perturb, evaluate, randomGenome, GENOME_SIZE } from '@dwp/protocol/walker.js'

const exec = promisify(execFile)

// ------------------------------------------------------------------- arguments

const argv = process.argv.slice(2)
const flag = (name: string): boolean => argv.includes(`--${name}`)
const value = (name: string, fallback: string): string => {
  const i = argv.indexOf(`--${name}`)
  return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1]! : fallback
}

const AGENTS = Number(value('agents', '4'))
const TASKS = Number(value('tasks', '24'))
const PORT = Number(value('port', '8099'))
const IMAGE = value('image', 'dwp-agent:test')
const SCHEMA = value('schema', 'dwp_docker_test')
const KEEP = flag('keep')
const SKIP_BUILD = flag('skip-build')
const SKIP_CHAOS = flag('skip-chaos')
const NAME = 'dwp-fleet-test'

/**
 * How a container reaches the control service running on this host.
 *
 * Docker Desktop provides host.docker.internal; on Linux the compose/run flag
 * --add-host=host.docker.internal:host-gateway does the same, which the run below
 * passes unconditionally because it is harmless where the name already resolves.
 */
const HOST_FROM_CONTAINER = value('host-alias', 'host.docker.internal')

// --------------------------------------------------------------------- plumbing

let failures = 0
const pass = (what: string, detail = ''): void => console.log(`  \u001b[32m✓\u001b[0m ${what}${detail ? `  ${detail}` : ''}`)
const fail = (what: string, detail = ''): void => { failures += 1; console.log(`  \u001b[31m✗\u001b[0m ${what}${detail ? `  ${detail}` : ''}`) }
const step = (what: string): void => console.log(`\n\u001b[1m${what}\u001b[0m`)
const check = (ok: boolean, what: string, detail = ''): boolean => { (ok ? pass : fail)(what, detail); return ok }

const docker = (args: string[]): Promise<{ stdout: string; stderr: string }> =>
  exec('docker', args, { maxBuffer: 64 * 1024 * 1024 })

async function waitFor<T>(
  what: string,
  probe: () => Promise<T | null>,
  { timeoutMs = 120_000, everyMs = 1_000 } = {},
): Promise<T> {
  const deadline = Date.now() + timeoutMs
  let last: unknown = null
  while (Date.now() < deadline) {
    try {
      const got = await probe()
      if (got !== null && got !== undefined) return got
    } catch (err) { last = err }
    await sleep(everyMs)
  }
  throw new Error(`timed out waiting for ${what}${last ? `: ${String(last)}` : ''}`)
}

// ------------------------------------------------------------ the control service

const ADMIN_TOKEN = randomBytes(24).toString('hex')
const ORIGIN = `http://127.0.0.1:${PORT}`

const admin = (path: string, init: RequestInit = {}): Promise<Response> =>
  fetch(`${ORIGIN}${path}`, {
    ...init,
    headers: { authorization: `Bearer ${ADMIN_TOKEN}`, 'content-type': 'application/json', ...(init.headers ?? {}) },
  })

/** The database password belongs to the container that holds it, not to a file here. */
async function databaseUrl(): Promise<string> {
  const container = process.env.DWP_DB_CONTAINER ?? 'dwp-db'
  const { stdout } = await docker(['inspect', container, '--format', '{{range .Config.Env}}{{println .}}{{end}}'])
  const password = stdout.split('\n').find(l => l.startsWith('POSTGRES_PASSWORD='))?.slice('POSTGRES_PASSWORD='.length)
  if (!password) throw new Error(`no POSTGRES_PASSWORD in container "${container}" — is it running?`)
  const { stdout: portOut } = await docker(['port', container, '5432/tcp']).catch(() => ({ stdout: '' }))
  const port = /:(\d+)\s*$/.exec(portOut.split('\n')[0] ?? '')?.[1] ?? '5433'
  const db = process.env.DWP_DB_NAME ?? 'ethan_main'
  return `postgres://dwp:${password}@127.0.0.1:${port}/${db}`
}

let control: ReturnType<typeof spawn> | null = null

/**
 * Start from an empty schema, because invites are rate limited and this run needs many.
 *
 * A control service allows ten pair codes per owner per hour and keeps used ones for an
 * hour after that, so the third run of a four-agent test in quick succession is refused
 * with "Device invite limit reached" — a real limit doing its real job, aimed at the
 * wrong thing. Dropping this run's own schema is the honest way past it: it resets only
 * the throwaway database this harness created, never the fleet next door in `public`.
 */
async function resetSchema(): Promise<void> {
  const container = process.env.DWP_DB_CONTAINER ?? 'dwp-db'
  const db = process.env.DWP_DB_NAME ?? 'ethan_main'
  await docker([
    'exec', container, 'psql', '-v', 'ON_ERROR_STOP=1', '-U', 'dwp', '-d', db,
    '-c', `DROP SCHEMA IF EXISTS "${SCHEMA}" CASCADE`,
  ])
}

async function startControl(): Promise<void> {
  const server = 'backend/.venv/bin/orchestrator-server'
  if (!existsSync(server)) throw new Error(`no backend at ${server} — run: uv sync --project backend`)

  control = spawn(server, [], {
    env: {
      ...process.env,
      PUBLIC_ORIGIN: '',
      DEMO_UI: 'true',
      // 0.0.0.0, because the point of the exercise is to be reached from a container.
      LISTEN_HOST: '0.0.0.0',
      LISTEN_PORT: String(PORT),
      ADMIN_TOKEN,
      DATABASE_URL: await databaseUrl(),
      // Its own schema, so this cannot touch the fleet already running on this machine.
      DATABASE_SCHEMA: SCHEMA,
      DWP_RELEASES_DIR: 'releases',
      DWP_FIXTURES_DIR: 'fixtures',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  const log: string[] = []
  control.stdout?.on('data', (b: Buffer) => log.push(b.toString()))
  control.stderr?.on('data', (b: Buffer) => log.push(b.toString()))
  control.on('exit', code => { if (code !== 0 && code !== null) console.error(log.slice(-40).join('')) })

  await waitFor('the control service', async () => {
    const res = await fetch(`${ORIGIN}/healthz`, { signal: AbortSignal.timeout(2_000) })
    return res.ok ? true : null
  }, { timeoutMs: 60_000, everyMs: 500 })
}

// ------------------------------------------------------------------- the agents

type Agent = { index: number; name: string; volume: string; guiPort: number; hostId?: string; label: string }
const agents: Agent[] = []

async function invite(): Promise<string> {
  const res = await admin('/v1/device-invites', { method: 'POST' })
  if (!res.ok) throw new Error(`could not create an invite: HTTP ${res.status} ${await res.text()}`)
  return (await res.json() as { code: string }).code
}

async function startAgent(index: number): Promise<Agent> {
  const code = await invite()
  const agent: Agent = {
    index,
    name: `${NAME}-${index}`,
    volume: `${NAME}-data-${index}`,
    guiPort: 43200 + index,
    label: `docker-${index}`,
  }
  await docker([
    'run', '-d',
    '--name', agent.name,
    '--add-host', 'host.docker.internal:host-gateway',
    '-v', `${agent.volume}:/data`,
    '-p', `127.0.0.1:${agent.guiPort}:43117`,
    '-e', `DWP_INVITE=http://${HOST_FROM_CONTAINER}:${PORT}/join?code=${code}`,
    '-e', `DWP_LABEL=${agent.label}`,
    '-e', `DWP_IMAGE=${IMAGE}`,
    IMAGE, 'gui', '--hidden',
  ])
  agents.push(agent)
  return agent
}

/** The window's address, read out of the container's own lock file. */
async function guiState(agent: Agent): Promise<Record<string, any>> {
  const { stdout } = await docker(['exec', agent.name, 'cat', '/data/gui.json'])
  const lock = JSON.parse(stdout) as { token: string }
  const res = await fetch(`http://127.0.0.1:${agent.guiPort}/${lock.token}/api/state`,
    { signal: AbortSignal.timeout(5_000) })
  if (!res.ok) throw new Error(`window on ${agent.name} answered HTTP ${res.status}`)
  return await res.json() as Record<string, any>
}

async function teardown(): Promise<void> {
  if (KEEP) {
    console.log(`\n  --keep: left ${agents.length} containers and the control service on ${ORIGIN} running.`)
    console.log(`  Clean up with:  docker rm -f ${agents.map(a => a.name).join(' ')} && docker volume rm ${agents.map(a => a.volume).join(' ')}\n`)
    return
  }
  for (const a of agents) {
    await docker(['rm', '-f', a.name]).catch(() => null)
    await docker(['volume', 'rm', a.volume]).catch(() => null)
  }
  control?.kill('SIGTERM')
  await sleep(500)
  control?.kill('SIGKILL')
}

// ------------------------------------------------------------------- work to do

type Worker = { id: string; state: 'alive' | 'unhealthy' | 'offline'; paused: boolean }
/**
 * The server's Task, not a guess at it.
 *
 * The id lives on `spec`, not on the task — a task is a spec plus the state of trying to
 * run it, and reading `task.id` silently yields undefined, which turns every lookup into
 * a miss and every wait into a timeout with nothing to show for it.
 */
type Task = {
  spec: { id: string; kind: string; job_id: string }
  state: 'queued' | 'assigned' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  worker_id?: string | null
  result?: any
  attestation?: { hostId?: string; outputHash?: string; taskId?: string } | null
  failure?: string
  generation?: number
}

/**
 * A genome the walker can actually evaluate.
 *
 * This was 64 hand-made numbers, and GENOME_SIZE is 308. The simulation read past the
 * end of the array, every weight came out undefined, and fitness was NaN — on the agent
 * and here alike. Which is precisely why it went unnoticed: `Math.abs(NaN - NaN)` is
 * NaN, `NaN > 1e-9` is false, so the comparison below reported a match on two values
 * that were not numbers. The check passed 150 times without once comparing a result.
 */
const WALKER_PARENT = randomGenome(20260919)
const WALKER_SIGMA = 0.12
const WALKER_STEPS = 220

function walkerTask(id: string, jobId: string, seeds: number[]) {
  return {
    id, job_id: jobId, kind: 'walker_evolution',
    payload: { generation: 1, parent: WALKER_PARENT, sigma: WALKER_SIGMA, seeds, steps: WALKER_STEPS },
    requirements: { runtime: 'cpu', vram_mib: 0 },
    max_attempts: 3, timeout_seconds: 120,
  }
}

function echoTask(id: string, jobId: string, nonce: string) {
  return {
    id, job_id: jobId, kind: 'echo',
    payload: { nonce, sleepMs: 150 },
    requirements: { runtime: 'cpu', vram_mib: 0 },
    max_attempts: 3, timeout_seconds: 60,
  }
}

/**
 * The server accepts at most 100 specs per request, so a stress run submits in chunks.
 *
 * Sent as one list, 200 tasks come back as a flat "invalid task submission" — a 400 that
 * names neither the limit nor the field, and reads like a malformed payload rather than
 * a batch that is merely too long.
 */
const SUBMIT_CHUNK = 100

async function submit(tasks: unknown[]): Promise<void> {
  for (let i = 0; i < tasks.length; i += SUBMIT_CHUNK) {
    const chunk = tasks.slice(i, i + SUBMIT_CHUNK)
    const res = await admin('/v1/tasks', { method: 'POST', body: JSON.stringify({ tasks: chunk }) })
    if (!res.ok) throw new Error(`submitting failed: HTTP ${res.status} ${await res.text()}`)
  }
}

const finished = (t: Task): boolean => ['succeeded', 'failed', 'cancelled'].includes(t.state)
const succeeded = (t: Task): boolean => t.state === 'succeeded'

/**
 * The listing returns the 500 most recent tasks. That is plenty for one run of this
 * harness and would quietly stop being plenty for a bigger one, so the shortfall is
 * reported rather than waited on until the timeout.
 */
const LISTING_LIMIT = 500

async function tasksById(ids: Set<string>): Promise<Map<string, Task>> {
  if (ids.size > LISTING_LIMIT) {
    throw new Error(`${ids.size} tasks exceeds the ${LISTING_LIMIT}-row listing; run fewer at once`)
  }
  const res = await admin('/v1/tasks')
  if (!res.ok) throw new Error(`could not list tasks: HTTP ${res.status}`)
  const all = await res.json() as Task[]
  return new Map(all.filter(t => ids.has(t.spec.id)).map(t => [t.spec.id, t]))
}

async function awaitTasks(ids: Set<string>, timeoutMs: number): Promise<Map<string, Task>> {
  return waitFor('every task to finish', async () => {
    const found = await tasksById(ids)
    if (found.size < ids.size) return null
    return [...found.values()].every(finished) ? found : null
  }, { timeoutMs, everyMs: 1_000 })
}

/**
 * The adapter's own output, fetched one task at a time.
 *
 * The listing deliberately omits `result` (TASK_SUMMARY_COLUMNS in store.py): 500 tasks
 * with their payloads inline would be a very large response for a page that only wants
 * to know what is running. Reading a result therefore means asking for that task, and
 * treating the listing's absent `result` as an empty result is how every verification
 * here first came back "the container sent undefined".
 */
async function fullTask(id: string): Promise<Task> {
  const res = await admin(`/v1/tasks/${encodeURIComponent(id)}`)
  if (!res.ok) throw new Error(`could not read task ${id}: HTTP ${res.status}`)
  return await res.json() as Task
}

/**
 * Which machine produced this, preferring the output's own claim.
 *
 * `worker_id` is the server's record of who it gave the task to; `output.hostId` is what
 * the machine itself signed into the result. They should agree, and the interesting
 * failure is exactly the case where they do not — so prefer the machine's own word and
 * let the spread check see it.
 */
function ranOn(task: Task, out: any): string | null {
  if (out && typeof out === 'object' && typeof out.hostId === 'string') return out.hostId
  return task.worker_id ?? null
}

// ------------------------------------------------------------------------ main

async function main(): Promise<void> {
  console.log(`\n\u001b[1mContainerised agent fleet — end to end\u001b[0m`)
  console.log(`  ${AGENTS} agents, ${TASKS} tasks, image ${IMAGE}, control on ${ORIGIN} (schema ${SCHEMA})`)

  step('Building the image')
  if (SKIP_BUILD) {
    console.log('  --skip-build: using whatever is tagged already')
  } else {
    const started = Date.now()
    await docker(['build', '-f', 'deploy/Dockerfile.agent', '-t', IMAGE, '.'])
    pass('image built', `${((Date.now() - started) / 1000).toFixed(0)}s`)
  }
  const { stdout: size } = await docker(['image', 'inspect', IMAGE, '--format', '{{.Size}}'])
  pass('image size', `${(Number(size.trim()) / 1024 / 1024).toFixed(0)} MB`)

  step('Starting a control service of its own')
  await resetSchema()
  await startControl()
  pass('control service healthy', ORIGIN)

  step(`Starting ${AGENTS} containers, each with only an invite link`)
  for (let i = 1; i <= AGENTS; i += 1) {
    await startAgent(i)
    pass(`started ${NAME}-${i}`)
  }

  /**
   * Every container must appear as its own worker.
   *
   * The failure this catches is the one that makes containerising a stateful agent
   * dangerous: a shared volume, or an identity baked into the image, would have all of
   * them present the same key. The server would accept each in turn and evict the last,
   * and the fleet would look like one flapping machine rather than N healthy ones.
   */
  step('Waiting for every container to appear as a distinct machine')
  const workers = await waitFor(`${AGENTS} workers online`, async () => {
    const res = await admin('/v1/workers')
    if (!res.ok) return null
    const all = await res.json() as Worker[]
    const online = all.filter(w => w.state === 'alive')
    return online.length >= AGENTS ? online : null
  }, { timeoutMs: 180_000 })
  const ids = new Set(workers.map(w => w.id))
  check(ids.size >= AGENTS, `${ids.size} distinct host identities for ${AGENTS} containers`)
  for (const [i, agent] of agents.entries()) {
    const state = await guiState(agent)
    agent.hostId = state.hostId
    check(state.runtime?.container === true, `${agent.name} knows it is containerised`)
    check(state.paired === true && state.connection === 'online', `${agent.name} is joined and online`,
      `${state.label} ${String(state.hostId).slice(0, 8)}`)
    if (i === 0) {
      check(state.runsAtLogin === true, 'the login toggle reports the container runtime, not a LaunchAgent')
      check(/image/i.test(String(state.updateNote)), 'the window says updates arrive as images')
    }
  }
  const hostIds = new Set(agents.map(a => a.hostId))
  check(hostIds.size === AGENTS, 'every container reports its own host id', [...hostIds].map(h => String(h).slice(0, 8)).join(' '))

  /**
   * Accuracy, not just delivery.
   *
   * The walker is deterministic: the same parent, sigma and seed produce the same
   * fitness on any machine. So the numbers a container sends back can be recomputed
   * here and compared exactly — which is the difference between "a result arrived" and
   * "the result is the one that machine was asked to produce".
   */
  step(`Submitting ${TASKS} tasks and checking what comes back`)
  const jobId = `dockertest-${randomUUID().slice(0, 8)}`
  const submitted = new Map<string, { kind: 'echo'; nonce: string } | { kind: 'walker'; seeds: number[] }>()
  const batch: unknown[] = []
  for (let i = 0; i < TASKS; i += 1) {
    const id = `${jobId}-${String(i).padStart(3, '0')}`
    if (i % 2 === 0) {
      const nonce = randomBytes(8).toString('hex')
      submitted.set(id, { kind: 'echo', nonce })
      batch.push(echoTask(id, jobId, nonce))
    } else {
      const seeds = Array.from({ length: 6 }, (_, k) => i * 1000 + k)
      submitted.set(id, { kind: 'walker', seeds })
      batch.push(walkerTask(id, jobId, seeds))
    }
  }
  const startedAt = Date.now()
  await submit(batch)
  const done = await awaitTasks(new Set(submitted.keys()), 300_000)
  const elapsed = (Date.now() - startedAt) / 1000
  pass('all tasks finished', `${elapsed.toFixed(1)}s, ${(TASKS / elapsed).toFixed(1)}/s`)

  const failed = [...done.values()].filter(t => !succeeded(t))
  check(failed.length === 0, `${TASKS - failed.length}/${TASKS} succeeded`,
    failed.length > 0 ? failed.slice(0, 3).map(t => `${t.spec.id}:${t.state}:${t.failure ?? ''}`).join(' ') : '')

  let echoChecked = 0
  let walkerChecked = 0
  let crossChecked = 0
  let wrong = 0
  /** Cached beside the task, so the spread check below does not fetch every result twice. */
  const outputs = new Map<string, any>()
  for (const [id, spec] of submitted) {
    const task = done.get(id)
    if (!task || !succeeded(task)) continue
    const full = await fullTask(id)
    const out = full.result
    outputs.set(id, out)

    /**
     * Three independent claims about who ran this, which must agree.
     *
     * The server recorded a worker_id when it handed the lease out; the adapter wrote a
     * hostId into its output; the machine signed an attestation naming itself and the
     * task. Under load these are the values that would diverge if two connections ever
     * got crossed — one container's result landing against another's lease — and nothing
     * else in this suite would notice, because each result on its own looks perfectly
     * well formed.
     */
    if (out?.hostId && full.worker_id && out.hostId !== full.worker_id) {
      wrong += 1
      fail(`${id} was leased to one machine and answered by another`, `${full.worker_id} vs ${out.hostId}`)
      continue
    }
    if (full.attestation?.hostId && out?.hostId && full.attestation.hostId !== out.hostId) {
      wrong += 1
      fail(`${id} signed attestation names a different machine`, `${full.attestation.hostId} vs ${out.hostId}`)
      continue
    }
    if (full.attestation?.taskId && full.attestation.taskId !== id) {
      wrong += 1
      fail(`${id} carries an attestation for a different task`, String(full.attestation.taskId))
      continue
    }
    crossChecked += 1

    if (spec.kind === 'echo') {
      if (out?.nonce !== spec.nonce) { wrong += 1; fail(`${id} echoed the wrong nonce`, `${out?.nonce} != ${spec.nonce}`); continue }
      if (out?.os !== 'linux') { wrong += 1; fail(`${id} did not run on the image's OS`, String(out?.os)); continue }
      echoChecked += 1
    } else {
      // Recompute locally. Identical inputs must give identical fitness.
      let ok = true
      for (const seed of spec.seeds) {
        const expected = evaluate(perturb(WALKER_PARENT, WALKER_SIGMA, seed), WALKER_STEPS)
        const got = (out?.results ?? []).find((r: any) => r.seed === seed)
        if (!got) { ok = false; fail(`${id} omitted seed ${seed}`); break }
        /**
         * Insist on real numbers before comparing them.
         *
         * Without this the comparison cannot fail on a pair of non-numbers: NaN is not
         * greater than the tolerance, and JSON turns NaN into null on the way here, so a
         * silently broken simulation reads as a perfect match. Reject the shape first,
         * then the value.
         */
        if (typeof got.fitness !== 'number' || !Number.isFinite(got.fitness)) {
          ok = false
          fail(`${id} seed ${seed} returned no usable fitness`, JSON.stringify(got.fitness))
          break
        }
        if (!Number.isFinite(expected.fitness)) {
          ok = false
          fail(`${id} seed ${seed}: the local recomputation is not a number`,
            'the test fixture is wrong, not the agent')
          break
        }
        if (Math.abs(got.fitness - expected.fitness) > 1e-9 || got.fell !== expected.fell) {
          ok = false
          fail(`${id} seed ${seed} disagrees`, `container ${got.fitness} vs local ${expected.fitness}`)
          break
        }
      }
      if (ok) walkerChecked += 1; else wrong += 1
    }
  }
  check(wrong === 0, 'every result matches what was asked for',
    `${echoChecked} nonces echoed back, ${walkerChecked} walker batches recomputed locally and identical`)
  check(crossChecked === done.size, 'lease, output and signature name the same machine on every task',
    `${crossChecked}/${done.size}`)

  /**
   * Work must actually spread. One container doing all of it would pass every check
   * above and mean the fleet is a single machine with N idle spectators.
   */
  const perHost = new Map<string, number>()
  for (const [id, task] of done) {
    const host = ranOn(task, outputs.get(id))
    if (host) perHost.set(host, (perHost.get(host) ?? 0) + 1)
  }
  check(perHost.size >= Math.min(AGENTS, 2), `work spread over ${perHost.size} of ${AGENTS} containers`,
    [...perHost.entries()].map(([h, n]) => `${String(h).slice(0, 6)}:${n}`).join(' '))

  /**
   * Each container's own window must agree with the server.
   *
   * This is the frontend requirement stated as a test: the "Recent work" panel, the
   * running list and the summary are served from the same process that did the work, so
   * if the server counted 7 runs on a host and its window says 3, one of them is lying.
   */
  step('Checking each window against the server')
  let recorded = 0
  for (const agent of agents) {
    const state = await guiState(agent)
    const serverSays = perHost.get(String(agent.hostId)) ?? 0
    const windowSays = state.history?.summary?.runs ?? 0
    recorded += windowSays
    check(windowSays >= serverSays,
      `${agent.name} window reports ${windowSays} runs, server attributed ${serverSays}`)
    if (windowSays > 0) {
      const newest = state.history?.recent?.[0]
      check(typeof newest?.adapter === 'string' && typeof newest?.durationMs === 'number',
        `${agent.name} recent work has an adapter and a duration`, `${newest?.adapter} ${Math.round(newest?.durationMs)}ms`)
    }
  }
  check(recorded >= TASKS, `windows account for ${recorded} runs across the fleet (>= ${TASKS} submitted)`)

  step('Pausing one container and confirming it stops taking work')
  const paused = agents[0]!
  await docker(['exec', paused.name, 'node', 'packages/agent/src/index.ts', 'pause'])
  const pausedState = await waitFor('the window to report paused', async () => {
    const s = await guiState(paused)
    return s.paused === true ? s : null
  }, { timeoutMs: 20_000 })
  check(pausedState.paused === true, `${paused.name} reports paused`)
  await docker(['exec', paused.name, 'node', 'packages/agent/src/index.ts', 'resume'])
  const resumed = await waitFor('the window to report resumed', async () => {
    const s = await guiState(paused)
    return s.paused === false ? s : null
  }, { timeoutMs: 20_000 })
  check(resumed.paused === false, `${paused.name} accepts work again`)

  /**
   * A stopping container must hand its work back, not merely disappear.
   *
   * This is the check that matters for `docker compose up -d`, which is now the ordinary
   * way to deploy: it stops every agent at once. An agent that just exits leaves each
   * task it held untouchable for the 45-second lease, so a fleet-wide restart stalls the
   * whole queue — and, before the hand-off was ordered correctly, the server handed the
   * task straight back to the same dying agent and burned a retry doing it.
   */
  step('Stopping a container while it holds work')
  {
    const signalJob = `signal-${randomUUID().slice(0, 8)}`
    const signalId = `${signalJob}-000`
    await submit([echoTask(signalId, signalJob, randomBytes(8).toString('hex'))])
    // A long sleep, so the task is unambiguously in flight when the signal lands.
    await submit([{
      id: `${signalJob}-hold`, job_id: signalJob, kind: 'echo',
      payload: { nonce: 'holding', sleepMs: 20_000 },
      requirements: { runtime: 'cpu', vram_mib: 0 },
      max_attempts: 3, timeout_seconds: 300,
    }])
    const holdId = `${signalJob}-hold`

    const holder = await waitFor('the long task to start running', async () => {
      const t = await fullTask(holdId).catch(() => null)
      return t && t.state === 'running' && t.worker_id ? t : null
    }, { timeoutMs: 60_000, everyMs: 500 })
    const victim = agents.find(a => a.hostId === holder.worker_id)
    check(victim !== undefined, 'found which container picked the task up', victim?.name ?? '?')

    const stoppedAt = Date.now()
    await docker(['stop', '-t', '30', victim!.name])
    const stopMs = Date.now() - stoppedAt
    const exit = (await docker(['inspect', victim!.name, '--format', '{{.State.ExitCode}}'])).stdout.trim()
    // 137 is SIGKILL: the signal was ignored and Docker lost patience.
    check(exit === '0', 'exited cleanly on SIGTERM rather than being killed', `exit ${exit} in ${stopMs}ms`)
    check(stopMs < 10_000, 'stopped without waiting out the kill timeout', `${stopMs}ms`)
    check(/agent.handing_off/.test((await docker(['logs', victim!.name])).stderr
      + (await docker(['logs', victim!.name])).stdout), 'it logged handing its work back')

    /**
     * Requeued in seconds, not in a lease. The old behaviour parked the task for 45s;
     * a generous ceiling here still fails loudly if the hand-off ever stops working.
     */
    const released = await waitFor('the task to be released', async () => {
      const t = await fullTask(holdId)
      return t.state === 'queued' || (t.worker_id && t.worker_id !== victim!.hostId) ? t : null
    }, { timeoutMs: 40_000, everyMs: 500 })
    const releaseMs = Date.now() - stoppedAt
    check(releaseMs < 15_000, 'the work was released promptly, not held for the lease',
      `${(releaseMs / 1000).toFixed(1)}s`)

    const settled = await awaitTasks(new Set([signalId, holdId]), 180_000)
    const hold = settled.get(holdId)!
    check(succeeded(hold), 'the handed-back task completed on another machine',
      `${hold.state} on ${String(hold.worker_id).slice(0, 8)} at generation ${hold.generation}`)
    /**
     * It must not have bounced back to the machine that was shutting down. That is what
     * happened when the decline went out before consent was withdrawn, and it cost a
     * retry as well as the wait.
     */
    check(hold.worker_id !== victim!.hostId, 'it did not go back to the container that was stopping')
    check((hold.generation ?? 0) <= 2, 'it took one retry, not several',
      `generation ${hold.generation}`)

    await docker(['start', victim!.name])
    const revived = await waitFor('the stopped container to come back', async () => {
      const s2 = await guiState(victim!).catch(() => null)
      return s2 && s2.connection === 'online' ? s2 : null
    }, { timeoutMs: 120_000 })
    check(revived.hostId === victim!.hostId, 'it came back as the same machine',
      String(revived.hostId).slice(0, 8))
    check(revived.paused === false, 'and accepting work again, not still paused from the hand-off')
  }

  if (!SKIP_CHAOS) {
    /**
     * Kill a container while it holds work, and check the network does not lose it.
     *
     * This is the claim that matters most for a fleet of other people's laptops: the
     * machine that goes away mid-task is the normal case, not the exception.
     */
    step('Killing a container mid-flight')
    const chaosJob = `chaos-${randomUUID().slice(0, 8)}`
    const chaosIds = new Set<string>()
    const chaosBatch: unknown[] = []
    for (let i = 0; i < AGENTS * 3; i += 1) {
      const id = `${chaosJob}-${String(i).padStart(3, '0')}`
      chaosIds.add(id)
      chaosBatch.push(walkerTask(id, chaosJob, Array.from({ length: 40 }, (_, k) => i * 100 + k)))
    }
    await submit(chaosBatch)
    await sleep(2_500)
    const victim = agents[agents.length - 1]!
    /**
     * What this machine had recorded before it was killed.
     *
     * The check after the restart used to be "it has some history", which passes or
     * fails on whether the scheduler happened to send this particular container any work
     * — nothing to do with what is being tested. Preservation is the actual claim, so
     * compare against the count taken here. `record()` uses appendFileSync, so anything
     * already finished is on disk and a SIGKILL cannot take it back.
     */
    const before = (await guiState(victim)).history?.summary?.runs ?? 0
    await docker(['kill', victim.name])
    pass(`killed ${victim.name} while the fleet was busy`)
    const after = await awaitTasks(chaosIds, 300_000)
    const lost = [...after.values()].filter(t => !succeeded(t))
    check(lost.length === 0, `all ${chaosIds.size} tasks completed despite a machine disappearing`,
      lost.length > 0 ? lost.slice(0, 3).map(t => `${t.spec.id}:${t.state}`).join(' ') : '')

    step('Restarting it and confirming it is the same machine')
    await docker(['start', victim.name])
    const back = await waitFor('the restarted container to reconnect', async () => {
      const s = await guiState(victim).catch(() => null)
      return s && s.connection === 'online' ? s : null
    }, { timeoutMs: 120_000 })
    check(back.hostId === victim.hostId, 'same host id after a restart — the volume carried its identity',
      String(back.hostId).slice(0, 8))
    const afterRuns = back.history?.summary?.runs ?? 0
    check(afterRuns >= before, 'its history survived being killed',
      `${before} runs before, ${afterRuns} after`)
  }

  step('Result')
  if (failures === 0) console.log(`\n  \u001b[32mEverything passed.\u001b[0m ${AGENTS} containers, ${TASKS} verified tasks.\n`)
  else console.log(`\n  \u001b[31m${failures} check${failures === 1 ? '' : 's'} failed.\u001b[0m\n`)
}

main()
  .catch((err: unknown) => {
    failures += 1
    console.error(`\n\u001b[31m  ${err instanceof Error ? err.stack : String(err)}\u001b[0m\n`)
  })
  .finally(async () => {
    await teardown()
    process.exit(failures === 0 ? 0 : 1)
  })
