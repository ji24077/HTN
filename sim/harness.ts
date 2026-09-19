import { spawn, execFileSync, type ChildProcess } from 'node:child_process'
import { createWriteStream, mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { startNetSim, type NetSim } from './netsim.ts'

/**
 * Brings up a complete, isolated copy of the system: its own database, a real control
 * service, real agent processes, and a simulated network between them.
 *
 * Nothing here is a mock. The agents are the same binaries a friend would run, and the
 * only thing standing in for reality is the network itself — which is the part we
 * cannot otherwise produce on one desk.
 */

/**
 * Port and database are per-run, not fixed.
 *
 * Both were hardcoded, so two people — or two sessions — running the suite at once
 * fought over the same port and the same database. The port guard then reported every
 * scenario failing at exactly 10s, which is indistinguishable from a genuinely broken
 * tree and is precisely what sends someone bisecting. A suite that cannot be run twice
 * at once is a suite people avoid running.
 */
function pickPort(): number {
  if (process.env.DWP_SIM_PORT) return Number(process.env.DWP_SIM_PORT)
  // Derived from the process id, so concurrent runs land on different ports without
  // coordinating, and a single run is still reproducible within itself.
  return 8890 + (process.pid % 900)
}

const SIM_PORT = pickPort()
const DB_NAME = process.env.DWP_SIM_DB ?? `dwp_sim_${SIM_PORT}`
/** Built by `cd ios/DWPAgentKit && swift build`; absent on a machine without Xcode. */
const SWIFT_AGENT = 'ios/DWPAgentKit/.build/debug/dwpagent'
const OPERATOR = { email: 'sim@local', password: 'simulator-password-1' }

/**
 * Which implementation this host runs.
 *
 * The iOS agent is a second, independent implementation of the same protocol, so the
 * only way to know it behaves like a host is to put it through the same scenarios as the
 * first one. `swift` runs the shared agent core as a macOS process — the same code the
 * phone runs, with a terminal instead of a screen.
 */
export type AgentKind = 'node' | 'swift'

export type Agent = {
  label: string
  hostId: string
  home: string
  kind: AgentKind
  proc: ChildProcess | null
}

export type Harness = {
  net: NetSim
  /** Origin the agents use — everything they do crosses the simulated network. */
  origin: string
  /** Origin that bypasses the simulator, for the test's own bookkeeping. */
  directOrigin: string
  logDir: string
  cookie: string
  enroll(label: string, kind?: AgentKind): Promise<Agent>
  start(agent: Agent): void
  stop(agent: Agent): void
  api(path: string, body?: unknown): Promise<Response>
  submitJob(spec: Record<string, unknown>): Promise<string>
  job(jobId: string): Promise<JobView>
  hosts(): Promise<{ id: string; label: string; online: boolean }[]>
  waitFor(label: string, predicate: () => Promise<boolean>, timeoutMs?: number): Promise<void>
  restartControl(): Promise<void>
  teardown(): Promise<void>
}

export type JobView = {
  job: { status: string; total_items: number }
  rollup: { state: string; n: number }[]
  tasks: { id: string; seq: number; state: string; attempts: number; host_label: string | null; output: unknown }[]
}

const sleep = (ms: number): Promise<void> => new Promise(r => setTimeout(r, ms))

function resetDatabase(): void {
  // A scenario that inherits another scenario's rows is not a test of anything.
  //
  // Each statement gets its own -c: psql wraps a single multi-statement string in one
  // transaction, and DROP DATABASE cannot run inside one.
  execFileSync('docker', [
    'exec', 'dwp-db', 'psql', '-U', 'dwp', '-d', 'postgres', '-v', 'ON_ERROR_STOP=1',
    '-c', `drop database if exists ${DB_NAME} with (force)`,
    '-c', `create database ${DB_NAME}`,
  ], { stdio: 'pipe' })
}

async function waitForHttp(url: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(2000) })
      if (res.ok) return true
    } catch {}
    await sleep(200)
  }
  return false
}

export async function startHarness(options: { logDir: string; controlPort?: number }): Promise<Harness> {
  /**
   * Refuse to run against a server this harness did not start.
   *
   * A control process left behind by an aborted run keeps the port, so the new one fails
   * to bind and dies — and the scenario then talks to the old server, holding a session
   * for a database that has just been dropped and recreated. The symptom is a 401 far
   * from the cause, which cost a long time to track down once already.
   */
  const { createServer } = await import('node:net')
  const port = options.controlPort ?? SIM_PORT

  const probePort = (): Promise<boolean> => new Promise(resolve => {
    const probe = createServer()
    probe.once('error', () => resolve(true))
    probe.once('listening', () => probe.close(() => resolve(false)))
    probe.listen(port, '127.0.0.1')
  })

  // Wait rather than fail on sight: the previous scenario's server was killed a moment
  // ago and the port takes a beat to come back. Failing immediately turned one stale
  // process into nine scenarios reporting a problem none of them had.
  let taken = await probePort()
  for (let i = 0; taken && i < 20; i++) {
    await sleep(500)
    taken = await probePort()
  }
  if (taken) {
    throw new Error(
      `port ${port} is still in use after 10s — most likely a control service left by an aborted run.\n` +
      `  Stop it first:  pkill -f 'packages/control/src/index.ts'`)
  }

  resetDatabase()

  const controlPort = options.controlPort ?? SIM_PORT
  const net = await startNetSim(controlPort)
  const directOrigin = `http://127.0.0.1:${controlPort}`
  const origin = `http://127.0.0.1:${net.port}`
  const homes: string[] = []
  // Every agent ever started, so teardown can guarantee none survive into the next
  // scenario. A leaked agent quietly retries forever and pollutes the next run's logs.
  const spawned: Agent[] = []

  const controlEnv = {
    ...process.env,
    DATABASE_URL: `postgres://dwp:dwp@localhost:5433/${DB_NAME}`,
    PORT: String(controlPort),
    BIND_HOST: '127.0.0.1',
    // Agents are told to dial the simulator, so every byte they send is impaired.
    PUBLIC_ORIGIN: origin,
    BOOTSTRAP_EMAIL: OPERATOR.email,
    BOOTSTRAP_PASSWORD: OPERATOR.password,
    DWP_LOG_DIR: options.logDir,
    // Overridable, so a failing scenario can be re-run with the rejection paths visible.
    DWP_LOG_LEVEL: process.env.DWP_LOG_LEVEL ?? 'info',
  }

  let control: ChildProcess

  const spawnControl = async (): Promise<void> => {
    control = spawn(process.execPath, ['packages/control/src/index.ts'], {
      env: controlEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    control.stdout?.resume()
    control.stderr?.resume()
    if (!(await waitForHttp(`${directOrigin}/health`, 20_000))) {
      throw new Error('control service did not come up')
    }
  }
  await spawnControl()

  // Log in against the direct origin: the test's own calls should not be affected by
  // impairments meant for the agents.
  const loginRes = await fetch(`${directOrigin}/auth/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(OPERATOR),
  })
  if (!loginRes.ok) throw new Error(`sim login failed: ${loginRes.status}`)
  let cookie = (loginRes.headers.getSetCookie?.()[0] ?? '').split(';')[0]!

  const api = async (path: string, body?: unknown): Promise<Response> =>
    fetch(`${directOrigin}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { cookie, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    })

  const harness: Harness = {
    net,
    origin,
    directOrigin,
    logDir: options.logDir,
    get cookie() { return cookie },

    async enroll(label: string, kind: AgentKind = 'node'): Promise<Agent> {
      const codeRes = await api('/hosts/pair-code', { label })
      const { code } = await codeRes.json() as { code: string }
      const home = mkdtempSync(join(tmpdir(), `dwp-sim-${label}-`))
      homes.push(home)

      // Pair through the simulator, exactly as a real agent would.
      //
      // This MUST be async. The simulated network runs on this process's event loop, so
      // a synchronous spawn here would block the very loop that has to carry the child's
      // request to the server — the child would wait forever for a reply only we can
      // deliver. Any future helper that shells out while agents are talking has the same
      // constraint.
      await new Promise<void>((resolve, reject) => {
        const [bin, argv] = kind === 'swift'
          ? [SWIFT_AGENT, ['pair', '--server', origin, '--code', code, '--label', label]]
          : [process.execPath, ['packages/agent/src/index.ts', 'pair',
                                '--server', origin, '--code', code, '--label', label]]
        const child = spawn(bin, argv,
          { env: { ...process.env, DWP_HOME: home, DWP_LOG_DIR: options.logDir }, stdio: ['ignore', 'pipe', 'pipe'] })

        let stderr = ''
        child.stdout?.resume()
        child.stderr?.on('data', (c: Buffer) => { stderr += c.toString('utf8') })
        child.on('error', reject)
        child.on('exit', code => code === 0
          ? resolve()
          : reject(new Error(`pairing ${label} failed (exit ${code}): ${stderr.slice(0, 300)}`)))
      })

      const cfg = JSON.parse(readFileSync(join(home, 'config.json'), 'utf8')) as { hostId: string }
      return { label, hostId: cfg.hostId, home, kind, proc: null }
    },

    start(agent: Agent): void {
      if (agent.proc) return
      if (!spawned.includes(agent)) spawned.push(agent)
      agent.proc = agent.kind === 'swift'
        ? spawn(SWIFT_AGENT, ['run'], {
            env: { ...process.env, DWP_HOME: agent.home, DWP_LOG_DIR: options.logDir, DWP_LOG_LEVEL: 'info' },
            stdio: ['ignore', 'pipe', 'pipe'],
          })
        : spawn(process.execPath, ['packages/agent/src/index.ts', 'run'], {
            env: { ...process.env, DWP_HOME: agent.home, DWP_LOG_DIR: options.logDir, DWP_LOG_LEVEL: 'info' },
            stdio: ['ignore', 'pipe', 'pipe'],
          })

      // The Swift agent logs to stdout rather than to the structured log directory, so
      // capture it — without this a failing scenario says only "timed out", which is the
      // least useful thing a test can say.
      if (agent.kind === 'swift') {
        mkdirSync(options.logDir, { recursive: true })
        const sink = createWriteStream(join(options.logDir, `${agent.label}.log`), { flags: 'a' })
        agent.proc.stdout?.pipe(sink)
        agent.proc.stderr?.pipe(sink)
      } else {
        agent.proc.stdout?.resume()
        agent.proc.stderr?.resume()
      }
    },

    stop(agent: Agent): void {
      if (!agent.proc) return
      agent.proc.kill('SIGKILL')
      agent.proc = null
    },

    api,

    async submitJob(spec): Promise<string> {
      const res = await api('/jobs', spec)
      if (!res.ok) throw new Error(`job rejected: ${res.status} ${await res.text()}`)
      const { jobId } = await res.json() as { jobId: string }
      return jobId
    },

    async job(jobId): Promise<JobView> {
      const res = await api(`/jobs/${jobId}`)
      if (!res.ok) {
        // Surface the failure instead of handing back a shape the caller will destructure
        // into undefined. A polling helper that crashes on `.status` reports a timeout,
        // which hides whatever the server actually said.
        throw new Error(`GET /jobs/${jobId} -> ${res.status} ${(await res.text()).slice(0, 200)}`)
      }
      return await res.json() as JobView
    },

    async hosts() {
      const res = await api('/hosts')
      const { hosts } = await res.json() as { hosts: { id: string; label: string; online: boolean }[] }
      return hosts
    },

    async waitFor(label, predicate, timeoutMs = 60_000): Promise<void> {
      const deadline = Date.now() + timeoutMs
      let lastErr: unknown = null
      while (Date.now() < deadline) {
        try { if (await predicate()) return } catch (err) { lastErr = err }
        await sleep(400)
      }
      throw new Error(`timed out after ${timeoutMs}ms waiting for: ${label}` +
        (lastErr ? ` (last error: ${String(lastErr)})` : ''))
    },

    async restartControl(): Promise<void> {
      control.kill('SIGKILL')
      await sleep(500)
      await spawnControl()
      const res = await fetch(`${directOrigin}/auth/login`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(OPERATOR),
      })
      cookie = (res.headers.getSetCookie?.()[0] ?? '').split(';')[0]!
    },

    async teardown(): Promise<void> {
      // Drop the per-run database. Without this every suite run leaves one behind.
      const dropDatabase = (): void => {
        try {
          execFileSync('docker', [
            'exec', 'dwp-db', 'psql', '-U', 'dwp', '-d', 'postgres',
            '-c', `drop database if exists ${DB_NAME} with (force)`,
          ], { stdio: 'pipe' })
        } catch {}
      }

      for (const agent of spawned) {
        if (agent.proc) { try { agent.proc.kill('SIGKILL') } catch {} ; agent.proc = null }
      }
      control.kill('SIGKILL')
      await net.close()
      for (const home of homes) { try { rmSync(home, { recursive: true, force: true }) } catch {} }
      dropDatabase()
    },
  }

  return harness
}

export { sleep, OPERATOR }
