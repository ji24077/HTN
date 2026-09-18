import { spawn, execFileSync, type ChildProcess } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
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

const DB_NAME = 'dwp_sim'
const OPERATOR = { email: 'sim@local', password: 'simulator-password-1' }

export type Agent = {
  label: string
  hostId: string
  home: string
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
  enroll(label: string): Promise<Agent>
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
  resetDatabase()

  const controlPort = options.controlPort ?? 8890
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
    DWP_LOG_LEVEL: 'info',
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

    async enroll(label: string): Promise<Agent> {
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
        const child = spawn(process.execPath, [
          'packages/agent/src/index.ts', 'pair', '--server', origin, '--code', code, '--label', label,
        ], { env: { ...process.env, DWP_HOME: home, DWP_LOG_DIR: options.logDir }, stdio: ['ignore', 'pipe', 'pipe'] })

        let stderr = ''
        child.stdout?.resume()
        child.stderr?.on('data', (c: Buffer) => { stderr += c.toString('utf8') })
        child.on('error', reject)
        child.on('exit', code => code === 0
          ? resolve()
          : reject(new Error(`pairing ${label} failed (exit ${code}): ${stderr.slice(0, 300)}`)))
      })

      const cfg = JSON.parse(readFileSync(join(home, 'config.json'), 'utf8')) as { hostId: string }
      return { label, hostId: cfg.hostId, home, proc: null }
    },

    start(agent: Agent): void {
      if (agent.proc) return
      if (!spawned.includes(agent)) spawned.push(agent)
      agent.proc = spawn(process.execPath, ['packages/agent/src/index.ts', 'run'], {
        env: { ...process.env, DWP_HOME: agent.home, DWP_LOG_DIR: options.logDir, DWP_LOG_LEVEL: 'info' },
        stdio: ['ignore', 'pipe', 'pipe'],
      })
      agent.proc.stdout?.resume()
      agent.proc.stderr?.resume()
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
      for (const agent of spawned) {
        if (agent.proc) { try { agent.proc.kill('SIGKILL') } catch {} ; agent.proc = null }
      }
      control.kill('SIGKILL')
      await net.close()
      for (const home of homes) { try { rmSync(home, { recursive: true, force: true }) } catch {} }
    },
  }

  return harness
}

export { sleep, OPERATOR }
