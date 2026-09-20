import { execFileSync, spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import type { ExecutionReporter } from '../execution.ts'
import { PythonCapability } from '@dwp/protocol'

let checked: { python: string; pythonRuntime?: PythonCapability } | undefined

/** Opt-in image feature, verified once before advertising uploaded-project support. */
export function pythonCapability(): PythonCapability | undefined {
  const python = process.env.DWP_PROGRAM_PYTHON
  if (!python || process.platform === 'win32') return undefined
  if (checked?.python === python) return checked.pythonRuntime
  let pythonRuntime: PythonCapability | undefined
  try {
    const result = execFileSync(python, ['-m', 'orchestrator.worker.dwp_program', '--check'], {
      timeout: 15000, maxBuffer: 16384, stdio: ['ignore', 'pipe', 'pipe'],
    })
    const report = JSON.parse(result.toString())
    if (report.runtime === 'cpu') pythonRuntime = PythonCapability.parse(report.python)
  } catch { /* An unavailable runtime must not be advertised to the scheduler. */ }
  checked = { python, pythonRuntime }
  return pythonRuntime
}

export function programAvailable(): boolean {
  return pythonCapability() !== undefined
}

export async function runProgram(
  input: unknown,
  ctx: {
    hostId: string; server: string; attempt: number; taskId: string; jobId: string;
    signal: AbortSignal; report: ExecutionReporter;
    cleaned(data: { [key: string]: unknown }): void;
  },
): Promise<unknown> {
  if (!programAvailable()) throw new Error('This image does not have the Python/PyTorch CPU runtime')
  ctx.signal.throwIfAborted()
  const spec = (input as { spec?: { id?: string; job_id?: string; kind?: string } })?.spec
  if (spec?.id !== ctx.taskId || spec?.job_id !== ctx.jobId || spec?.kind !== 'python_project') {
    throw new Error('Project assignment does not match its task envelope')
  }
  ctx.report.step('Starting Python project worker on CPU')
  return await new Promise((resolve, reject) => {
    const child = spawn(process.env.DWP_PROGRAM_PYTHON!, ['-m', 'orchestrator.worker.dwp_program'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      // Do not pass the agent identity, invite, telemetry keys, or arbitrary secrets.
      env: { PATH: '/usr/local/bin:/usr/bin:/bin', PYTHONUNBUFFERED: '1' },
    })
    let result: unknown
    let received = false
    let protocolError: Error | undefined
    let diagnostics = ''
    // SIGTERM is handled by the bridge: it cancels execution and reaps the project's
    // process group before exiting. The bridge also watches for agent death.
    const abort = () => child.kill('SIGTERM')
    ctx.signal.addEventListener('abort', abort, { once: true })
    if (ctx.signal.aborted) abort()
    const lines = createInterface({ input: child.stdout })
    lines.on('line', line => {
      try {
        if (Buffer.byteLength(line) > 65536) throw new Error('Oversized project bridge message')
        const { kind, data } = JSON.parse(line)
        if (kind === 'result') {
          if (received) throw new Error('Duplicate project result')
          received = true; result = data
        } else if (kind === 'progress') ctx.report.progress(data.done, data.total)
        else if (kind === 'stdout' || kind === 'stderr') {
          if (kind === 'stdout') ctx.report.stdout(String(data.text))
          else ctx.report.stderr(String(data.text))
        } else if (kind === 'cleaned') ctx.cleaned(data)
      } catch (error) {
        protocolError = error instanceof Error ? error : new Error(String(error))
        abort()
      }
    })
    child.stderr.on('data', chunk => {
      diagnostics = (diagnostics + String(chunk)).slice(-2000)
      ctx.report.stderr(String(chunk))
    })
    child.stdin.on('error', () => { /* Exit reports startup failure without an unhandled EPIPE. */ })
    child.on('error', reject)
    child.on('close', code => {
      ctx.signal.removeEventListener('abort', abort)
      lines.close()
      if (ctx.signal.aborted) reject(new Error('Project execution cancelled'))
      else if (protocolError) reject(protocolError)
      else if (code !== 0 || !received) reject(new Error(`Project worker exited (${code}): ${diagnostics}`))
      else resolve(result)
    })
    child.stdin.end(JSON.stringify({
      spec, attempt: ctx.attempt, server: ctx.server, worker_id: ctx.hostId,
    }))
  })
}
