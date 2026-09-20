import { createRequire } from 'node:module'
import { execFile } from 'node:child_process'
import { existsSync, rmSync, readdirSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { promisify } from 'node:util'
import { platform } from 'node:os'
import { installRoot } from './paths.ts'
import { loadConfig, saveConfig } from './config.ts'
import { programAvailable } from './adapters/program.ts'

const exec = promisify(execFile)
const require_ = createRequire(import.meta.url)

// pnpm is a .cmd shim on Windows. Node refuses to spawn one without a shell, and
// CreateProcess only ever appends .exe — so without this, `pnpm agent enable ml` fails
// there with an opaque EINVAL on the one command that makes a machine useful for ML.
const IS_WINDOWS = process.platform === 'win32'

/**
 * Heavy workloads are opt-in.
 *
 * ONNX Runtime alone is 287 MB, and two thirds of that is binaries for operating systems
 * the machine is not running. Making every friend download it before they can contribute
 * anything is the single biggest barrier to adding a machine — especially when `echo`
 * and the walker need nothing at all and can start work immediately.
 */
export type WorkloadId = 'ml' | 'browser'

type Workload = {
  id: WorkloadId
  adapter: string
  package: string
  version: string
  describe: string
  approxMb: number
  /** Extra step after install, e.g. downloading a browser. */
  postInstall?: string[]
}

export const WORKLOADS: Workload[] = [
  {
    id: 'ml',
    adapter: 'cpu_inference_batch',
    package: 'onnxruntime-node',
    version: '1.30.0',
    describe: 'machine-learning inference (ONNX models)',
    approxMb: 85,
  },
  {
    id: 'browser',
    adapter: 'remote_browser_session',
    package: 'playwright',
    version: '1.63.0',
    describe: 'remote browser sessions',
    approxMb: 180,
    postInstall: ['exec', 'playwright', 'install', 'chromium'],
  },
]

/** Where onnxruntime-node keeps its per-platform binaries, if it is installed. */
function findOnnxBinDir(): string | null {
  const candidates = [
    join(installRoot(), 'packages', 'agent', 'node_modules', 'onnxruntime-node', 'bin', 'napi-v6'),
    join(installRoot(), 'node_modules', 'onnxruntime-node', 'bin', 'napi-v6'),
  ]
  for (const c of candidates) if (existsSync(c)) return c

  // pnpm keeps the real files under .pnpm; the paths above are symlinks into it.
  const store = join(installRoot(), 'node_modules', '.pnpm')
  if (!existsSync(store)) return null
  try {
    for (const entry of readdirSync(store)) {
      if (!entry.startsWith('onnxruntime-node@')) continue
      const c = join(store, entry, 'node_modules', 'onnxruntime-node', 'bin', 'napi-v6')
      if (existsSync(c)) return c
    }
  } catch {}
  return null
}

/** Is the package present? Resolve rather than import, so nothing heavy is loaded. */
export function isInstalled(pkg: string): boolean {
  try {
    require_.resolve(pkg)
    return true
  } catch {
    return false
  }
}

/** Adapters this machine can actually run right now. */
export function availableAdapters(): string[] {
  const adapters = ['echo', 'walker_evolution']   // pure JavaScript, always available
  for (const w of WORKLOADS) if (isInstalled(w.package)) adapters.push(w.adapter)
  if (programAvailable()) adapters.push('python_project', 'python_program')
  return adapters
}

/**
 * Delete the platform binaries this machine can never execute.
 *
 * onnxruntime-node ships macOS, Linux and Windows builds in one package — 287 MB, of
 * which only the current platform's 85 MB is usable. The rest is pure waste on every
 * machine that installs it.
 */
export function pruneForeignBinaries(): { freedMb: number; kept: string } {
  const current = platform()
  let freed = 0
  try {
    // Locate the package by path rather than by resolving it.
    //
    // require.resolve caches a failed lookup, and this runs in the same process that
    // just installed the package — so the one command that adds a runtime was unable to
    // see it, and the prune silently did nothing.
    const binDir = findOnnxBinDir()
    if (!binDir) return { freedMb: 0, kept: current }

    for (const name of readdirSync(binDir)) {
      if (name === current) continue
      const path = join(binDir, name)
      freed += dirSizeMb(path)
      rmSync(path, { recursive: true, force: true })
    }
  } catch {
    return { freedMb: 0, kept: current }
  }
  return { freedMb: Math.round(freed), kept: current }
}

function dirSizeMb(path: string): number {
  let bytes = 0
  const walk = (p: string): void => {
    for (const e of readdirSync(p, { withFileTypes: true })) {
      const child = join(p, e.name)
      if (e.isDirectory()) walk(child)
      // statSync imported at the top: `require` does not exist in an ES module, and
      // this is the second time that has slipped in — it typechecks and fails only when
      // the line actually runs.
      else {
        try { bytes += statSync(child).size } catch {}
      }
    }
  }
  try { walk(path) } catch {}
  return bytes / 1024 / 1024
}

/** Install a workload's runtime. Shared by `enable` and by update, which restores them. */
export async function installWorkload(id: WorkloadId, opts: { quiet?: boolean } = {}): Promise<void> {
  const workload = WORKLOADS.find(w => w.id === id)
  if (!workload) throw new Error(`unknown workload "${id}"`)
  const say = (msg: string): void => { if (!opts.quiet) console.log(msg) }

  const agentDir = join(installRoot(), 'packages', 'agent')
  say(`\n  Installing ${workload.describe} (about ${workload.approxMb} MB)…\n`)
  // --prod, because the tree was installed that way. Without it pnpm refuses with
  // ERR_PNPM_INCLUDED_DEPS_CONFLICT: the add would pull in development tooling the
  // install deliberately left out.
  await exec('pnpm', ['add', '--prod', `${workload.package}@${workload.version}`], {
    cwd: agentDir, timeout: 900_000, shell: IS_WINDOWS,
  })

  if (workload.postInstall) {
    say(`  Fetching what it needs to run…\n`)
    await exec('pnpm', workload.postInstall, { cwd: agentDir, timeout: 900_000, shell: IS_WINDOWS })
  }

  if (id === 'ml') {
    const { freedMb, kept } = pruneForeignBinaries()
    if (freedMb > 0) say(`  Removed ${freedMb} MB of binaries for other platforms (kept ${kept}).\n`)
  }
}

export async function enableWorkload(id: WorkloadId): Promise<void> {
  await installWorkload(id)

  // Remember the choice where an update cannot overwrite it.
  const cfg = loadConfig()
  if (cfg) {
    const enabled = new Set(cfg.enabledWorkloads ?? [])
    enabled.add(id)
    saveConfig({ ...cfg, enabledWorkloads: [...enabled] })
  }

  const workload = WORKLOADS.find(w => w.id === id)!
  console.log(`  Enabled. Restart the agent to start taking ${workload.adapter} work.\n`)
}
