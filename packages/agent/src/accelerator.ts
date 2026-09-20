/**
 * What compute device this machine can actually reach, and why not when it cannot.
 *
 * The server asserted `runtime: "cpu", vram_mib: 0` for every machine that ever
 * connected. That was true of the whole fleet, and it was also unfalsifiable: a laptop
 * with an idle RTX in it and a container that physically cannot see a GPU produced
 * byte-identical records, so no dashboard could distinguish "no hardware" from "hardware
 * we never wired up". Replacing an assertion with a measurement is the point of this
 * module; `reason` is the field that makes the measurement checkable.
 *
 * Detection is deliberately cheap and deliberately pessimistic. It never claims a device
 * it has not seen the inference runtime accept, because the cost of a false positive is
 * a scheduler routing GPU work to a machine that will fail it, while the cost of a false
 * negative is a line of text saying so on a dashboard.
 */
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { platform } from 'node:os'
import { createLogger, type AcceleratorReport, type Runtime, type RuntimePreference } from '@dwp/protocol'
import { isInstalled } from './workloads.ts'

const log = createLogger({ component: 'agent' })
const require_ = createRequire(import.meta.url)

/** The ONNX Runtime package that every non-CPU path in this agent goes through. */
const ORT = 'onnxruntime-node'

/**
 * Providers this platform would try, best first, if the runtime were present.
 *
 * This mirrors `preferredProviders()` in adapters/inference.ts rather than importing it,
 * because that module pulls in the runtime itself and this one must answer on a machine
 * where the runtime is absent — which is every machine running the default image.
 *
 * Note what `linux` means here: inside a container `process.platform` is always `linux`,
 * whatever the host is. A Windows machine running this agent in Docker therefore never
 * reaches the DirectML branch, which is not a bug in this function but a real property
 * of the deployment, and one the dashboard should be able to show rather than hide.
 */
function candidateProviders(): string[] {
  if (platform() === 'darwin') return ['coreml', 'cpu']
  if (platform() === 'win32') return ['dml', 'cpu']
  return ['cuda', 'cpu']
}

/** The protocol runtime a given execution provider corresponds to. */
function runtimeFor(provider: string): Runtime {
  if (provider === 'cuda') return 'cuda'
  if (provider === 'coreml') return 'mps'
  // DirectML has no protocol runtime. Reporting it as `cuda` would make the scheduler
  // match CUDA work onto a machine that cannot run it, so a dml-only machine stays
  // `cpu` for matching purposes and says so in `reason`.
  return 'cpu'
}

/**
 * Total VRAM, when the platform will state it without guessing.
 *
 * Only NVIDIA is asked, because only NVIDIA answers a single command with a single
 * number. Apple's unified memory has no separate VRAM figure to report and inventing one
 * from total RAM would be a fabrication; the field stays 0 and `device` carries the name.
 */
function nvidiaVramMib(): { vramMib: number; device: string | null } {
  try {
    const out = execFileSync(
      'nvidia-smi',
      ['--query-gpu=memory.total,name', '--format=csv,noheader,nounits'],
      { encoding: 'utf8', timeout: 2_000, stdio: ['ignore', 'pipe', 'ignore'] },
    )
    const first = out.split('\n')[0]?.trim()
    if (!first) return { vramMib: 0, device: null }
    const [mib, ...name] = first.split(',')
    const parsed = Number.parseInt((mib ?? '').trim(), 10)
    return {
      vramMib: Number.isFinite(parsed) && parsed > 0 ? parsed : 0,
      device: name.join(',').trim() || null,
    }
  } catch {
    // Absent, not on PATH, or no NVIDIA driver. All three mean the same thing here.
    return { vramMib: 0, device: null }
  }
}

/**
 * Ask the installed runtime which backends it was actually compiled with.
 *
 * `listSupportedBackends` is the only question that can be answered without building a
 * session, and a session needs a model this module has no business loading. Where the
 * function is missing — older builds — the answer is "cannot tell", which is reported as
 * such rather than assumed either way.
 */
function supportedBackends(): { names: string[] | null; detail: string } {
  try {
    // Required lazily: on the default image this package is absent by design, and a
    // static import would make this module unloadable there.
    const ort = require_(ORT) as { listSupportedBackends?: () => Array<{ name: string }> }
    if (typeof ort.listSupportedBackends !== 'function') {
      return { names: null, detail: 'inference runtime does not report its backends' }
    }
    return { names: ort.listSupportedBackends().map(b => b.name), detail: '' }
  } catch (err) {
    return { names: null, detail: `inference runtime failed to load: ${String(err).slice(0, 80)}` }
  }
}

/**
 * Probe this machine once and say what it found.
 *
 * `preference` is applied to the *reported* runtime rather than to the probe: a machine
 * forced to CPU still reports the device it has, so an operator can see what they are
 * turning down, but reports `runtime: 'cpu'` because that is where its work will run.
 */
export function detectAccelerator(preference: RuntimePreference = 'auto'): AcceleratorReport {
  const providers = candidateProviders()
  const cpuOnly = (reason: string, extra: Partial<AcceleratorReport> = {}): AcceleratorReport => ({
    runtime: 'cpu',
    vramMib: 0,
    available: false,
    reason,
    device: null,
    providers,
    ...extra,
  })

  if (!isInstalled(ORT)) {
    // The overwhelmingly common case, and the one the old hard-coded "cpu" hid: the
    // default image is built `--target agent`, which has no inference runtime at all.
    return cpuOnly('no inference runtime in this image (build --target ml)')
  }

  const { names, detail } = supportedBackends()
  if (names === null) return cpuOnly(detail)

  const usable = providers.filter(p => p !== 'cpu' && names.includes(p))
  if (usable.length === 0) {
    return cpuOnly(`inference runtime has no device backend here (has: ${names.join(', ') || 'none'})`)
  }

  const best = usable[0]!
  const runtime = runtimeFor(best)
  const { vramMib, device } = best === 'cuda' ? nvidiaVramMib() : { vramMib: 0, device: null }

  /**
   * Claiming CUDA requires a device, not merely a runtime that knows the word.
   *
   * `listSupportedBackends()` reports cuda on every linux build of onnxruntime-node --
   * measured, with `bundled: false`, in a container with no GPU and no CUDA libraries at
   * all. Taken at face value that made an ordinary CPU machine advertise `runtime:
   * 'cuda'`, which is worse than advertising nothing: the scheduler would route GPU work
   * to it, the provider would fail to load, and the task would run on the CPU with the
   * timings silently meaningless.
   *
   * `nvidia-smi` answering is the cheap, honest test. Where a GPU is genuinely passed
   * into a container the toolkit puts both the device and the CUDA libraries there
   * together, so this is also a good proxy for the libraries being present.
   */
  if (best === 'cuda' && vramMib === 0) {
    return cpuOnly('the inference runtime lists cuda, but no NVIDIA device is visible here')
  }

  if (runtime === 'cpu') {
    // A real device the protocol cannot express — DirectML today. Say so plainly rather
    // than either claiming a GPU the scheduler would mis-route work to, or claiming
    // nothing at all.
    return cpuOnly(`${best} is available but has no scheduler runtime; work stays on the CPU`, {
      available: false,
      device,
      vramMib,
    })
  }

  const report: AcceleratorReport = {
    runtime: preference === 'cpu' ? 'cpu' : runtime,
    vramMib,
    available: true,
    reason: preference === 'cpu' ? 'a device is available but this machine is set to CPU' : '',
    device,
    providers,
  }
  log.info('accelerator.detected', {
    backend: best,
    runtime: report.runtime,
    vramMib,
    device,
    preference,
  })
  return report
}

/**
 * The provider list to hand the inference runtime, given the operator's preference.
 *
 * `cpu` is not merely "prefer the CPU": it removes every device provider from the list,
 * so there is no path by which a machine set to CPU quietly ends up on a GPU because a
 * later fallback happened to succeed.
 */
export function providersFor(preference: RuntimePreference): string[] | null {
  if (preference === 'cpu') return ['cpu']
  return null   // null means "leave the runtime's own preference alone"
}
