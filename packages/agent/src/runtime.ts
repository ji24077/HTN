/**
 * Where this agent is running, and what that means for the things it may do to itself.
 *
 * A container is not just another operating system. Three of the agent's behaviours are
 * wrong inside one and right outside it, and every one of them was a real failure before
 * this module existed:
 *
 *   - It rewrites its own install to update. In a container the filesystem is a layer
 *     someone built and can rebuild; a self-update writes into it, survives until the
 *     next `docker run`, and then silently reverts. Worse, the running image and the
 *     image in the registry disagree, so "which version is this machine on" stops having
 *     an answer.
 *   - It registers a login service. There is no login. The restart policy is the
 *     container runtime's, and a LaunchAgent/systemd unit written inside a container is
 *     a file nothing will ever read.
 *   - It opens a browser window. There is no browser, no display, and no one sitting in
 *     front of it. The window is opened on the *host*, against a published port.
 *
 * Detection is deliberately generous about what counts as a container and deliberately
 * explicit about the override. DWP_CONTAINER settles it either way, because the one
 * thing worse than guessing wrong is guessing wrong with no way to say so.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { cpus, freemem, hostname, totalmem } from 'node:os'

/** Loopback names a browser on the host can legitimately use to reach a published port. */
const LOOPBACK = new Set(['127.0.0.1', 'localhost', '::1', '[::1]'])

function envFlag(name: string): boolean | null {
  const raw = process.env[name]
  if (raw === undefined || raw === '') return null
  return !/^(0|false|no|off)$/i.test(raw)
}

/**
 * Sniff the container runtime, cached because it cannot change while we run.
 *
 * `/.dockerenv` covers Docker, `/run/.containerenv` covers Podman, and the cgroup path
 * covers the rest — including Kubernetes, where neither marker file exists. Any one of
 * them is enough; none of them is required, which is what DWP_CONTAINER is for.
 */
let detected: boolean | null = null

export function isContainer(): boolean {
  const override = envFlag('DWP_CONTAINER')
  if (override !== null) return override
  if (detected !== null) return detected
  detected = existsSync('/.dockerenv') || existsSync('/run/.containerenv') || cgroupLooksContainerised()
  return detected
}

function cgroupLooksContainerised(): boolean {
  try {
    return /\b(docker|containerd|kubepods|podman|lxc)\b/.test(readFileSync('/proc/1/cgroup', 'utf8'))
  } catch {
    return false
  }
}

/**
 * The address the window server binds to.
 *
 * Loopback outside a container, because the window is for whoever is sitting at the
 * machine and nobody else. Inside one, loopback means the container's own loopback,
 * which no published port can reach — so the page would be served to nothing. Binding
 * 0.0.0.0 there is not a weakening: the container's network is whatever the operator
 * published, the path token is still required, and the Host check below still runs.
 */
export function guiBindHost(): string {
  return process.env.DWP_GUI_HOST ?? (isContainer() ? '0.0.0.0' : '127.0.0.1')
}

/**
 * Pin the port when asked.
 *
 * Outside a container the agent walks a range to avoid a collision. A container has its
 * own network namespace, so there is nothing to collide with and the port is the one
 * thing the operator has already written down in a `-p` flag. Walking away from it would
 * publish a port with nothing behind it.
 */
export function guiPort(): number | null {
  const raw = process.env.DWP_GUI_PORT
  if (raw === undefined || raw === '') return isContainer() ? 43117 : null
  const port = Number(raw)
  return Number.isInteger(port) && port > 0 && port < 65536 ? port : null
}

/** Should launching the app try to put a window on a screen? */
export function shouldOpenWindow(): boolean {
  if (envFlag('DWP_NO_WINDOW') === true) return false
  return !isContainer()
}

/**
 * Is this Host header one we will answer to?
 *
 * The check exists to stop DNS rebinding: a name the attacker controls, resolving to an
 * address that reaches this server, would otherwise let a page in someone's browser
 * drive it. The defence is the *name*, not the port — an attacker who can pick the name
 * can pick the port too, and the earlier version's exact `127.0.0.1:<port>` comparison
 * bought nothing while breaking every published container whose host port differs from
 * the container port, which is most of them.
 *
 * DWP_GUI_ALLOWED_HOSTS widens it by name for the deliberate cases: a tailnet name, or a
 * hostname on a home LAN. Explicit, so nothing is reachable by a name nobody chose.
 */
export function hostAllowed(header: string | undefined): boolean {
  const host = (header ?? '').trim().toLowerCase()
  if (host === '') return false
  const name = host.startsWith('[')
    ? host.slice(0, host.indexOf(']') + 1)
    : (host.split(':')[0] ?? '')
  if (name === '') return false
  if (LOOPBACK.has(name)) return true

  const extra = (process.env.DWP_GUI_ALLOWED_HOSTS ?? '')
    .split(',').map(s => s.trim().toLowerCase()).filter(s => s !== '')
  if (extra.includes('*')) return true
  if (extra.includes(name)) return true

  /**
   * A container answers to its own name by default.
   *
   * Reaching one agent's window from a browser in another container, or from the host by
   * container name, is ordinary and the token still gates it. The container id is its
   * hostname unless the operator set one, so this is not a name an outsider can choose.
   */
  return isContainer() && name === hostname().toLowerCase()
}

/**
 * How this agent is kept alive, in the words of whoever is looking after it.
 *
 * Shown in the window in place of the login-service toggle, which a container cannot
 * honour. We cannot read the restart policy from in here — that is the daemon's, not
 * ours — so this says who owns the decision rather than pretending to know it.
 */
export function supervisorNote(): string {
  return isContainer()
    ? 'This agent runs in a container. It restarts when the container runtime restarts it '
      + '— set that with `restart: unless-stopped` in your compose file.'
    : 'Starts by itself after a restart, with no window open.'
}

/**
 * What this container is actually allowed to use, as opposed to what the host has.
 *
 * `os.cpus()` and `os.totalmem()` read the host's /proc, which Docker does not
 * virtualise — so a container capped at 1.5 CPUs and 512 MB cheerfully told the
 * scheduler it had 15 cores and 12 GB. Measured on this machine: every agent in a
 * four-container fleet reported identical host figures while each was limited to a
 * fraction of it. A scheduler that scores machines on those numbers would send the
 * biggest slices to the smallest containers.
 *
 * cgroup v2 first, because that is what any current Docker uses; v1 after, for older
 * hosts. Absent or unlimited reads as null, and the caller keeps the host's figure —
 * which is correct for a container run with no limits at all.
 */
export type ContainerLimits = { cpus: number | null; memoryBytes: number | null }

function readTrimmed(path: string): string | null {
  try {
    const text = readFileSync(path, 'utf8').trim()
    return text === '' ? null : text
  } catch {
    return null
  }
}

export function containerLimits(): ContainerLimits {
  if (!isContainer()) return { cpus: null, memoryBytes: null }
  return { cpus: cpuQuota(), memoryBytes: memoryLimit() }
}

function cpuQuota(): number | null {
  // v2: "<quota> <period>", or "max <period>" when uncapped.
  const v2 = readTrimmed('/sys/fs/cgroup/cpu.max')
  if (v2) {
    const [quota, period] = v2.split(/\s+/)
    if (quota && quota !== 'max' && period) {
      const q = Number(quota)
      const p = Number(period)
      if (Number.isFinite(q) && Number.isFinite(p) && q > 0 && p > 0) return q / p
    }
    return null
  }
  // v1: a quota of -1 means uncapped.
  const q = Number(readTrimmed('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'))
  const p = Number(readTrimmed('/sys/fs/cgroup/cpu/cpu.cfs_period_us'))
  if (Number.isFinite(q) && Number.isFinite(p) && q > 0 && p > 0) return q / p
  return null
}

function memoryLimit(): number | null {
  const raw = readTrimmed('/sys/fs/cgroup/memory.max')
    ?? readTrimmed('/sys/fs/cgroup/memory/limit_in_bytes')
  if (raw === null || raw === 'max') return null
  const bytes = Number(raw)
  /**
   * An "unlimited" cgroup reports a number near 2^63, not the word `max`, on plenty of
   * runtimes. Treating that as a real limit would have the agent claim nine exabytes of
   * memory, so anything at or above the host's own total is no limit at all.
   */
  if (!Number.isFinite(bytes) || bytes <= 0 || bytes >= totalmem()) return null
  return bytes
}

/** How much of the memory limit is still free, for a container that has one. */
export function containerFreeBytes(): number | null {
  const limit = memoryLimit()
  if (limit === null) return null
  const used = Number(readTrimmed('/sys/fs/cgroup/memory.current')
    ?? readTrimmed('/sys/fs/cgroup/memory/usage_in_bytes'))
  if (!Number.isFinite(used) || used < 0) return null
  return Math.max(0, limit - used)
}

/**
 * Which image this container came from, if the operator said.
 *
 * Docker does not tell a container its own image, so this can only ever be what the
 * compose file passed in. Worth having anyway: it is the answer to "what version is that
 * machine running", and without it a container reports the release version it was handed
 * when it paired — a number describing the *server*, not the image it is executing.
 */
export function imageReference(): string | null {
  const image = process.env.DWP_IMAGE?.trim()
  return image && isContainer() ? image : null
}

/**
 * The address the window is reachable at from outside this container.
 *
 * A container cannot discover its own published port — Docker maps it in the host's
 * network namespace and tells the process nothing. So the agent printed its *internal*
 * port, and anyone who had remapped it (because something already held 43117, which is
 * the common case on a machine that also runs the desktop app) was handed a URL that
 * does not work, with no hint that the real one differs.
 *
 * Whoever wrote the `-p` flag knows the answer, so they can pass it in.
 */
export function guiPublicOrigin(): string | null {
  const origin = process.env.DWP_GUI_PUBLIC_ORIGIN?.trim().replace(/\/+$/, '')
  return origin && origin !== '' ? origin : null
}

const MB = 1024 * 1024

/**
 * What this machine can offer, which in a container is not what the host has.
 *
 * Docker does not virtualise /proc, so `cpus()` and `totalmem()` report the whole host
 * from inside a container capped at a fraction of it. Reporting those made every agent
 * in a four-container fleet claim 15 cores and 12 GB while each was limited to two and
 * one — figures a scheduler would use to decide who gets the big slices.
 */
export function logicalCores(): number {
  const { cpus: quota } = containerLimits()
  if (quota === null) return cpus().length
  /**
   * A fractional quota has to become an integer, because that is what the protocol
   * carries. Rounded rather than floored: `--cpus 1.5` is meaningfully more than one
   * core's worth of work, and flooring every fractional limit to 1 would make a machine
   * look like the smallest possible worker whatever it was given.
   */
  return Math.max(1, Math.round(quota))
}

export function totalRamMb(): number {
  const { memoryBytes } = containerLimits()
  return Math.round((memoryBytes ?? totalmem()) / MB)
}

/**
 * Memory that could actually be given to a new task, in bytes, or null if unknowable.
 *
 * `os.freemem()` is the obvious answer and is badly wrong on two of the three platforms
 * this runs on. On macOS it counts only wholly untouched pages: measured on a 24 GB
 * MacBook with nothing much running, it reports 125 MB, because everything else is
 * cache the kernel would hand over the instant anyone asked. A guard reading that would
 * refuse every task forever while appearing switched on, which is worse than not
 * offering the guard. Linux has the same shape of problem — MemFree excludes the page
 * cache — and answers it with MemAvailable, which is the kernel's own estimate of what a
 * new allocation could get.
 */
export function availableBytes(): number | null {
  const inContainer = containerFreeBytes()
  if (inContainer !== null) return inContainer
  try {
    if (process.platform === 'linux') {
      const meminfo = readFileSync('/proc/meminfo', 'utf8')
      const kb = /^MemAvailable:\s+(\d+) kB$/m.exec(meminfo)?.[1]
      if (kb) return Number(kb) * 1024
      return freemem()
    }
    if (process.platform === 'darwin') return darwinAvailable()
  } catch {
    return freemem()
  }
  // Windows' freemem() is the commit-limit figure and means roughly the right thing.
  return freemem()
}

/**
 * macOS, via vm_stat. Cached: it is a subprocess, and this is read on every offer.
 *
 * Free plus inactive plus speculative plus purgeable is what Activity Monitor calls
 * available, and what the kernel will surrender without swapping.
 */
let darwinMemory: { at: number; bytes: number | null } | null = null

function darwinAvailable(): number | null {
  const now = Date.now()
  if (darwinMemory && now - darwinMemory.at < 5_000) return darwinMemory.bytes
  let bytes: number | null = null
  try {
    const out = execFileSync('vm_stat', [], { encoding: 'utf8', timeout: 2_000 })
    const pageSize = Number(/page size of (\d+) bytes/.exec(out)?.[1] ?? 4096)
    const pages = (label: string): number =>
      Number(new RegExp(`^${label}:\\s+(\\d+)\\.`, 'm').exec(out)?.[1] ?? 0)
    const free = pages('Pages free') + pages('Pages inactive')
      + pages('Pages speculative') + pages('Pages purgeable')
    bytes = free > 0 ? free * pageSize : null
  } catch {
    bytes = null
  }
  darwinMemory = { at: now, bytes }
  return bytes
}

export function freeRamMb(): number {
  return Math.round((availableBytes() ?? freemem()) / MB)
}

/** Identity of the container, for a window that has to say which one it is showing. */
export function containerIdentity(): { id: string; image: string | null } | null {
  if (!isContainer()) return null
  return { id: hostname(), image: process.env.DWP_IMAGE ?? null }
}
