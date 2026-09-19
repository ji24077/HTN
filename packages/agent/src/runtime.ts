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
import { existsSync, readFileSync } from 'node:fs'
import { hostname, totalmem } from 'node:os'

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

/** Identity of the container, for a window that has to say which one it is showing. */
export function containerIdentity(): { id: string; image: string | null } | null {
  if (!isContainer()) return null
  return { id: hostname(), image: process.env.DWP_IMAGE ?? null }
}
