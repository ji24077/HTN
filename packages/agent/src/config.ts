import { existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs'
import { AGENT_HOME, CONFIG_PATH, PAUSE_PATH } from './paths.ts'
import type { WorkerTelemetry } from '@dwp/protocol'
import type { Limits } from './limits.ts'

export type AgentConfig = {
  server: string
  wsUrl: string
  hostId: string
  label: string
  allowCompute: boolean
  allowBrowser: boolean
  maxConcurrency: number
  /**
   * Whether this machine may use a device beyond its CPU, as the operator last set it.
   *
   * Stored rather than held in memory because the control service sets it and the
   * container runtime restarts the process: a preference that lived only in RAM would
   * silently revert to `auto` on every `docker restart`, and the dashboard would go on
   * showing the value the operator chose while the machine had already forgotten it.
   * Absent means `auto`, which is what every machine did before this setting existed.
   */
  runtimePreference?: 'auto' | 'cpu'
  /**
   * What this machine is willing to do, and how much of itself it will give.
   *
   * Kept in the config rather than on the server because it is the owner's decision
   * about their own hardware: it has to survive the control service being unreachable,
   * and it must not be something the network can quietly widen. Absent means no limits,
   * which is how every machine behaved before this existed.
   */
  limits?: Limits
  /** Delivered by the paired platform; persists across GUI/service/binary restarts. */
  telemetry?: WorkerTelemetry
  /**
   * The operator's release signing key, pinned at pairing.
   *
   * Updates are verified against this and nothing else. Accepting a key that arrives
   * alongside a release would let whoever served the release also vouch for it.
   */
  releaseKey?: string | null
  installedRelease?: string | null
  autoUpdate?: boolean
  /**
   * A dependency reconcile that still has to happen, named by the release that needs it.
   *
   * Windows holds an exclusive handle on every native addon a running process has loaded,
   * so an update that changes dependencies cannot replace them from inside the agent that
   * is using them. The install is deferred to the next start instead, when the process
   * holding those files no longer exists.
   */
  pendingInstall?: string | null
  /**
   * Fingerprint of the dependency files *as shipped* in the last release installed.
   *
   * Compared against the same files in the next bundle, which is the only comparison
   * that answers "did this release change dependencies". Comparing against the copies on
   * disk does not: `pnpm install` rewrites the lockfile locally, so every update would
   * look like a dependency change and reinstall for nothing.
   */
  depFingerprint?: string | null
  /**
   * Optional workloads this machine has chosen to run.
   *
   * Kept here rather than in package.json because an update replaces package.json, which
   * would silently strip them.
   */
  enabledWorkloads?: ('ml' | 'browser')[]
}

export function loadConfig(): AgentConfig | null {
  if (!existsSync(CONFIG_PATH)) return null
  return JSON.parse(readFileSync(CONFIG_PATH, 'utf8')) as AgentConfig
}

export function saveConfig(c: AgentConfig): void {
  mkdirSync(AGENT_HOME, { recursive: true, mode: 0o700 })
  writeFileSync(CONFIG_PATH, JSON.stringify(c, null, 2) + '\n', { mode: 0o600 })
}

/**
 * Forget which network this computer belongs to.
 *
 * The keypair in agent.key is deliberately left alone: it is this machine's identity,
 * not its membership, and keeping it means a machine that rejoins the same network is
 * recognisably the same machine. Only the enrolment goes — server address, host id, and
 * the release key pinned at pairing, which must not survive into a different network.
 */
export function clearConfig(): void {
  if (existsSync(CONFIG_PATH)) unlinkSync(CONFIG_PATH)
}

/**
 * Pause is a local file, deliberately.
 *
 * The owner's kill switch has to work when the control service is unreachable, so
 * it must not be a server round-trip.
 */
export const isPaused = (): boolean => existsSync(PAUSE_PATH)

export function setPaused(paused: boolean): void {
  mkdirSync(AGENT_HOME, { recursive: true, mode: 0o700 })
  if (paused) writeFileSync(PAUSE_PATH, new Date().toISOString(), { mode: 0o600 })
  else if (existsSync(PAUSE_PATH)) unlinkSync(PAUSE_PATH)
}
