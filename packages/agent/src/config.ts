import { existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs'
import { AGENT_HOME, CONFIG_PATH, PAUSE_PATH } from './paths.ts'

export type AgentConfig = {
  server: string
  wsUrl: string
  hostId: string
  label: string
  allowCompute: boolean
  allowBrowser: boolean
  maxConcurrency: number
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
