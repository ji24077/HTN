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
