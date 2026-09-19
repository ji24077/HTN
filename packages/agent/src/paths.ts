import { homedir } from 'node:os'
import { join } from 'node:path'

/** DWP_HOME lets several agents coexist on one machine for local testing. */
export const AGENT_HOME = process.env.DWP_HOME ?? join(homedir(), '.dwp')
export const KEY_PATH = join(AGENT_HOME, 'agent.key')
export const CONFIG_PATH = join(AGENT_HOME, 'config.json')
export const PAUSE_PATH = join(AGENT_HOME, 'paused')
export const AGENT_VERSION = '0.2.0'

import { dirname, join as joinPath } from 'node:path'
import { fileURLToPath } from 'node:url'

/** Where this agent is installed — three levels up from packages/agent/src. */
export function installRoot(): string {
  return joinPath(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
}
