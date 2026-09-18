import { homedir } from 'node:os'
import { join } from 'node:path'

/** DWP_HOME lets several agents coexist on one machine for local testing. */
export const AGENT_HOME = process.env.DWP_HOME ?? join(homedir(), '.dwp')
export const KEY_PATH = join(AGENT_HOME, 'agent.key')
export const CONFIG_PATH = join(AGENT_HOME, 'config.json')
export const PAUSE_PATH = join(AGENT_HOME, 'paused')
export const AGENT_VERSION = '0.1.0'
