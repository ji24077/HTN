import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

/** DWP_HOME lets several agents coexist on one machine for local testing. */
export const AGENT_HOME = process.env.DWP_HOME ?? join(homedir(), '.dwp')
export const KEY_PATH = join(AGENT_HOME, 'agent.key')
export const CONFIG_PATH = join(AGENT_HOME, 'config.json')
export const PAUSE_PATH = join(AGENT_HOME, 'paused')
/**
 * Baked in at build time rather than hand-edited.
 *
 * A compiled binary has no package.json beside it to read, and a constant maintained by
 * hand drifts from the real version silently — which is exactly what happened: this said
 * 0.2.0 while the package said 0.3.0. `bun build --define` sets it; running from source
 * falls back to the package.
 */
declare const __DWP_VERSION__: string | undefined
export const AGENT_VERSION: string =
  typeof __DWP_VERSION__ === 'string' ? __DWP_VERSION__ : readVersionFromPackage()

function readVersionFromPackage(): string {
  try {
    return (JSON.parse(
      readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
    ) as { version: string }).version
  } catch {
    return 'dev'
  }
}

import { dirname, join as joinPath } from 'node:path'
import { fileURLToPath } from 'node:url'

/** Where this agent is installed — three levels up from packages/agent/src. */
export function installRoot(): string {
  return joinPath(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
}

/**
 * True when running as a compiled single-file binary rather than from source.
 *
 * Bun serves an embedded bundle from a virtual filesystem, so the module URL points
 * there instead of at a real path on disk. Almost everything about updating and about
 * the instructions we print differs between the two.
 */
export function isCompiledBinary(): boolean {
  return import.meta.url.includes('$bunfs') || import.meta.url.includes('B:/~BUN')
}

/** How this agent is invoked, for messages people are meant to copy. */
export function invocation(): string {
  return isCompiledBinary() ? process.execPath : 'pnpm agent'
}
