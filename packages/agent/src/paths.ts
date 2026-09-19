import { closeSync, openSync, readFileSync, readSync } from 'node:fs'
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

/**
 * Which kind of Windows executable this is: 2 = no console, 3 = console.
 *
 * Read out of our own PE header rather than baked in at build time, because the two
 * Windows builds are the same compiled bytes with two header bytes changed — a compile-
 * time constant would be identical in both and therefore useless. Asking the file what
 * it is cannot drift from what it is.
 *
 * This decides which build an update downloads. Without it, a windowless app would take
 * the console build on its first update and sprout a command prompt it never had, which
 * would read as the update having broken it.
 */
export function windowsSubsystem(): number | null {
  if (process.platform !== 'win32') return null
  try {
    const fd = openSync(process.execPath, 'r')
    try {
      const head = Buffer.alloc(1024)
      readSync(fd, head, 0, head.length, 0)
      const peOffset = head.readUInt32LE(0x3c)
      if (peOffset + 92 > head.length) return null
      if (head.toString('ascii', peOffset, peOffset + 4) !== 'PE\u0000\u0000') return null
      return head.readUInt16LE(peOffset + 24 + 68)
    } finally {
      closeSync(fd)
    }
  } catch {
    return null
  }
}
