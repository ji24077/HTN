import { createHash } from 'node:crypto'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

/**
 * Content-addressed artifact store.
 *
 * Files live in fixtures/ named by their own SHA-256. A request can only ask for a hash,
 * so there is no path to traverse and nothing to guess a filename for — and a host can
 * verify what it received is exactly what the job named.
 */
const DIR = process.env.DWP_FIXTURES_DIR ?? 'fixtures'
const HEX64 = /^[0-9a-f]{64}$/

export type Artifact = { bytes: Buffer; sha256: string }

export function readArtifact(sha256: string): Artifact | null {
  if (!HEX64.test(sha256)) return null
  const path = join(DIR, sha256)
  if (!existsSync(path) || !statSync(path).isFile()) return null

  const bytes = readFileSync(path)
  // Cheap insurance against a corrupted or swapped file on disk: never serve bytes whose
  // hash does not match the name they are served under.
  if (createHash('sha256').update(bytes).digest('hex') !== sha256) return null
  return { bytes, sha256 }
}

export function manifest(): Record<string, unknown> | null {
  const path = join(DIR, 'manifest.json')
  if (!existsSync(path)) return null
  try { return JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown> } catch { return null }
}
