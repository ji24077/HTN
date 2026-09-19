import { existsSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { SignedRelease, type SignedRelease as Signed } from '@dwp/protocol'

/**
 * Where published agent releases live.
 *
 * The server only stores and serves these. It holds no signing key, so it can distribute
 * a release but cannot create one — which is what keeps "the server was compromised"
 * from becoming "every friend's laptop was compromised".
 */
const DIR = process.env.DWP_RELEASES_DIR ?? 'releases'
const HEX64 = /^[0-9a-f]{64}$/

export function latestRelease(): Signed | null {
  const path = join(DIR, 'latest.json')
  if (!existsSync(path)) return null
  try {
    return SignedRelease.parse(JSON.parse(readFileSync(path, 'utf8')))
  } catch {
    return null
  }
}

export function releaseBundle(sha256: string): Buffer | null {
  if (!HEX64.test(sha256)) return null
  const path = join(DIR, sha256)
  if (!existsSync(path) || !statSync(path).isFile()) return null
  const bytes = readFileSync(path)
  // Never serve bytes whose hash disagrees with the name they are served under.
  if (createHash('sha256').update(bytes).digest('hex') !== sha256) return null
  return bytes
}
