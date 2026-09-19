/**
 * Package the agent, sign it, and make it available to every connected computer.
 *
 *   node scripts/publish-release.ts ["what changed"]
 *
 * Replaces copying a file to each machine by hand. Agents notice the new version on
 * their next heartbeat and can install it themselves, having verified it was signed by
 * this machine's release key rather than merely served by the server.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync, chmodSync, statSync } from 'node:fs'
import { createPrivateKey } from 'node:crypto'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { hashBytes, signRelease, generateReleaseKey, type ReleaseManifest } from '@dwp/protocol'

const RELEASES = 'releases'
const KEY_DIR = process.env.DWP_HOME ?? join(homedir(), '.dwp')
const KEY_PATH = join(KEY_DIR, 'release.key')
const notes = process.argv[2] ?? ''

/**
 * The release signing key never leaves this machine and is not the server's.
 *
 * That separation is the whole point: someone who takes over the server still cannot
 * sign a release, and agents install nothing unsigned.
 */
function loadOrCreateKey(): ReturnType<typeof createPrivateKey> {
  mkdirSync(KEY_DIR, { recursive: true, mode: 0o700 })
  if (!existsSync(KEY_PATH)) {
    const { privateKeyPem, publicKeySpki } = generateReleaseKey()
    writeFileSync(KEY_PATH, privateKeyPem, { mode: 0o600 })
    console.log(`\n  Created a release signing key at ${KEY_PATH}`)
    console.log(`  Public key: ${publicKeySpki.slice(0, 32)}…`)
    console.log(`  Back this up. Losing it means every agent must re-pair to trust new releases.\n`)
  }
  const mode = statSync(KEY_PATH).mode & 0o777
  if (mode & 0o077) throw new Error(`${KEY_PATH} is mode ${mode.toString(8)}; expected 600`)
  return createPrivateKey(readFileSync(KEY_PATH, 'utf8'))
}

const privateKey = loadOrCreateKey()
mkdirSync(RELEASES, { recursive: true })

/**
 * Only what an agent needs to run.
 *
 * The control service, the simulator, the docs and the fixtures all stay behind: a
 * smaller bundle is a smaller thing to verify, a faster thing to ship, and it keeps
 * server-side code off machines that have no business running it.
 */
const tmp = join(RELEASES, '.staging.tar.gz')
execFileSync('tar', [
  '--exclude', 'node_modules',
  '--exclude', '.git',
  '--exclude', '.dwp',
  '--exclude', '.DS_Store',
  '-czf', tmp,
  'package.json', 'pnpm-workspace.yaml', 'pnpm-lock.yaml', 'tsconfig.json',
  'packages/protocol', 'packages/agent', 'scripts/join.sh',
], { stdio: 'pipe' })

const bytes = readFileSync(tmp)
const sha256 = hashBytes(bytes)
const agentPkg = JSON.parse(readFileSync('packages/agent/package.json', 'utf8')) as { version: string }
const version = `${agentPkg.version}+${sha256.slice(0, 8)}`

const manifest: ReleaseManifest = {
  version,
  sha256,
  bytes: bytes.length,
  createdAt: new Date().toISOString(),
  notes,
}
const signed = signRelease(privateKey, manifest)

writeFileSync(join(RELEASES, sha256), bytes)
writeFileSync(join(RELEASES, 'latest.json'), JSON.stringify(signed, null, 2) + '\n')
chmodSync(join(RELEASES, sha256), 0o644)
execFileSync('rm', ['-f', tmp])

console.log(`\n  Published ${version}`)
console.log(`    size   ${(bytes.length / 1024).toFixed(0)} KB`)
console.log(`    sha256 ${sha256.slice(0, 24)}…`)
if (notes) console.log(`    notes  ${notes}`)
console.log(`\n  Connected computers will pick this up automatically.`)
console.log(`  Anyone can also apply it now with:  pnpm agent update\n`)
