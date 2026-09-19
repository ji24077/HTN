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
import { existsSync, mkdirSync, readFileSync, writeFileSync, chmodSync, rmSync, statSync } from 'node:fs'
import { createPrivateKey } from 'node:crypto'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { hashBytes, signRelease, generateReleaseKey, type ReleaseManifest } from '@dwp/protocol'
// One implementation of the Windows key ACL, shared with the agent rather than copied:
// two versions of a control that protects a signing key is one version too many.
import { restrictToCurrentUser } from '../packages/agent/src/winacl.ts'
import { IS_WINDOWS } from './lib/platform.ts'

// Must agree with the control service, which reads the same variable.
const RELEASES = process.env.DWP_RELEASES_DIR ?? 'releases'
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
    if (IS_WINDOWS) {
      const acl = restrictToCurrentUser(KEY_PATH)
      if (!acl.ok) {
        console.log(`\n  WARNING: could not restrict access to ${KEY_PATH} — ${acl.detail}`)
        console.log(`  Anyone who can read it can sign releases every agent will install.`)
      }
    }
    console.log(`\n  Created a release signing key at ${KEY_PATH}`)
    console.log(`  Public key: ${publicKeySpki.slice(0, 32)}…`)
    console.log(`  Back this up. Losing it means every agent must re-pair to trust new releases.\n`)
  }
  // Windows reports 0o666 for every writable file, so this test would reject a
  // well-protected key there. Its ACL is applied at creation instead — same split as
  // the agent's identity key in packages/agent/src/keys.ts.
  if (!IS_WINDOWS) {
    const mode = statSync(KEY_PATH).mode & 0o777
    if (mode & 0o077) throw new Error(`${KEY_PATH} is mode ${mode.toString(8)}; expected 600`)
  }
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
  'packages/protocol', 'packages/agent', 'scripts/join.sh', 'scripts/join.ps1',
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
rmSync(tmp, { force: true })

console.log(`\n  Published ${version}`)
console.log(`    size   ${(bytes.length / 1024).toFixed(0)} KB`)
console.log(`    sha256 ${sha256.slice(0, 24)}…`)
if (notes) console.log(`    notes  ${notes}`)
console.log(`\n  Connected computers will pick this up automatically.`)
console.log(`  Anyone can also apply it now with:  pnpm agent update\n`)
