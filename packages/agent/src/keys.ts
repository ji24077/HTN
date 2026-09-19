import { generateKeyPairSync, createPrivateKey, createPublicKey } from 'node:crypto'
import type { KeyObject } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { AGENT_HOME, KEY_PATH } from './paths.ts'
import { restrictToCurrentUser } from './winacl.ts'

const IS_WINDOWS = process.platform === 'win32'

/**
 * The host's identity never leaves this machine.
 *
 * The control service stores only the public half, which is why a database read
 * there cannot impersonate this host or forge its results.
 */
export function ensureKeypair(): { privateKey: KeyObject; publicKeySpki: string } {
  mkdirSync(AGENT_HOME, { recursive: true, mode: 0o700 })

  if (!existsSync(KEY_PATH)) {
    const { privateKey } = generateKeyPairSync('ed25519')
    writeFileSync(KEY_PATH, privateKey.export({ type: 'pkcs8', format: 'pem' }), { mode: 0o600 })
    if (IS_WINDOWS) {
      // Windows ignores the mode above, so the equivalent has to be applied explicitly.
      const acl = restrictToCurrentUser(KEY_PATH)
      console.log(acl.ok
        ? `[agent] key protected — ${acl.detail}`
        : `[agent] WARNING: could not restrict access to ${KEY_PATH} — ${acl.detail}\n` +
          `         Anyone who can read that file can act as this host. Fix with:\n` +
          `         icacls "${KEY_PATH}" /inheritance:r /grant:r "%USERNAME%":F`)
    } else {
      chmodSync(KEY_PATH, 0o600)
    }
  }

  // Refuse to run with a world- or group-readable key rather than warn about it.
  //
  // Windows is exempt because the test would be meaningless there, not because the key
  // matters less: it has no mode bits, so Node reports 0o666 for every writable file and
  // this check would reject a perfectly well-protected key on every Windows host. The
  // control for those is the ACL applied at creation above.
  if (!IS_WINDOWS) {
    const mode = statSync(KEY_PATH).mode & 0o777
    if (mode & 0o077) {
      throw new Error(`${KEY_PATH} has mode ${mode.toString(8)}; expected 600. Run: chmod 600 ${KEY_PATH}`)
    }
  }

  const privateKey = createPrivateKey(readFileSync(KEY_PATH, 'utf8'))
  const publicKeySpki = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')
  return { privateKey, publicKeySpki }
}
