import { generateKeyPairSync, createPrivateKey, createPublicKey } from 'node:crypto'
import type { KeyObject } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { AGENT_HOME, KEY_PATH } from './paths.ts'

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
    chmodSync(KEY_PATH, 0o600)
  }

  // Refuse to run with a world- or group-readable key rather than warn about it.
  const mode = statSync(KEY_PATH).mode & 0o777
  if (mode & 0o077) {
    throw new Error(`${KEY_PATH} has mode ${mode.toString(8)}; expected 600. Run: chmod 600 ${KEY_PATH}`)
  }

  const privateKey = createPrivateKey(readFileSync(KEY_PATH, 'utf8'))
  const publicKeySpki = createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64')
  return { privateKey, publicKeySpki }
}
