import { createHash, createPublicKey, generateKeyPairSync, sign, verify } from 'node:crypto'
import type { KeyObject } from 'node:crypto'
import { z } from 'zod'

/**
 * Signed agent releases.
 *
 * Shipping code to someone else's laptop is the most dangerous thing this system can do,
 * so the trust model is deliberately narrow: the bundle is signed by a key the operator
 * holds, and each agent pins that key when it pairs. The server only distributes bytes.
 *
 * The consequence that matters: a compromised control service cannot push code. It can
 * serve a bundle, but it cannot produce a signature for one, and an agent installs
 * nothing it cannot verify against the key it pinned on day one.
 */

export const ReleaseManifest = z.object({
  version: z.string(),
  sha256: z.string().length(64),
  bytes: z.number().int().positive(),
  createdAt: z.string(),
  notes: z.string().max(500).default(''),
})
export type ReleaseManifest = z.infer<typeof ReleaseManifest>

export const SignedRelease = z.object({
  manifest: ReleaseManifest,
  /** Ed25519 over the canonical manifest string, by the operator's release key. */
  signature: z.string(),
  /** base64 SPKI, so an agent can confirm it is the key it pinned. */
  publicKey: z.string(),
})
export type SignedRelease = z.infer<typeof SignedRelease>

/**
 * The exact bytes that get signed.
 *
 * Field order is fixed: both sides rebuild this independently, and any disagreement must
 * read as an invalid signature rather than as a subtly different release.
 */
export function releasePayload(m: ReleaseManifest): Buffer {
  return Buffer.from(
    `dwp-release/v1\n${m.version}\n${m.sha256}\n${m.bytes}\n${m.createdAt}\n`,
    'utf8',
  )
}

export function signRelease(privateKey: KeyObject, manifest: ReleaseManifest): SignedRelease {
  return {
    manifest,
    signature: sign(null, releasePayload(manifest), privateKey).toString('base64url'),
    publicKey: createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64'),
  }
}

export function verifyRelease(release: SignedRelease, pinnedPublicKey: string): boolean {
  // The key must be the one pinned at pairing. Trusting the key the release arrives with
  // would make the signature decorative.
  if (release.publicKey !== pinnedPublicKey) return false
  try {
    const key = createPublicKey({
      key: Buffer.from(pinnedPublicKey, 'base64'), format: 'der', type: 'spki',
    })
    return verify(null, releasePayload(release.manifest), key,
      Buffer.from(release.signature, 'base64url'))
  } catch {
    return false
  }
}

export const hashBytes = (b: Buffer): string => createHash('sha256').update(b).digest('hex')

export function generateReleaseKey(): { privateKeyPem: string; publicKeySpki: string } {
  const { privateKey } = generateKeyPairSync('ed25519')
  return {
    privateKeyPem: privateKey.export({ type: 'pkcs8', format: 'pem' }).toString(),
    publicKeySpki: createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64'),
  }
}
