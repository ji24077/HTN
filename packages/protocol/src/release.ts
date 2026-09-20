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

const BinaryArtifact = z.object({
  target: z.string().min(1),
  file: z.string().regex(/^[A-Za-z0-9._-]+$/),
  sha256: z.string().regex(/^[a-f0-9]{64}$/),
})

export const SignedBinaryRelease = SignedRelease.extend({
  binaries: z.array(BinaryArtifact).min(1),
  apps: z.array(BinaryArtifact).default([]),
})
export type SignedBinaryRelease = z.infer<typeof SignedBinaryRelease>

/** The same signed set used by build-binaries, including desktop app archives. */
export function binaryReleasePayload(
  binaries: readonly { target: string; sha256: string }[],
  apps: readonly { target: string; sha256: string }[] = [],
): Buffer {
  return Buffer.from(JSON.stringify([
    ...binaries.map(b => `${b.target}:${b.sha256}`),
    ...apps.map(a => `app:${a.target}:${a.sha256}`),
  ].sort()))
}

/** A signed manifest authenticates the entries only after their digest is checked. */
export function verifyBinaryRelease(value: unknown, pinnedPublicKey: string): boolean {
  const parsed = SignedBinaryRelease.safeParse(value)
  if (!parsed.success) return false
  const release = parsed.data
  if (!verifyRelease(release, pinnedPublicKey)) return false
  if (new Set(release.binaries.map(b => b.target)).size !== release.binaries.length) return false
  if (new Set(release.apps.map(a => a.target)).size !== release.apps.length) return false
  return hashBytes(binaryReleasePayload(release.binaries, release.apps)) === release.manifest.sha256
}

export function generateReleaseKey(): { privateKeyPem: string; publicKeySpki: string } {
  const { privateKey } = generateKeyPairSync('ed25519')
  return {
    privateKeyPem: privateKey.export({ type: 'pkcs8', format: 'pem' }).toString(),
    publicKeySpki: createPublicKey(privateKey).export({ type: 'spki', format: 'der' }).toString('base64'),
  }
}
