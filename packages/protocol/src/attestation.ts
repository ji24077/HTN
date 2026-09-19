import { createHash, createPublicKey, sign, verify } from 'node:crypto'
import type { KeyObject } from 'node:crypto'

/**
 * The exact bytes a host signs when it reports a result.
 *
 * Field order is fixed and must never change without a version bump — both sides
 * build this string independently, and a mismatch reads as a forged result.
 */
export function attestationPayload(a: {
  taskId: string
  attempt: number
  hostId: string
  outputHash: string
  startedAt: string
  finishedAt: string
}): Buffer {
  return Buffer.from(
    `dwp-attest/v1\n${a.taskId}\n${a.attempt}\n${a.hostId}\n${a.outputHash}\n${a.startedAt}\n${a.finishedAt}\n`,
    'utf8',
  )
}

export function hashOutput(output: unknown): string {
  return createHash('sha256').update(JSON.stringify(output ?? null)).digest('hex')
}

export function signAttestation(privateKey: KeyObject, a: Parameters<typeof attestationPayload>[0]): string {
  return sign(null, attestationPayload(a), privateKey).toString('base64url')
}

export function verifyAttestation(
  publicKeySpkiB64: string,
  signatureB64Url: string,
  a: Parameters<typeof attestationPayload>[0],
): boolean {
  try {
    const key = createPublicKey({ key: Buffer.from(publicKeySpkiB64, 'base64'), format: 'der', type: 'spki' })
    return verify(null, attestationPayload(a), key, Buffer.from(signatureB64Url, 'base64url'))
  } catch {
    return false
  }
}
