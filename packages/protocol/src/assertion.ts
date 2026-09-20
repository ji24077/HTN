import { createPublicKey, sign, verify } from 'node:crypto'
import type { KeyObject } from 'node:crypto'

/**
 * Host connection credentials: a short-lived EdDSA-signed assertion, minted fresh
 * for every connect attempt. No long-lived bearer token exists on the wire, and the
 * control database holds only public keys — a database read cannot impersonate a host.
 */
const TTL_SECONDS = 120
const MAX_SKEW_SECONDS = 120

type Claims = { iss: string; aud: string; iat: number; exp: number; jti: string }

const b64u = (b: Buffer) => b.toString('base64url')
const json = (o: unknown) => Buffer.from(JSON.stringify(o), 'utf8')

export function mintAssertion(hostId: string, privateKey: KeyObject, audience = 'dwp-control'): string {
  const iat = Math.floor(Date.now() / 1000)
  const claims: Claims = { iss: hostId, aud: audience, iat, exp: iat + TTL_SECONDS, jti: crypto.randomUUID() }
  const signingInput = `${b64u(json({ alg: 'EdDSA', typ: 'JWT' }))}.${b64u(json(claims))}`
  return `${signingInput}.${b64u(sign(null, Buffer.from(signingInput, 'utf8'), privateKey))}`
}

export type AssertionResult =
  | { ok: true; hostId: string; jti: string }
  | { ok: false; reason: string }

/**
 * Verify an assertion against a host's stored public key.
 *
 * The caller supplies `seenJti` so replay rejection stays where the state lives.
 * Every failure path returns a reason rather than throwing — these are logged as
 * auth events, not crashes.
 */
export function verifyAssertion(
  token: string,
  lookupPublicKey: (hostId: string) => string | undefined,
  seenJti: (jti: string) => boolean,
  audience = 'dwp-control',
): AssertionResult {
  const parts = token.split('.')
  if (parts.length !== 3) return { ok: false, reason: 'malformed' }
  const [h, p, s] = parts as [string, string, string]

  let header: { alg?: string; typ?: string }
  let claims: Claims
  try {
    header = JSON.parse(Buffer.from(h, 'base64url').toString('utf8'))
    claims = JSON.parse(Buffer.from(p, 'base64url').toString('utf8'))
  } catch {
    return { ok: false, reason: 'undecodable' }
  }

  // Pin the algorithm before touching the signature: never let the token choose.
  if (header.alg !== 'EdDSA' || header.typ !== 'JWT') return { ok: false, reason: 'bad-header' }
  if (claims.aud !== audience) return { ok: false, reason: 'bad-audience' }
  if (typeof claims.iss !== 'string' || typeof claims.jti !== 'string') return { ok: false, reason: 'bad-claims' }

  const now = Math.floor(Date.now() / 1000)
  if (claims.exp < now) return { ok: false, reason: 'expired' }
  if (claims.iat > now + MAX_SKEW_SECONDS) return { ok: false, reason: 'future-iat' }
  if (claims.exp - claims.iat > TTL_SECONDS + MAX_SKEW_SECONDS) return { ok: false, reason: 'ttl-too-long' }

  const spki = lookupPublicKey(claims.iss)
  if (!spki) return { ok: false, reason: 'unknown-host' }

  let okSig = false
  try {
    const key = createPublicKey({ key: Buffer.from(spki, 'base64'), format: 'der', type: 'spki' })
    okSig = verify(null, Buffer.from(`${h}.${p}`, 'utf8'), key, Buffer.from(s, 'base64url'))
  } catch {
    return { ok: false, reason: 'bad-key' }
  }
  if (!okSig) return { ok: false, reason: 'bad-signature' }
  if (seenJti(claims.jti)) return { ok: false, reason: 'replayed' }

  return { ok: true, hostId: claims.iss, jti: claims.jti }
}
