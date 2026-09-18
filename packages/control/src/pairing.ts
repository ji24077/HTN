import { randomInt } from 'node:crypto'
import { pool, tx } from './db.ts'
import { config } from './config.ts'
import { sha256 } from './auth.ts'
import { record } from './events.ts'

// Crockford base32 minus I, L, O, U — no character a person can misread aloud.
const ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'

function newCode(): string {
  const pick = () => Array.from({ length: 4 }, () => ALPHABET[randomInt(ALPHABET.length)]!).join('')
  return `${pick()}-${pick()}`
}

export async function issuePairCode(userId: string, label: string): Promise<{ code: string; expiresAt: Date }> {
  const code = newCode()
  const expiresAt = new Date(Date.now() + config.pairCodeTtlMs)
  await pool.query(`insert into pair_codes(code_hash, user_id, label, expires_at) values ($1,$2,$3,$4)`,
    [sha256(code), userId, label, expiresAt])
  return { code, expiresAt }
}

export type PairResult =
  | { ok: true; hostId: string; label: string }
  | { ok: false; reason: 'invalid-or-expired' }

/**
 * Consume a pairing code and bind exactly one public key to a new host row.
 *
 * Consumption and host creation share a transaction, so a code cannot enroll two
 * hosts even if two agents race with the same code.
 */
export async function redeemPairCode(code: string, publicKey: string, label?: string): Promise<PairResult> {
  return tx(async client => {
    const { rows } = await client.query<{ user_id: string; label: string }>(
      `update pair_codes set consumed_at = now()
       where code_hash = $1 and consumed_at is null and expires_at > now()
       returning user_id, label`,
      [sha256(code)],
    )
    const claim = rows[0]
    if (!claim) return { ok: false, reason: 'invalid-or-expired' } as const

    const hostLabel = label?.trim() || claim.label
    const host = await client.query<{ id: string }>(
      `insert into hosts(owner_id, label, public_key) values ($1,$2,$3) returning id`,
      [claim.user_id, hostLabel, publicKey],
    )
    const hostId = host.rows[0]!.id
    await client.query(
      `insert into run_events(host_id, actor, category, type, payload) values ($1,'user','auth','host.paired',$2)`,
      [hostId, JSON.stringify({ label: hostLabel })],
    )
    return { ok: true, hostId, label: hostLabel } as const
  })
}

export async function revokeHost(hostId: string, ownerId: string): Promise<boolean> {
  const { rowCount } = await pool.query(
    `update hosts set revoked_at = now(), online = false where id = $1 and owner_id = $2 and revoked_at is null`,
    [hostId, ownerId],
  )
  if (rowCount) await record({ hostId, actor: 'user', category: 'auth', type: 'host.revoked' })
  return Boolean(rowCount)
}
