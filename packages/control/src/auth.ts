import { randomBytes, scrypt as scryptCb, timingSafeEqual, createHash } from 'node:crypto'
import { promisify } from 'node:util'
import { pool } from './db.ts'
import { config } from './config.ts'

const scrypt = promisify(scryptCb) as (p: string, s: Buffer, l: number) => Promise<Buffer>
const SESSION_TTL_MS = 12 * 60 * 60 * 1000

export async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16)
  const key = await scrypt(password, salt, 64)
  return `scrypt$${salt.toString('base64')}$${key.toString('base64')}`
}

export async function checkPassword(password: string, stored: string): Promise<boolean> {
  const [scheme, saltB64, keyB64] = stored.split('$')
  if (scheme !== 'scrypt' || !saltB64 || !keyB64) return false
  const expected = Buffer.from(keyB64, 'base64')
  const actual = await scrypt(password, Buffer.from(saltB64, 'base64'), expected.length)
  return expected.length === actual.length && timingSafeEqual(expected, actual)
}

export const sha256 = (s: string) => createHash('sha256').update(s).digest('hex')

export type User = { id: string; email: string; role: string }

/**
 * Seed the single operator account so a fresh clone is usable without manual SQL.
 *
 * The password in the environment is authoritative. If it no longer matches what is
 * stored, the stored one is replaced — otherwise rotating BOOTSTRAP_PASSWORD would
 * silently do nothing and the next login would fail with an unexplained 401, which is
 * exactly what happened the first time `share.ts` generated a strong password for an
 * account that already existed.
 */
export async function bootstrapUser(): Promise<User> {
  const existing = await pool.query<User & { password_hash: string }>(
    `select id, email, role, password_hash from users where email = $1`, [config.bootstrapEmail])
  const user = existing.rows[0]
  if (user) {
    if (!(await checkPassword(config.bootstrapPassword, user.password_hash))) {
      await pool.query(`update users set password_hash = $2 where id = $1`,
        [user.id, await hashPassword(config.bootstrapPassword)])
      console.log(`[auth] operator password for ${config.bootstrapEmail} updated to match the environment`)
    }
    return { id: user.id, email: user.email, role: user.role }
  }
  const { rows } = await pool.query<User>(
    `insert into users(email, password_hash, role) values ($1,$2,'admin') returning id, email, role`,
    [config.bootstrapEmail, await hashPassword(config.bootstrapPassword)],
  )
  console.log(`[auth] seeded operator account ${config.bootstrapEmail}`)
  return rows[0]!
}

export async function login(email: string, password: string): Promise<string | null> {
  const { rows } = await pool.query<{ id: string; password_hash: string }>(
    `select id, password_hash from users where email = $1`, [email],
  )
  const user = rows[0]
  if (!user || !(await checkPassword(password, user.password_hash))) return null
  const token = randomBytes(32).toString('base64url')
  await pool.query(`insert into sessions(token_hash, user_id, expires_at) values ($1,$2,now() + $3::interval)`,
    [sha256(token), user.id, `${SESSION_TTL_MS} milliseconds`])
  return token
}

export async function userForToken(token: string | undefined): Promise<User | null> {
  if (!token) return null
  const { rows } = await pool.query<User>(
    `select u.id, u.email, u.role from sessions s
     join users u on u.id = s.user_id
     where s.token_hash = $1 and s.expires_at > now()`, [sha256(token)],
  )
  return rows[0] ?? null
}
