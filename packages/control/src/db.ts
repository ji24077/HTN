import { readdir, readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import pg from 'pg'
import { config } from './config.ts'

export const pool = new pg.Pool({ connectionString: config.databaseUrl, max: 10 })

export async function migrate(): Promise<void> {
  const dir = join(dirname(fileURLToPath(import.meta.url)), 'migrations')
  await pool.query(`create table if not exists _migrations (name text primary key, applied_at timestamptz default now())`)
  const applied = new Set((await pool.query<{ name: string }>(`select name from _migrations`)).rows.map(r => r.name))
  for (const name of (await readdir(dir)).filter(f => f.endsWith('.sql')).sort()) {
    if (applied.has(name)) continue
    const sql = await readFile(join(dir, name), 'utf8')
    const client = await pool.connect()
    try {
      await client.query('begin')
      await client.query(sql)
      await client.query(`insert into _migrations(name) values ($1)`, [name])
      await client.query('commit')
      console.log(`[db] applied ${name}`)
    } catch (err) {
      await client.query('rollback')
      throw err
    } finally {
      client.release()
    }
  }
}

/** Run `fn` inside one transaction. Rolls back on any throw. */
export async function tx<T>(fn: (c: pg.PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect()
  try {
    await client.query('begin')
    const out = await fn(client)
    await client.query('commit')
    return out
  } catch (err) {
    await client.query('rollback')
    throw err
  } finally {
    client.release()
  }
}
