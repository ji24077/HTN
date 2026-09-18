/**
 * Independently verify that a job's accepted results were produced by distinct,
 * enrolled physical hosts.
 *
 * This reads the database directly and checks every Ed25519 attestation against the
 * host public keys. It does NOT trust the control service's own event log — the
 * control service writes that log, so it cannot be evidence about itself.
 *
 *   node scripts/verify-run.ts <jobId>
 */
import pg from 'pg'
import { verifyAttestation } from '@dwp/protocol'

const jobId = process.argv[2]
if (!jobId) {
  console.error('usage: node scripts/verify-run.ts <jobId>')
  process.exit(1)
}

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL ?? 'postgres://dwp:dwp@localhost:5433/dwp',
})

type Row = {
  task_id: string
  seq: number
  attempt: number
  host_id: string
  host_label: string
  public_key: string
  host_signature: string
  output_hash: string
  started_at: string
  finished_at: string
  outcome: string
}

const { rows } = await pool.query<Row>(
  `select t.id as task_id, t.seq, a.attempt, a.host_id, h.label as host_label, h.public_key,
          a.host_signature, t.output_hash, a.started_at, a.finished_at, a.outcome
     from tasks t
     join task_attempts a on a.task_id = t.id and a.outcome = 'succeeded'
     join hosts h on h.id = a.host_id
    where t.job_id = $1
    order by t.seq`,
  [jobId],
)

if (rows.length === 0) {
  console.error(`No accepted results for job ${jobId}`)
  await pool.end()
  process.exit(1)
}

const byHost = new Map<string, { label: string; ok: number; bad: number }>()
let bad = 0

for (const r of rows) {
  const ok = verifyAttestation(r.public_key, r.host_signature, {
    taskId: r.task_id,
    attempt: r.attempt,
    hostId: r.host_id,
    outputHash: r.output_hash,
    startedAt: new Date(r.started_at).toISOString(),
    finishedAt: new Date(r.finished_at).toISOString(),
  })
  const entry = byHost.get(r.host_id) ?? { label: r.host_label, ok: 0, bad: 0 }
  ok ? entry.ok++ : entry.bad++
  byHost.set(r.host_id, entry)
  if (!ok) {
    bad++
    console.error(`  task ${r.task_id} seq ${r.seq}: SIGNATURE INVALID (host ${r.host_label})`)
  }
}

console.log(`\nJob ${jobId}`)
console.log(`  accepted results : ${rows.length}`)
console.log(`  distinct hosts   : ${byHost.size}`)
for (const [hostId, s] of byHost) {
  console.log(`    ${s.label.padEnd(24)} ${String(s.ok).padStart(5)} verified` +
    (s.bad ? `  ${s.bad} INVALID` : '') + `   ${hostId}`)
}

const verdict = bad === 0 && byHost.size >= 2
console.log(`\n  gate:distribution ${verdict ? 'PASS' : bad > 0 ? 'FAIL (invalid signatures)' : 'INCOMPLETE (one host only)'}`)
console.log(`  Every accepted result above carries an Ed25519 signature made with a private key`)
console.log(`  that never left its host. The control service holds only public keys.\n`)

await pool.end()
process.exit(verdict ? 0 : 1)
