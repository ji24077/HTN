import { execFileSync } from 'node:child_process'

for (let i = 0; i < 40; i++) {
  try {
    execFileSync('docker', ['exec', 'dwp-db', 'pg_isready', '-U', 'dwp', '-d', 'dwp'], { stdio: 'ignore' })
    console.log('[db] ready')
    process.exit(0)
  } catch {
    await new Promise(r => setTimeout(r, 500))
  }
}
console.error('[db] never became ready')
process.exit(1)
