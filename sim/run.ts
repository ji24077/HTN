/**
 * Run the network simulation suite.
 *
 *   node sim/run.ts                 every scenario
 *   node sim/run.ts flaky-wifi      one scenario
 *   node sim/run.ts --list          what exists and why it matters
 *
 * Each scenario gets a fresh database, a fresh control service, fresh agent processes,
 * and its own simulated network. Logs land in .dwp/sim-logs/<scenario>/ so a failure can
 * be read afterwards with: node scripts/watch-log.ts --dir .dwp/sim-logs/<scenario>
 */
import { rmSync } from 'node:fs'
import { startHarness } from './harness.ts'
import { scenarios, type Ctx, type Scenario } from './scenarios.ts'

const args = process.argv.slice(2)
const GREEN = '\x1b[32m', RED = '\x1b[31m', DIM = '\x1b[2m', BOLD = '\x1b[1m', YEL = '\x1b[33m', RESET = '\x1b[0m'

if (args.includes('--list')) {
  console.log(`\n${BOLD}Network scenarios${RESET}\n`)
  for (const s of scenarios) {
    console.log(`  ${BOLD}${s.name}${RESET}\n    ${s.what}\n    ${DIM}${s.why}${RESET}\n`)
  }
  process.exit(0)
}

const wanted = args.filter(a => !a.startsWith('--'))
const selected = wanted.length > 0
  ? scenarios.filter(s => wanted.includes(s.name))
  : scenarios

if (selected.length === 0) {
  console.error(`No scenario matched: ${wanted.join(', ')}\nTry --list`)
  process.exit(1)
}

// Compressed timings. The mechanisms under test — heartbeat detection, lease expiry,
// backoff — are the same at any scale; only the clock changes, which is what makes a
// two-minute suite able to cover behaviour that takes minutes in production.
process.env.DWP_HEARTBEAT_SECONDS = '2'
process.env.DWP_OFFLINE_AFTER_SECONDS = '6'
process.env.DWP_LEASE_SECONDS = '5'

type Result = { scenario: Scenario; passed: number; failed: number; error?: string; ms: number }
const results: Result[] = []

for (const scenario of selected) {
  const logDir = `.dwp/sim-logs/${scenario.name}`
  try { rmSync(logDir, { recursive: true, force: true }) } catch {}

  console.log(`\n${BOLD}${scenario.name}${RESET}  ${DIM}${scenario.what}${RESET}`)
  const started = Date.now()
  let passed = 0, failed = 0, error: string | undefined

  const ctx: Ctx = {
    h: null as never,
    check(name, ok, detail) {
      if (ok) { passed += 1; console.log(`    ${GREEN}ok${RESET}   ${name}${detail ? `  ${DIM}${detail}${RESET}` : ''}`) }
      else { failed += 1; console.log(`    ${RED}FAIL${RESET} ${name}${detail ? `  ${RED}${detail}${RESET}` : ''}`) }
    },
    note(text) { console.log(`    ${DIM}·    ${text}${RESET}`) },
  }

  let harness: Awaited<ReturnType<typeof startHarness>> | null = null
  try {
    harness = await startHarness({ logDir })
    ;(ctx as { h: unknown }).h = harness
    await scenario.run(ctx)
  } catch (err) {
    error = err instanceof Error ? err.message : String(err)
    failed += 1
    console.log(`    ${RED}ERROR${RESET} ${error}`)
  } finally {
    if (harness) { try { await harness.teardown() } catch {} }
  }

  const ms = Date.now() - started
  results.push({ scenario, passed, failed, error, ms })
  console.log(`    ${DIM}${(ms / 1000).toFixed(1)}s · logs: ${logDir}${RESET}`)
}

// ------------------------------------------------------------------- summary

const totalPassed = results.reduce((n, r) => n + r.passed, 0)
const totalFailed = results.reduce((n, r) => n + r.failed, 0)

console.log(`\n${BOLD}${'-'.repeat(70)}${RESET}`)
for (const r of results) {
  const mark = r.failed === 0 ? `${GREEN}PASS${RESET}` : `${RED}FAIL${RESET}`
  console.log(`  ${mark}  ${r.scenario.name.padEnd(22)} ${String(r.passed).padStart(2)} ok` +
    (r.failed ? `  ${RED}${r.failed} failed${RESET}` : '') +
    `  ${DIM}${(r.ms / 1000).toFixed(1)}s${RESET}`)
}
console.log(`${BOLD}${'-'.repeat(70)}${RESET}`)
console.log(`  ${totalFailed === 0 ? GREEN + 'ALL SCENARIOS PASS' : RED + 'FAILURES PRESENT'}${RESET}` +
  `  ${totalPassed} checks passed, ${totalFailed} failed\n`)

if (totalFailed > 0) {
  console.log(`  ${YEL}To investigate a failure:${RESET}`)
  for (const r of results.filter(x => x.failed > 0)) {
    console.log(`    node scripts/watch-log.ts --dir .dwp/sim-logs/${r.scenario.name} --no-follow`)
  }
  console.log('')
}

process.exit(totalFailed === 0 ? 0 : 1)
