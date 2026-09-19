/**
 * Does the app's page actually parse?
 *
 * The page is assembled inside a TypeScript template literal, which is a small minefield:
 * a `\n` is consumed at build time and emitted as a real line break, and a literal
 * `</script>` in a comment ends the script block early. Both produce a page whose script
 * never parses -- and neither shows up as an error anywhere a user or a typecheck can
 * see. The app simply sits on "starting…" forever, because no script ran to replace the
 * placeholder. That cost an evening once; this makes it cost one command.
 */
import { spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'

const home = mkdtempSync(join(tmpdir(), 'dwp-guicheck-'))
const child = spawn(process.execPath, ['packages/agent/src/index.ts', 'gui', '--hidden'],
  { env: { ...process.env, DWP_HOME: home }, stdio: 'ignore', detached: true })
child.unref()

const sleep = (ms: number): Promise<void> => new Promise(r => setTimeout(r, ms))
let lock: { port: number; token: string } | null = null
for (let i = 0; i < 30 && !lock; i++) {
  await sleep(500)
  try { lock = JSON.parse(readFileSync(join(home, 'gui.json'), 'utf8')) } catch { /* not yet */ }
}
if (!lock) { try { process.kill(-child.pid!) } catch {} ; console.error('\n  the app never started\n'); process.exit(1) }

const html = await (await fetch(`http://127.0.0.1:${lock.port}/${lock.token}/`)).text()
try { process.kill(-child.pid!) } catch { /* already gone */ }

const script = /<script>([\s\S]*?)<\/script>/.exec(html)?.[1]
if (!script) { console.error('\n  no script block in the page\n'); process.exit(1) }

const file = join(home, 'page.js')
writeFileSync(file, script)
try {
  execFileSync(process.execPath, ['--check', file], { stdio: 'pipe' })
  console.log(`\n  page script parses — ${script.length} bytes\n`)
} catch (err) {
  const detail = (err as { stderr?: Buffer }).stderr?.toString() ?? String(err)
  console.error(`\n  the page script does not parse:\n\n${detail}\n`)
  process.exit(1)
}
