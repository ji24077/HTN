import { execFile } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { chromium } from 'playwright'

const exec = promisify(execFile)

/**
 * Verify the architecture's "no exposed CDP" claim on this machine rather than trusting it.
 *
 * Playwright is documented to drive Chromium over a stdio pipe, but a documented
 * default is not evidence: this reads the live process table and fails loudly if a
 * TCP debugging port is present.
 */
async function assertNoDebugPort(): Promise<{ checked: boolean; pipeFlagSeen: boolean }> {
  if (process.platform === 'win32') return { checked: false, pipeFlagSeen: false }
  const { stdout } = await exec('ps', ['-ax', '-o', 'args='], { maxBuffer: 8 * 1024 * 1024 })
  const chromeLines = stdout
    .split('\n')
    .filter(l => /Chromium|chrome|headless_shell/i.test(l) && /--user-data-dir/.test(l))
  const offender = chromeLines.find(l => /--remote-debugging-port(=|\s)/.test(l))
  if (offender) {
    throw new Error(`Chromium is exposing a TCP debugging port. Refusing to continue.\n  ${offender.slice(0, 200)}`)
  }
  return { checked: true, pipeFlagSeen: chromeLines.some(l => /--remote-debugging-pipe/.test(l)) }
}

export async function browserProbe(url: string): Promise<void> {
  const target = new URL(url)
  if (target.protocol !== 'http:' && target.protocol !== 'https:') {
    throw new Error(`refusing non-HTTP target: ${target.protocol}`)
  }

  // A profile directory that has never seen the owner's personal browser, deleted on exit.
  const profile = await mkdtemp(join(tmpdir(), 'dwp-browser-'))
  console.log(`[browser] ephemeral profile ${profile}`)

  // launchPersistentContext, not launch: this pins the profile to a directory we own
  // and delete, rather than a Playwright-managed temp dir we only hope is cleaned up.
  const context = await chromium.launchPersistentContext(profile, {
    headless: true,
    viewport: { width: 1280, height: 800 },
  })
  try {
    const debug = await assertNoDebugPort()
    console.log(
      `[browser] debug-port check: ${debug.checked ? 'passed' : 'skipped (win32)'}` +
        `${debug.pipeFlagSeen ? ' - driven over --remote-debugging-pipe' : ''}`,
    )

    const page = await context.newPage()
    const response = await page.goto(target.toString(), { waitUntil: 'domcontentloaded', timeout: 30_000 })
    const body = await page.evaluate(() => document.body.innerText)

    console.log(`\n[browser] GET ${target}  ->  ${response?.status()}`)
    try {
      const seen = JSON.parse(body) as { observedIp?: string; userAgent?: string }
      console.log(`\n  The site observed this egress address: ${seen.observedIp}`)
      console.log(`  User agent: ${seen.userAgent}\n`)
      console.log('  Compare that address with this host public IP. If they match, the page was')
      console.log('  fetched by THIS machine - which is the whole point of a remote browser session.')
    } catch {
      console.log(body.slice(0, 500))
    }
  } finally {
    await context.close()
    await rm(profile, { recursive: true, force: true })
    console.log('[browser] profile deleted')
  }
}
