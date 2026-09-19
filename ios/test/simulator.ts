/**
 * The real iOS build, in the Simulator, against a real control service.
 *
 *   node ios/test/simulator.ts
 *
 * The scenario suite exercises the shared core as a macOS process, which covers the
 * protocol and every reliability path. It cannot cover the things that are only true of
 * an actual iOS build: that `os` reports `ios` and the control service's schema accepts
 * it, that the Keychain-or-file store works inside a sandbox, that ONNX Runtime's iOS
 * slice loads, and that the app pairs and starts without anyone tapping it.
 *
 * Requires: pnpm db:up, and a built app —
 *   cd ios && xcodegen generate && xcodebuild -project DWPAgent.xcodeproj -scheme DWPAgent \
 *     -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath .build-ios build
 */
import { execFileSync } from 'node:child_process'
import { existsSync, rmSync } from 'node:fs'
import { startHarness, sleep, type Harness } from '../../sim/harness.ts'

const GREEN = '\x1b[32m', RED = '\x1b[31m', DIM = '\x1b[2m', BOLD = '\x1b[1m', RESET = '\x1b[0m'
const DEVICE = process.env.DWP_SIM_DEVICE ?? 'iPhone 17 Pro'
const APP = 'ios/.build-ios/Build/Products/Debug-iphonesimulator/DWPAgent.app'

/**
 * Read the identifier out of the build rather than hardcoding it.
 *
 * `Local.xcconfig` sets the bundle id per machine, because a free Apple ID cannot
 * register one another developer already holds. A constant here would work on exactly one
 * checkout and fail confusingly everywhere else.
 */
const bundleId = (): string => {
  try {
    return execFileSync('/usr/libexec/PlistBuddy',
      ['-c', 'Print :CFBundleIdentifier', `${APP}/Info.plist`],
      { encoding: 'utf8' }).trim()
  } catch {
    return 'com.dwp.agent'
  }
}

let passed = 0, failed = 0
const check = (name: string, ok: boolean, detail = ''): void => {
  if (ok) { passed += 1; console.log(`    ${GREEN}ok${RESET}   ${name}${detail ? `  ${DIM}${detail}${RESET}` : ''}`) }
  else { failed += 1; console.log(`    ${RED}FAIL${RESET} ${name}${detail ? `  ${RED}${detail}${RESET}` : ''}`) }
}
const note = (text: string): void => console.log(`    ${DIM}·    ${text}${RESET}`)

const simctl = (args: string[]): string =>
  execFileSync('xcrun', ['simctl', ...args], { encoding: 'utf8', timeout: 300_000 }).trim()

const BUNDLE_ID = existsSync(APP) ? bundleId() : 'com.dwp.agent'

if (!existsSync(APP)) {
  console.error(`\nNo built app at ${APP}\n\n  cd ios && xcodegen generate && \\\n` +
    `    xcodebuild -project DWPAgent.xcodeproj -scheme DWPAgent \\\n` +
    `      -destination 'platform=iOS Simulator,name=${DEVICE}' -derivedDataPath .build-ios build\n`)
  process.exit(1)
}

process.env.DWP_HEARTBEAT_SECONDS = '2'
process.env.DWP_OFFLINE_AFTER_SECONDS = '6'
process.env.DWP_LEASE_SECONDS = '10'

const logDir = '.dwp/ios-logs/simulator'
try { rmSync(logDir, { recursive: true, force: true }) } catch {}

console.log(`\n${BOLD}The iOS app in the Simulator${RESET}  ${DIM}${DEVICE}${RESET}\n`)

let harness: Harness | undefined
let booted = false

try {
  // ------------------------------------------------------------------ simulator
  const devices = JSON.parse(simctl(['list', 'devices', 'available', '--json'])) as
    { devices: Record<string, { udid: string; name: string; state: string }[]> }
  const device = Object.values(devices.devices).flat().find(d => d.name === DEVICE)
  if (!device) throw new Error(`no simulator named "${DEVICE}" — try DWP_SIM_DEVICE="iPhone 17"`)

  if (device.state !== 'Booted') {
    note(`booting ${DEVICE}…`)
    simctl(['boot', device.udid])
    booted = true
  }
  simctl(['bootstatus', device.udid, '-b'])
  check('the simulator is booted', true, device.udid)

  // The control service must come up before the app is launched: the app pairs on its
  // first frame, and a refused connection there is indistinguishable from a broken one.
  harness = await startHarness({ logDir, controlPort: 8894 })
  const { code } = await (await harness.api('/hosts/pair-code', { label: 'iPhone' })).json() as { code: string }
  check('the control service issued a pairing code', Boolean(code), code)

  // A leftover install from a previous run would keep its old pairing and never use the
  // code we just issued.
  try { simctl(['uninstall', device.udid, BUNDLE_ID]) } catch {}
  simctl(['install', device.udid, APP])
  check('the app installed', true)

  // Launch arguments land in UserDefaults, which is how the app is driven without taps.
  // It pairs against the direct origin rather than through the network simulator — the
  // point here is the iOS build, not impairment, which the scenario suite already covers.
  simctl([
    'launch', device.udid, BUNDLE_ID,
    '-dwpServer', harness.directOrigin,
    '-dwpCode', code,
    '-dwpLabel', 'iPhone',
    '-dwpAutoStart', 'YES',
  ])
  note(`launched against ${harness.directOrigin}`)

  // ------------------------------------------------------------------- pairing
  await harness.waitFor('the phone pairs and connects', async () =>
    (await harness!.hosts()).some(h => h.label === 'iPhone' && h.online), 120_000)
  check('the iOS app paired and connected on its own', true)

  const { hosts } = await (await harness.api('/hosts')).json() as {
    hosts: { id: string; label: string; os: string; arch: string; cpu_model: string
             agent_version: string; adapters: string[]; total_ram_mb: number
             logical_cores: number; online: boolean }[]
  }
  const phone = hosts.find(h => h.label === 'iPhone')!

  // The reason the protocol change was needed: the zod enum gates `hello`, and an
  // unknown value means the host connects, is never recorded, and is never given work.
  check('it reports itself as an iOS host', phone.os === 'ios', `os=${phone.os}`)
  check('the control service accepted the iOS capability record',
    Boolean(phone.cpu_model) && phone.logical_cores > 0,
    `${phone.cpu_model}, ${phone.logical_cores} cores, ${phone.total_ram_mb} MB`)
  check('it advertises both adapters', 
    phone.adapters?.includes('echo') && phone.adapters?.includes('cpu_inference_batch'),
    (phone.adapters ?? []).join(', '))
  check('it reports the iOS agent version', phone.agent_version?.includes('ios'), phone.agent_version)

  // ---------------------------------------------------------------------- echo
  const echoJob = await harness.submitJob({ adapter: 'echo', mode: 'each', count: 2, sleepMs: 50 })
  await harness.waitFor('echo finishes', async () =>
    (await harness!.job(echoJob)).job.status !== 'running', 120_000)

  const echoView = await harness.job(echoJob)
  const echoDone = echoView.tasks.filter(t => t.state === 'succeeded')
  check('the phone ran real tasks', echoDone.length === echoView.job.total_items,
    `${echoDone.length}/${echoView.job.total_items}`)
  const echoOut = echoDone[0]?.output as { os: string; arch: string; hostname: string } | undefined
  check('the result was computed on iOS', echoOut?.os === 'ios',
    `os=${echoOut?.os} arch=${echoOut?.arch} host=${echoOut?.hostname}`)

  const { events: echoEvents } = await (await harness.api(`/jobs/${echoJob}/events`)).json() as
    { events: { type: string }[] }
  check('every signature verified against the key the phone generated',
    !echoEvents.some(e => e.type === 'result.signature_invalid'))

  // ----------------------------------------------------------------- inference
  if (existsSync('fixtures/manifest.json')) {
    const mlJob = await harness.submitJob({
      adapter: 'cpu_inference_batch', count: 100, batchSize: 100, hostId: phone.id,
    })
    await harness.waitFor('inference finishes on the phone', async () =>
      (await harness!.job(mlJob)).job.status !== 'running', 300_000)

    const mlView = await harness.job(mlJob)
    const mlDone = mlView.tasks.filter(t => t.state === 'succeeded')
    check('ONNX Runtime ran inside the iOS app', mlDone.length === 1,
      `${mlDone.length} of 1 slice`)

    if (mlDone.length === 1) {
      const out = mlDone[0]!.output as {
        count: number; correct: number; predictions: number[]
        itemsPerSecond: number; modelLoadMs: number
      }
      const accuracy = out.correct / out.count
      check('it classified every digit', out.predictions.length === out.count, `${out.count} digits`)
      check('it is as accurate as the model should be', accuracy > 0.9,
        `${(accuracy * 100).toFixed(1)}%`)
      note(`throughput ${out.itemsPerSecond}/s, model load ${out.modelLoadMs}ms `
         + `${DIM}(a simulator on this Mac, not phone silicon)${RESET}`)
    }
  } else {
    note('no fixtures — skipping inference (run: node scripts/build-fixtures.ts)')
  }

  // --------------------------------------------------------------- backgrounding
  // What iOS actually does when the app leaves the screen. On the Simulator background
  // processing does not run at all, so the honest assertion is that the system notices
  // and the fleet recovers — not that work continues.
  simctl(['terminate', device.udid, BUNDLE_ID])
  note('app terminated, as iOS would when reclaiming a suspended app')
  await harness.waitFor('the server notices the phone is gone', async () =>
    !(await harness!.hosts()).find(h => h.label === 'iPhone')!.online, 60_000)
  check('a vanished phone is marked offline rather than left hanging', true)

} catch (err) {
  failed += 1
  console.log(`    ${RED}ERROR${RESET} ${err instanceof Error ? err.message : String(err)}`)
} finally {
  try { await harness?.teardown() } catch {}
  if (booted) note('leaving the simulator booted for the next run')
}

console.log(`\n${passed} passed, ${failed} failed\n`)
process.exit(failed === 0 ? 0 : 1)
