import './instrument.ts'
import { ensureKeypair } from './keys.ts'
import { loadConfig, saveConfig, setPaused, isPaused } from './config.ts'
import { connect } from './transport.ts'
import { pair } from './pair.ts'
import { probe } from './capability.ts'
import { browserProbe } from './adapters/browser.ts'
import { applyUpdate, completePendingInstall, currentVersion } from './update.ts'
import { WORKLOADS, enableWorkload, isInstalled, availableAdapters, type WorkloadId } from './workloads.ts'
import { invocation } from './paths.ts'
import { installService, uninstallService, serviceStatus } from './service.ts'
import { runGui, guiAddress } from './gui.ts'
import { AGENT_HOME, AGENT_VERSION } from './paths.ts'

const [given, ...rest] = process.argv.slice(2)

/**
 * No arguments means "open the app"; `--help` means "tell me about yourself".
 *
 * This is the app convention rather than the command-line one, and it is chosen because
 * a double-clicked icon cannot pass arguments and cannot be told apart from a bare
 * invocation in a terminal. Deciding by whether stdout is a terminal looks like it would
 * work and does not: double-clicking a console executable on Windows allocates a console,
 * so the packaged app would print help into a box and exit, which is precisely the case
 * that has to work.
 */
const command = given ?? 'gui'

function flag(name: string): string | undefined {
  const i = rest.indexOf(`--${name}`)
  return i >= 0 ? rest[i + 1] : undefined
}

function requireConfig() {
  const cfg = loadConfig()
  if (!cfg) {
    console.error(`No agent config at ${AGENT_HOME}. Pair first:\n  pnpm agent pair --server <url> --code <CODE>`)
    process.exit(1)
  }
  return cfg
}

switch (command) {
  case 'pair': {
    const server = flag('server')
    const code = flag('code')
    if (!server || !code) {
      console.error('usage: agent pair --server <url> --code <CODE> [--label <name>]')
      process.exit(1)
    }
    await pair(server, code, flag('label'))
    break
  }

  case 'run': {
    const cfg = requireConfig()
    const { privateKey } = ensureKeypair()
    // Before anything loads a native addon: an update may have left its dependency step
    // for a process that is not holding those files open. This one is not, yet.
    await completePendingInstall()
    if (flag('allow-browser') !== undefined) { cfg.allowBrowser = true; saveConfig(cfg) }
    console.log(`[agent] ${cfg.label} (${cfg.hostId}) v${AGENT_VERSION} pid=${process.pid}${isPaused() ? '  [PAUSED]' : ''}`)
    connect(cfg, privateKey)
    break
  }

  /**
   * Re-point an already-enrolled host at a new server address.
   *
   * A tunnel URL changes every time the tunnel restarts. The host's identity does not,
   * so this must not require re-pairing — the keypair and host id stay exactly as they
   * are and only the address changes.
   */
  case 'set-server': {
    const cfg = requireConfig()
    const server = flag('server') ?? rest[0]
    if (!server) {
      console.error('usage: agent set-server --server <url>')
      process.exit(1)
    }
    const origin = server.replace(/\/+$/, '')
    cfg.server = origin
    cfg.wsUrl = `${origin.replace(/^http/, 'ws')}/agent/connect`
    saveConfig(cfg)
    console.log(`Now pointing at ${origin}\n  host id stays ${cfg.hostId} — no re-pairing needed.`)
    break
  }

  /**
   * Adopt this server's release key, for an agent that paired before signed releases
   * existed. Shown and confirmed explicitly, because it is the moment trust is
   * established and it should not be possible to do by accident.
   */
  case 'trust-updates': {
    const cfg = requireConfig()
    const res = await fetch(`${cfg.server}/release/latest`).catch(() => null)
    if (!res?.ok) {
      console.error(`\n  ${cfg.server} is not offering a signed release.\n`)
      process.exit(1)
    }
    const release = await res.json() as { publicKey: string; manifest: { version: string } }
    if (cfg.releaseKey === release.publicKey) {
      console.log(`\n  Already trusting this key. Nothing to do.\n`)
      break
    }
    if (flag('yes') === undefined) {
      console.log(`\n  ${cfg.server} signs its releases with:\n\n      ${release.publicKey}\n`)
      console.log(`  Current release: ${release.manifest.version}`)
      console.log(`\n  Only trust this if it matches the key the person running the network told you.`)
      console.log(`  To accept:  pnpm agent trust-updates --yes\n`)
      break
    }
    saveConfig({ ...cfg, releaseKey: release.publicKey })
    console.log(`\n  Trusted. Updates signed by this key will now install.\n`)
    break
  }

  /** What this computer can run, and what it could run if you added something. */
  case 'workloads': {
    console.log(`\n  This computer can run:\n`)
    for (const a of availableAdapters()) console.log(`    ${a}`)
    const missing = WORKLOADS.filter(w => !isInstalled(w.package))
    if (missing.length === 0) {
      console.log(`\n  Everything is installed.\n`)
    } else {
      console.log(`\n  Available to add:\n`)
      for (const w of missing) {
        console.log(`    ${w.id.padEnd(8)} ${w.describe}`)
        console.log(`             about ${w.approxMb} MB    pnpm agent enable ${w.id}`)
      }
      console.log('')
    }
    break
  }

  case 'enable': {
    const id = (flag('workload') ?? rest[0]) as WorkloadId | undefined
    if (!id || !WORKLOADS.some(w => w.id === id)) {
      console.error(`\n  usage: agent enable <${WORKLOADS.map(w => w.id).join('|')}>\n`)
      process.exit(1)
    }
    await enableWorkload(id)
    break
  }

  /**
   * Run at login, so nobody has to keep a terminal window open. Per-user throughout —
   * no sudo, no root daemon, and stoppable by whoever installed it.
   */
  case 'install-service': {
    requireConfig()
    const where = await installService()
    console.log(`\n  This computer will now join the network automatically when you log in.`)
    console.log(`    ${where}`)
    console.log(`\n  Stop that with:  ${invocation()} uninstall-service`)
    console.log(`  Pause work without uninstalling:  ${invocation()} pause\n`)
    break
  }

  case 'uninstall-service': {
    await uninstallService()
    console.log(`\n  It will no longer start on login. Your pairing and keys are untouched.\n`)
    break
  }

  case 'service-status': {
    const s = await serviceStatus()
    if (!s.installed) {
      console.log(`\n  Not set to run at login.\n  Turn that on with:  ${invocation()} install-service\n`)
    } else {
      console.log(`\n  Runs at login: yes`)
      console.log(`  Running now:   ${s.running ? 'yes' : 'no'}`)
      console.log(`  Defined in:    ${s.path}\n`)
    }
    break
  }

  case 'update': {
    const cfg = requireConfig()
    const { privateKey } = ensureKeypair()
    console.log(`  currently on ${currentVersion() ?? 'an unknown version'}; checking ${cfg.server}…`)
    const result = await applyUpdate(cfg, privateKey, { force: flag('force') !== undefined })
    if (result.status === 'updated') {
      console.log(`\n  Updated ${result.from ?? 'unknown'} -> ${result.to}\n  Start it again with:  ${invocation()} run\n`)
    } else if (result.status === 'current') {
      console.log(`\n  Already on ${result.version}. Nothing to do.\n`)
    } else {
      console.error(`\n  ${result.status === 'refused' ? 'Refused' : 'Could not update'}: ${result.reason}\n`)
      process.exit(1)
    }
    break
  }

  case 'pause':
  case 'resume': {
    setPaused(command === 'pause')
    console.log(command === 'pause'
      ? 'Paused. No new work will be accepted, and running work was cancelled.'
      : 'Resumed. This host will accept work again.')
    break
  }

  case 'status': {
    const cfg = loadConfig()
    const cap = probe(['echo'])
    console.log(JSON.stringify({ home: AGENT_HOME, paused: isPaused(), config: cfg, capability: cap }, null, 2))
    break
  }

  /**
   * Gate 1.7: launch a real Chromium on this host and report what a team-operated
   * endpoint observed. If that address is this machine's egress, the browser ran here.
   */
  /**
   * The desktop app: a window onto this same process.
   *
   * `--hidden` starts it without opening a window, which is what the login service
   * registers — the window is then opened on demand by launching the app again, which
   * hands off to the process already running rather than starting a second agent.
   */
  case 'gui': {
    await runGui({ hidden: rest.includes('--hidden') })
    break
  }

  /** For anyone who closed the window and wants it back from a terminal. */
  case 'gui-address': {
    const address = guiAddress()
    console.log(address
      ? `\n  ${address}\n`
      : `\n  Not running. Start it with:  ${invocation()} gui\n`)
    break
  }

  case 'browser-probe': {
    const cfg = requireConfig()
    await browserProbe(flag('url') ?? `${cfg.server}/whoami`)
    break
  }

  case 'help':
  case '--help':
  case '-h':
  default:
    console.log(`dwp-agent v${AGENT_VERSION}

  (no arguments)                                        open the desktop window
  gui [--hidden]                                        the same, named explicitly
  pair --server <url> --code <CODE> [--label <name>]   enroll this computer
  run [--allow-browser]                                 connect and accept work
  set-server --server <url>                             point at a new server address
  update [--force]                                      install the latest signed release
  trust-updates [--yes]                                 adopt this server's release key
  workloads                                             what this computer can run
  enable <ml|browser>                                   add an optional workload
  install-service                                       start automatically at login
  uninstall-service | service-status                    manage that
  pause | resume                                        local kill switch (works offline)
  status                                                show identity and capability
  browser-probe [--url <url>]                           launch Chromium and report observed egress
`)
}
