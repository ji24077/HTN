import { ensureKeypair } from './keys.ts'
import { loadConfig, saveConfig, setPaused, isPaused } from './config.ts'
import { connect } from './transport.ts'
import { pair } from './pair.ts'
import { probe } from './capability.ts'
import { browserProbe } from './adapters/browser.ts'
import { AGENT_HOME, AGENT_VERSION } from './paths.ts'

const [command, ...rest] = process.argv.slice(2)

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
  case 'browser-probe': {
    const cfg = requireConfig()
    await browserProbe(flag('url') ?? `${cfg.server}/whoami`)
    break
  }

  default:
    console.log(`dwp-agent v${AGENT_VERSION}

  pair --server <url> --code <CODE> [--label <name>]   enroll this computer
  run [--allow-browser]                                 connect and accept work
  set-server --server <url>                             point at a new server address
  pause | resume                                        local kill switch (works offline)
  status                                                show identity and capability
  browser-probe [--url <url>]                           launch Chromium and report observed egress
`)
}
