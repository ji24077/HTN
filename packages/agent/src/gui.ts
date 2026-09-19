/**
 * The desktop app.
 *
 * There is no second implementation here and no second process: the app *is* the agent.
 * This module adds a loopback HTTP server and a window onto the same process that holds
 * the connection, so the thing the window displays is the live state object rather than a
 * scrape of a log or a guess from a child process's stdout. That also means the app and
 * the agent can never disagree about their version, which is the drift that docs/07
 * warned about and the reason a separate shell was rejected.
 *
 * The window is a browser window, opened chromeless where a Chromium is available (Edge
 * is present on every Windows install, so that is the common case). The trade is stated
 * plainly rather than hidden: no tray icon, and the UI is a browser. What it buys is no
 * Rust toolchain, no Electron runtime, one file to ship, and updates that already work.
 */
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { spawn } from 'node:child_process'
import { randomBytes, timingSafeEqual } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { platform } from 'node:os'
import { join } from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'
import { createLogger } from '@dwp/protocol'
import { AGENT_HOME, AGENT_VERSION, isCompiledBinary } from './paths.ts'
import { isPaused, loadConfig, setPaused, type AgentConfig } from './config.ts'
import { ensureKeypair } from './keys.ts'
import { pairHost } from './pair.ts'
import { connect, type AgentHandle, type AgentState } from './transport.ts'
import { applyUpdate, completePendingInstall, restartIntoNewVersion } from './update.ts'
import { installService, serviceStatus, uninstallService, type ServiceStatus } from './service.ts'
import { availableAdapters } from './workloads.ts'
import { restrictToCurrentUser } from './winacl.ts'

const log = createLogger({ component: 'gui' })

const LOCK_PATH = join(AGENT_HOME, 'gui.json')
/**
 * A fixed port range rather than an ephemeral port, so the open window survives a
 * restart.
 *
 * Updating replaces this executable and restarts the process. With an ephemeral port the
 * window the owner was looking at would point at a dead port and stay broken until they
 * relaunched — the update would look like a crash. Keeping the port and the token means
 * the page's next poll simply succeeds again.
 */
const PORT_BASE = 43117
const PORT_TRIES = 20

/**
 * Where the window is and who is serving it.
 *
 * Deliberately *not* deleted when the process exits. It is the durable half — the token
 * and the port — and deleting it cost exactly what it was meant to protect: the process
 * that restarts after an update was reading this file to keep the window's address
 * stable, and its predecessor's exit handler removed the file first, so the replacement
 * minted a new token and every window open on the old address broke.
 *
 * `pid` is therefore a claim, not a fact. Liveness is decided by asking the port, which
 * is the only question that cannot be stale.
 */
type Lock = { port: number; token: string; pid: number; startedAt: string; version: string }

function readLock(): Lock | null {
  try { return JSON.parse(readFileSync(LOCK_PATH, 'utf8')) as Lock } catch { return null }
}

function writeLock(lock: Lock): void {
  mkdirSync(AGENT_HOME, { recursive: true, mode: 0o700 })
  const isNew = !existsSync(LOCK_PATH)
  // The token in here is an authenticator for everything the window can do, so it gets
  // the same protection as the private key it sits beside.
  writeFileSync(LOCK_PATH, JSON.stringify(lock, null, 2) + '\n', { mode: 0o600 })
  /**
   * Windows has no mode bits, so the 0o600 above is decoration there — the same gap
   * `keys.ts` closes for the private key, closed the same way and only on creation, since
   * the ACL survives later writes and spawning two processes on every start would not.
   */
  if (isNew && process.platform === 'win32') {
    const acl = restrictToCurrentUser(LOCK_PATH)
    if (!acl.ok) log.warn('gui.token_acl_failed', { detail: acl.detail, path: LOCK_PATH })
  }
}

/** Is a DWP window already serving on this port? */
async function probe(port: number, token: string): Promise<boolean> {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/${token}/api/state`,
      { signal: AbortSignal.timeout(1_500) })
    if (!res.ok) return false
    const body = await res.json() as { dwp?: boolean }
    return body.dwp === true
  } catch {
    return false
  }
}

/**
 * Open the window.
 *
 * `--app=` gives a Chromium window with no tabs or address bar, which is what makes this
 * read as an application rather than as a web page. Edge ships with Windows, so that
 * path is reliable there; a Mac without Chrome or Edge falls back to an ordinary tab in
 * the default browser, which works and looks like a web page. That is the honest cost of
 * not shipping a runtime.
 */
function openWindow(url: string): void {
  const chromium = platform() === 'win32'
    ? [
        join(process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
        join(process.env['ProgramFiles'] ?? 'C:\\Program Files', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
        join(process.env['ProgramFiles'] ?? 'C:\\Program Files', 'Google', 'Chrome', 'Application', 'chrome.exe'),
        join(process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)', 'Google', 'Chrome', 'Application', 'chrome.exe'),
      ]
    : platform() === 'darwin'
      ? [
          '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
          '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
        ]
      : ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/microsoft-edge']

  const found = chromium.find(p => existsSync(p))
  if (found) {
    const child = spawn(found, [`--app=${url}`, '--window-size=520,660'], {
      detached: true, stdio: 'ignore',
    })
    child.unref()
    return
  }

  // No Chromium: hand it to whatever opens links.
  const [cmd, args] = platform() === 'win32'
    ? ['cmd', ['/c', 'start', '', url]]
    : platform() === 'darwin'
      ? ['open', [url]]
      : ['xdg-open', [url]]
  const child = spawn(cmd, args, { detached: true, stdio: 'ignore' })
  child.unref()
}

/** Accept either a full invite link or a bare code, and say which fields are missing. */
function parseInvite(text: string, fallbackServer?: string): { server: string; code: string } | { error: string } {
  const trimmed = text.trim()
  if (trimmed === '') return { error: 'Paste the invite link you were sent.' }

  if (/^https?:\/\//i.test(trimmed)) {
    let url: URL
    try { url = new URL(trimmed) } catch { return { error: 'That does not look like a link. Paste the whole thing, starting with https://' } }
    const code = url.searchParams.get('code')
    if (!code) {
      return { error: 'That link has no invite code in it. It should end with ?code=SOMETHING' }
    }
    return { server: url.origin, code }
  }

  // A bare code is only usable if we already know where to send it.
  if (!fallbackServer) {
    return { error: 'That looks like just the code. Paste the whole invite link instead, so this computer knows which network to join.' }
  }
  return { server: fallbackServer, code: trimmed }
}

// ----------------------------------------------------------------- the page

/**
 * One self-contained page: no CDN, no build step, no external font.
 *
 * It has to work on a machine that has just met this project and may have no internet
 * beyond the control service, so every byte it needs is here.
 */
function page(token: string): string {
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DWP Agent</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f6f7f9; --card: #ffffff; --ink: #14181f; --dim: #667085; --line: #e3e6eb;
    --ok: #12855f; --warn: #b26a00; --bad: #c0392b; --busy: #0b6e8c; --accent: #0b6e8c;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14181f; --card: #1c222b; --ink: #eef1f5; --dim: #97a0ad; --line: #2b323d;
      --ok: #3ecf8e; --warn: #e0a33a; --bad: #f06a5d; --busy: #52b6d8; --accent: #52b6d8;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink); padding: 22px 18px 28px;
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 460px; margin: 0 auto; }
  h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
  .sub { color: var(--dim); font-size: 12px; margin-top: 2px; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px; margin-top: 14px;
  }
  .state { display: flex; align-items: center; gap: 10px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex: none; background: var(--dim); }
  .dot.ok { background: var(--ok); }
  .dot.warn { background: var(--warn); }
  .dot.bad { background: var(--bad); }
  .dot.busy { background: var(--busy); animation: pulse 1.4s ease-in-out infinite; }
  .dot.wait { background: var(--dim); animation: pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.35 } }
  @media (prefers-reduced-motion: reduce) { .dot { animation: none !important } }
  .headline { font-size: 17px; font-weight: 600; letter-spacing: -0.01em; }
  .note { color: var(--dim); font-size: 12.5px; margin-top: 8px; }
  .advice {
    margin-top: 12px; padding: 10px 12px; border-radius: 8px; font-size: 12.5px;
    background: color-mix(in srgb, var(--warn) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--warn) 35%, transparent);
    white-space: pre-wrap;
  }
  dl { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 0; font-size: 12.5px; }
  dt { color: var(--dim); }
  dd { margin: 0; overflow-wrap: anywhere; }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 11px 0; border-top: 1px solid var(--line); }
  .row:first-child { border-top: 0; padding-top: 2px; }
  .row .label { font-size: 13px; }
  .row .hint { color: var(--dim); font-size: 11.5px; margin-top: 1px; }
  button {
    font: inherit; font-size: 13px; padding: 7px 13px; border-radius: 8px; cursor: pointer;
    border: 1px solid var(--line); background: var(--card); color: var(--ink); flex: none;
  }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button:disabled { opacity: 0.5; cursor: default; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; width: 100%; padding: 10px; }
  button.danger:hover { border-color: var(--bad); color: var(--bad); }
  input, textarea {
    font: inherit; font-size: 13px; width: 100%; padding: 9px 11px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--bg); color: var(--ink); resize: vertical;
  }
  label.field { display: block; font-size: 12px; color: var(--dim); margin: 12px 0 5px; }
  .err { color: var(--bad); font-size: 12.5px; margin-top: 10px; white-space: pre-wrap; }
  .task { font-size: 12.5px; color: var(--dim); margin-top: 6px; white-space: pre-wrap; }
  .task code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink); }
  footer { color: var(--dim); font-size: 11.5px; margin-top: 16px; text-align: center; }
  [hidden] { display: none !important; }
</style>
</head><body>
<div class="wrap">
  <h1>DWP Agent</h1>
  <div class="sub" id="sub">starting…</div>

  <!-- Not yet paired -->
  <section id="join" hidden>
    <div class="card">
      <div class="headline">Join a network</div>
      <div class="note">Paste the invite link you were sent. It works once and expires ten minutes after it was made.</div>
      <label class="field" for="invite">Invite link</label>
      <textarea id="invite" rows="2" placeholder="https://example.com/join?code=ABCD-1234" autocomplete="off" spellcheck="false"></textarea>
      <label class="field" for="name">What should this computer be called? (optional)</label>
      <input id="name" placeholder="e.g. Sam's laptop" autocomplete="off">
      <div style="margin-top:14px"><button class="primary" id="joinBtn">Join</button></div>
      <div class="err" id="joinErr" hidden></div>
    </div>
  </section>

  <!-- Paired -->
  <section id="main" hidden>
    <div class="card">
      <div class="state">
        <span class="dot" id="dot"></span>
        <span class="headline" id="headline">…</span>
      </div>
      <div class="task" id="tasks" hidden></div>
      <div class="advice" id="advice" hidden></div>
    </div>

    <div class="card">
      <dl>
        <dt>This computer</dt><dd id="label">—</dd>
        <dt>Network</dt><dd id="server">—</dd>
        <dt>Can run</dt><dd id="adapters">—</dd>
        <dt>Version</dt><dd id="version">—</dd>
      </dl>
    </div>

    <div class="card">
      <div class="row" id="retryRow" hidden>
        <div>
          <div class="label">Take over the connection</div>
          <div class="hint">Use this once you have stopped the other copy. It retries by
          itself every minute or so anyway.</div>
        </div>
        <button id="retryBtn">Try again</button>
      </div>
      <div class="row">
        <div>
          <div class="label" id="pauseLabel">Accepting work</div>
          <div class="hint">Pausing stops new work and cancels anything running. It works even offline.</div>
        </div>
        <button id="pauseBtn">Pause</button>
      </div>
      <div class="row">
        <div>
          <div class="label">Start automatically when I log in</div>
          <div class="hint" id="loginHint">—</div>
        </div>
        <button id="loginBtn">—</button>
      </div>
      <div class="row">
        <div>
          <div class="label">Updates</div>
          <div class="hint" id="updateHint">Installed automatically, signed by the person running the network.</div>
        </div>
        <button id="updateBtn">Check now</button>
      </div>
      <div class="row">
        <div>
          <div class="label">Stop the agent</div>
          <div class="hint">Closing this window leaves it running. This stops it until next login.</div>
        </div>
        <button class="danger" id="quitBtn">Quit</button>
      </div>
    </div>
    <div class="err" id="actionErr" hidden></div>
  </section>

  <footer id="footer">Closing this window does not stop the agent.</footer>
</div>
<script>
const BASE = '/${token}'
const $ = id => document.getElementById(id)
let quitting = false

const show = (el, on) => { el.hidden = !on }

async function api(path, body) {
  const res = await fetch(BASE + '/api/' + path, body === undefined ? {} : {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  })
  const text = await res.text()
  let parsed = null
  try { parsed = text ? JSON.parse(text) : null } catch {}
  if (!res.ok) throw new Error((parsed && parsed.error) || text || ('HTTP ' + res.status))
  return parsed
}

/**
 * Standing down is correct, but "another copy is running" with no next step is a dead
 * end — so say which copy it probably is, and give the command that ends it, for the
 * machine actually being looked at.
 */
function stoodDownHelp(platform) {
  if (platform === 'win32') return 'It is most likely one installed from PowerShell, at ' +
    '%USERPROFILE%\\.dwp\\bin\\dwp-agent.exe, started by a logon task. To hand over to this app, ' +
    'run these two lines in PowerShell and open this app again:\n\n' +
    '    schtasks /Delete /F /TN "DWP Agent"\n' +
    '    Stop-Process -Name dwp-agent -Force'
  if (platform === 'darwin') return 'It is most likely one installed from a terminal, at ' +
    '~/.dwp/bin/dwp-agent. To hand over to this app, run this and open this app again:\n\n' +
    '    ~/.dwp/bin/dwp-agent uninstall-service; pkill -f dwp-agent'
  return 'It is most likely one installed from a terminal, at ~/.dwp/bin/dwp-agent. ' +
    'Stop that one, then open this app again.'
}

function describe(s) {
  if (s.stoodDown) return ['warn', 'Stopped — another copy is running',
    'Another agent on this computer already has this identity, so this one stood down ' +
    'rather than fight it for the connection. Your computer is still doing the work — ' +
    'the other copy is doing it, so nothing is broken.\n\n' + stoodDownHelp(s.platform)]
  if (s.paused) return ['warn', 'Paused', 'No work will be accepted until you resume.']
  if (s.connection === 'online' && s.running.length > 0) return ['busy', 'Working', null]
  if (s.connection === 'online') return ['ok', 'Connected', 'Waiting for work. Nothing to do right now.']
  if (s.connection === 'connecting') return ['wait', s.attempt > 1 ? 'Reconnecting…' : 'Connecting…', null]
  return ['bad', 'Offline', s.lastLostReason ? 'Last seen: ' + s.lastLostReason : 'Not connected.']
}

function render(s) {
  show($('join'), !s.paired)
  show($('main'), s.paired)
  $('sub').textContent = s.paired ? (s.updating ? 'installing an update…' : 'v' + s.version) : 'not yet joined'
  if (!s.paired) return

  const [kind, headline, note] = describe(s)
  $('dot').className = 'dot ' + kind
  $('headline').textContent = headline

  if (s.running.length > 0) {
    $('tasks').innerHTML = s.running.map(t =>
      'Running <code>' + t.adapter + '</code> for ' + Math.round((Date.now() - t.startedAt) / 1000) + 's').join('<br>')
    show($('tasks'), true)
  } else if (note) {
    $('tasks').textContent = note
    show($('tasks'), true)
  } else {
    show($('tasks'), false)
  }

  show($('advice'), Boolean(s.advice))
  if (s.advice) $('advice').textContent = s.advice

  $('label').textContent = s.label
  $('server').textContent = s.server
  $('adapters').textContent = s.adapters.join(', ')
  $('version').textContent = 'v' + s.version + (s.pinnedKey ? ' — updates verified' : ' — updates unsigned, manual only')

  show($('retryRow'), s.stoodDown)
  $('pauseLabel').textContent = s.paused ? 'Paused' : 'Accepting work'
  $('pauseBtn').textContent = s.paused ? 'Resume' : 'Pause'
  $('loginBtn').textContent = s.runsAtLogin ? 'Turn off' : 'Turn on'
  $('loginHint').textContent = s.runsAtLogin
    ? 'On. It joins by itself after a restart, with no window open.'
    : 'Off. It only runs while this app is open.'
  $('updateHint').textContent = s.updating ? 'Installing an update. It will restart itself.' : s.updateNote
}

async function tick() {
  if (quitting) return
  try {
    render(await api('state'))
    show($('footer'), true)
    $('footer').textContent = 'Closing this window does not stop the agent.'
  } catch {
    // A restart after an update lands here for a second or two. Say so rather than
    // showing an error that looks like a crash.
    $('footer').textContent = 'Reconnecting to the agent…'
  }
}

function busy(btn, on) { btn.disabled = on }

async function act(btn, fn) {
  show($('actionErr'), false)
  busy(btn, true)
  try { await fn() } catch (err) {
    $('actionErr').textContent = err.message
    show($('actionErr'), true)
  } finally { busy(btn, false); await tick() }
}

$('joinBtn').onclick = () => act($('joinBtn'), async () => {
  show($('joinErr'), false)
  try {
    await api('pair', { invite: $('invite').value, label: $('name').value })
  } catch (err) {
    $('joinErr').textContent = err.message
    show($('joinErr'), true)
  }
})
$('retryBtn').onclick = () => act($('retryBtn'), () => api('retry', {}))
$('pauseBtn').onclick = () => act($('pauseBtn'), () => api('pause', { paused: $('pauseBtn').textContent === 'Pause' }))
$('loginBtn').onclick = () => act($('loginBtn'), () => api('login-at-start', { enabled: $('loginBtn').textContent === 'Turn on' }))
$('updateBtn').onclick = () => act($('updateBtn'), async () => {
  const r = await api('update', {})
  $('updateHint').textContent = r.message
})
$('quitBtn').onclick = () => act($('quitBtn'), async () => {
  quitting = true
  await api('quit', {}).catch(() => {})
  document.querySelector('.wrap').innerHTML =
    '<h1>DWP Agent</h1><div class="sub">stopped</div>' +
    '<div class="card"><div class="headline">Stopped</div>' +
    '<div class="note">This computer has left the network. Open the app again to rejoin' +
    ', or it will start by itself at your next login if you left that turned on.</div></div>'
})

tick()
setInterval(tick, 1000)
</script>
</body></html>
`
}

// ------------------------------------------------------------------ the app

type GuiOptions = { hidden: boolean }

export async function runGui(opts: GuiOptions): Promise<void> {
  const existing = readLock()
  /**
   * Are we the replacement this process's own parent spawned to finish an update?
   *
   * If so the lock belongs to a process that is on its way out, and handing off to it
   * would be handing off to something about to exit. Take the port back instead — after
   * waiting for it, since the parent may not have released it yet.
   */
  const supersedingParent = existing !== null && existing.pid === process.ppid

  if (existing && !supersedingParent && await probe(existing.port, existing.token)) {
    const url = `http://127.0.0.1:${existing.port}/${existing.token}/`
    if (!opts.hidden) {
      openWindow(url)
      console.log(`\n  Already running. Opened the existing window.\n  ${url}\n`)
    } else {
      console.log(`\n  Already running on ${url} — nothing to do.\n`)
    }
    return
  }

  // Reuse the token so a window left open across a restart keeps working.
  const token = existing?.token ?? randomBytes(16).toString('hex')
  const tokenBuf = Buffer.from(token)

  let config: AgentConfig | null = loadConfig()
  let state: AgentState = {
    connection: 'offline', attempt: 0, connectedSince: null, running: [],
    lastLostReason: null, advice: null, stoodDown: false, updating: false,
  }
  let service: ServiceStatus = { installed: false, platform: platform() }
  const refreshService = async (): Promise<void> => {
    // Spawning launchctl/schtasks on every poll would be a process per second; the answer
    // only changes when something here changes it, plus the occasional outside edit.
    service = await serviceStatus().catch(() => ({ installed: false, platform: platform() }) as ServiceStatus)
  }
  await refreshService()
  const serviceTimer = setInterval(() => void refreshService(), 30_000)
  serviceTimer.unref()

  let connected = false
  let agent: AgentHandle | null = null
  const startConnection = async (cfg: AgentConfig): Promise<void> => {
    if (connected) return
    connected = true
    // An update may have deferred its dependency step to a process that is not holding
    // those files open. This one is not, yet.
    await completePendingInstall()
    const { privateKey } = ensureKeypair()
    agent = connect(cfg, privateKey, next => { state = next })
  }

  const send = (res: ServerResponse, code: number, body: unknown): void => {
    const text = JSON.stringify(body)
    res.writeHead(code, {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store',
      // This page needs nothing from anywhere else, so forbid everything it might fetch.
      'content-security-policy': "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'",
      'referrer-policy': 'no-referrer',
    })
    res.end(text)
  }

  const readBody = async (req: IncomingMessage): Promise<Record<string, unknown>> => {
    const chunks: Buffer[] = []
    let size = 0
    for await (const chunk of req) {
      size += (chunk as Buffer).length
      if (size > 64 * 1024) throw new Error('body too large')
      chunks.push(chunk as Buffer)
    }
    if (chunks.length === 0) return {}
    return JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>
  }

  let quitTimer: NodeJS.Timeout | undefined

  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log.error('gui.request_failed', { err })
      if (!res.headersSent) send(res, 500, { error: err instanceof Error ? err.message : String(err) })
    })
  })

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    /**
     * Two guards, both necessary, neither sufficient alone.
     *
     * The Host check stops DNS rebinding: a name the attacker controls that resolves to
     * 127.0.0.1 would otherwise let a page in their browser drive this server. The token
     * in the path stops any other local process or page that has not read a file only
     * this user can read — and anything that *can* read it could already read the agent's
     * private key sitting beside it, so this grants nothing new.
     */
    const host = req.headers.host ?? ''
    const port = (server.address() as { port: number } | null)?.port
    if (host !== `127.0.0.1:${port}` && host !== `localhost:${port}`) {
      send(res, 403, { error: 'wrong host' })
      return
    }

    const url = new URL(req.url ?? '/', `http://127.0.0.1:${port}`)
    const [, given, ...rest] = url.pathname.split('/')
    const supplied = Buffer.from(given ?? '')
    const authorised = supplied.length === tokenBuf.length && timingSafeEqual(supplied, tokenBuf)
    if (!authorised) {
      send(res, 404, { error: 'not found' })
      return
    }

    const route = rest.join('/')

    if (req.method === 'GET' && (route === '' || route === 'index.html')) {
      const body = page(token)
      res.writeHead(200, {
        'content-type': 'text/html; charset=utf-8',
        'cache-control': 'no-store',
        'content-security-policy': "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; form-action 'none'",
        'referrer-policy': 'no-referrer',
      })
      res.end(body)
      return
    }

    if (req.method === 'GET' && route === 'api/state') {
      send(res, 200, {
        dwp: true,
        version: AGENT_VERSION,
        paired: config !== null,
        label: config?.label ?? null,
        hostId: config?.hostId ?? null,
        server: config?.server ?? null,
        pinnedKey: Boolean(config?.releaseKey),
        paused: isPaused(),
        platform: platform(),
        adapters: availableAdapters(),
        runsAtLogin: service.installed,
        updateNote: config?.releaseKey
          ? 'Installed automatically, verified against the key this computer pinned when it joined.'
          : 'This network offers no signed releases, so updates stay manual.',
        connection: state.connection,
        attempt: state.attempt,
        connectedSince: state.connectedSince,
        running: state.running,
        lastLostReason: state.lastLostReason,
        advice: state.advice,
        stoodDown: state.stoodDown,
        updating: state.updating,
      })
      return
    }

    if (req.method !== 'POST') {
      send(res, 405, { error: 'method not allowed' })
      return
    }

    const body = await readBody(req)

    if (route === 'api/pair') {
      if (config) { send(res, 409, { error: 'This computer has already joined.' }); return }
      const parsed = parseInvite(String(body.invite ?? ''), undefined)
      if ('error' in parsed) { send(res, 400, { error: parsed.error }); return }
      const label = String(body.label ?? '').trim()
      const outcome = await pairHost(parsed.server, parsed.code, label === '' ? undefined : label)
      if (!outcome.ok) {
        send(res, 400, { error: outcome.hint ? `${outcome.message}\n\n${outcome.hint}` : outcome.message })
        return
      }
      config = loadConfig()
      log.info('gui.paired', { hostId: outcome.hostId, label: outcome.label })
      if (config) void startConnection(config)
      send(res, 200, { ok: true })
      return
    }

    if (route === 'api/retry') {
      if (!agent) { send(res, 400, { error: 'Join a network first.' }); return }
      agent.retryNow()
      log.info('gui.retry_requested')
      send(res, 200, { ok: true })
      return
    }

    if (route === 'api/pause') {
      setPaused(body.paused === true)
      log.info('gui.pause', { paused: body.paused === true })
      send(res, 200, { ok: true })
      return
    }

    if (route === 'api/login-at-start') {
      try {
        if (body.enabled === true) {
          if (!config) { send(res, 400, { error: 'Join a network first.' }); return }
          // 'gui' mode, not 'run': one process, which the window attaches to. Installing
          // the headless agent as well would put two agents on one host identity.
          await installService('gui')
        } else {
          await uninstallService()
        }
      } catch (err) {
        send(res, 500, { error: `Could not change that: ${err instanceof Error ? err.message : String(err)}` })
        return
      }
      await refreshService()
      send(res, 200, { ok: true, runsAtLogin: service.installed })
      return
    }

    if (route === 'api/update') {
      if (!config) { send(res, 400, { error: 'Join a network first.' }); return }
      const { privateKey } = ensureKeypair()
      const result = await applyUpdate(config, privateKey, { force: false })
      if (result.status === 'updated') {
        send(res, 200, { message: `Updated to ${result.to}. Restarting…` })
        // Let the response flush, then hand over. The window keeps the same address and
        // token, so its next poll finds the new process.
        setTimeout(() => {
          server.close()
          restartIntoNewVersion()
        }, 300)
        return
      }
      send(res, 200, {
        message: result.status === 'current'
          ? `Already up to date (${result.version}).`
          : `${result.status === 'refused' ? 'Refused' : 'Could not check'}: ${result.reason}`,
      })
      return
    }

    if (route === 'api/quit') {
      send(res, 200, { ok: true })
      log.info('gui.quit')
      // The lock stays behind on purpose, so reopening the app returns to the same
      // address. Nothing is listening there until it does.
      quitTimer = setTimeout(() => process.exit(0), 250)
      quitTimer.unref()
      return
    }

    send(res, 404, { error: 'not found' })
  }

  /**
   * Take the preferred port, waiting for it if our predecessor still holds it.
   *
   * An update restart spawns the replacement before this process has exited, so the
   * socket can be a few hundred milliseconds from being free. Falling straight through to
   * another port would work but would break the window that is open on the old one.
   */
  const listenOn = (port: number): Promise<boolean> => new Promise(resolve => {
    const onError = (err: NodeJS.ErrnoException): void => {
      server.removeListener('error', onError)
      if (err.code === 'EADDRINUSE') resolve(false)
      else resolve(false)
    }
    server.once('error', onError)
    server.listen(port, '127.0.0.1', () => {
      server.removeListener('error', onError)
      resolve(true)
    })
  })

  let bound = 0
  const preferred = existing?.port ?? PORT_BASE
  const waitForPreferred = supersedingParent ? 12 : 1
  for (let i = 0; i < waitForPreferred && bound === 0; i += 1) {
    if (await listenOn(preferred)) bound = preferred
    else if (i + 1 < waitForPreferred) await sleep(250)
  }
  for (let i = 0; bound === 0 && i < PORT_TRIES; i += 1) {
    if (await listenOn(PORT_BASE + i)) bound = PORT_BASE + i
  }
  if (bound === 0) {
    console.error(`\n  Could not open a local port in ${PORT_BASE}–${PORT_BASE + PORT_TRIES - 1}.\n` +
      `  Something else is using all of them.\n`)
    process.exit(1)
  }

  writeLock({ port: bound, token, pid: process.pid, startedAt: new Date().toISOString(), version: AGENT_VERSION })
  const url = `http://127.0.0.1:${bound}/${token}/`
  log.info('gui.listening', { port: bound, hidden: opts.hidden, paired: config !== null })

  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.on(signal, () => process.exit(0))
  }

  if (config) await startConnection(config)

  if (opts.hidden) {
    console.log(`  DWP Agent running in the background. Window: ${url}`)
  } else {
    openWindow(url)
    console.log(`\n  DWP Agent\n  ${url}\n\n` +
      `  Closing the window leaves it running. Quit from inside the window, or press Ctrl-C here.\n` +
      (isCompiledBinary() ? '' : `  (Running from source. A packaged app hides this console.)\n`))
  }
}

/** Where the window is, for someone who lost it. */
export function guiAddress(): string | null {
  const lock = readLock()
  return lock ? `http://127.0.0.1:${lock.port}/${lock.token}/` : null
}
