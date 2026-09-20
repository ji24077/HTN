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
import { AGENT_HOME, AGENT_VERSION, invocation, isCompiledBinary } from './paths.ts'
import { clearConfig, isPaused, loadConfig, saveConfig, setPaused, type AgentConfig } from './config.ts'
import { ensureKeypair } from './keys.ts'
import { pairHost, parseInvite, autoEnrol } from './pair.ts'
import {
  containerIdentity, freeRamMb, guiBindHost, guiPort, guiPublicOrigin, hostAllowed,
  isContainer, shouldOpenWindow, supervisorNote,
} from './runtime.ts'
import { connect, type AgentHandle, type AgentState } from './transport.ts'
import {
  budgetState, loadPerCore, onBattery, type BudgetWindow, type Limits, type Weekday,
} from './limits.ts'
import * as history from './history.ts'
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
  let lock: Lock
  try { lock = JSON.parse(readFileSync(LOCK_PATH, 'utf8')) as Lock } catch { return null }
  // A lock outlives the process that wrote it. Trusting it blindly sends the window to a
  // dead port, where the page sits on its placeholder forever with no way out from inside
  // the app -- so check the recorded pid is actually alive first. Signal 0 tests for
  // existence without delivering anything; EPERM means it exists under another user.
  try { process.kill(lock.pid, 0) } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'EPERM') return null
  }
  return lock
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
  /**
   * A container has no display, no browser and nobody in front of it. Spawning xdg-open
   * there is not merely useless: it is a missing binary, and the ENOENT from an unhandled
   * spawn takes the whole agent down on start — which is how the first containerised
   * agent managed to exit before it had finished joining.
   */
  if (!shouldOpenWindow()) {
    console.log(`\n  Open this from a browser on the host:\n  ${url}\n`)
    return
  }
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

/** Minimal escaping for the few values the 404 above interpolates. */
function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

/**
 * Accept only what the rules can actually act on.
 *
 * Every field is clamped rather than merely type-checked, because these numbers are
 * consulted on the hot path for every offer and a negative or absurd one would not throw
 * -- it would quietly make a rule that never fires, or one that never stops firing. The
 * clamps are the same ones the input controls advertise, so the page and this agree.
 */
function sanitiseLimits(raw: unknown): Limits {
  if (raw === null || raw === undefined) return {}
  if (typeof raw !== 'object') throw new Error('limits must be an object')
  const input = raw as Record<string, unknown>
  const limits: Limits = {}

  if (Array.isArray(input.workloads)) {
    const workloads = input.workloads.filter((w): w is string => typeof w === 'string').slice(0, 32)
    // An empty list is "everything", not "nothing" -- see allowedWorkloads.
    if (workloads.length > 0) limits.workloads = workloads
  }

  const budget = input.budget as { minutes?: unknown; per?: unknown } | undefined
  if (budget && typeof budget === 'object') {
    const minutes = Number(budget.minutes)
    const per: BudgetWindow = budget.per === 'hour' ? 'hour' : 'day'
    // A week of minutes is the ceiling; beyond that it is not a limit.
    if (Number.isFinite(minutes) && minutes > 0) {
      limits.budget = { minutes: Math.min(Math.round(minutes), 10_080), per }
    }
  }

  const schedule = input.schedule as { from?: unknown; to?: unknown; days?: unknown } | undefined
  if (schedule && typeof schedule === 'object') {
    const clock = /^\d{1,2}:\d{2}$/
    const from = String(schedule.from ?? '')
    const to = String(schedule.to ?? '')
    if (clock.test(from) && clock.test(to) && from !== to) {
      limits.schedule = { from, to }
      if (Array.isArray(schedule.days)) {
        const days = [...new Set(schedule.days.map(Number).filter(d => Number.isInteger(d) && d >= 0 && d <= 6))]
        if (days.length > 0 && days.length < 7) limits.schedule.days = days as Weekday[]
      }
    }
  }

  const pressure = input.pressure as Record<string, unknown> | undefined
  if (pressure && typeof pressure === 'object') {
    const out: NonNullable<Limits['pressure']> = {}
    const load = Number(pressure.maxLoadPerCore)
    if (Number.isFinite(load) && load > 0) out.maxLoadPerCore = Math.min(load, 64)
    const ram = Number(pressure.minFreeRamMb)
    if (Number.isFinite(ram) && ram > 0) out.minFreeRamMb = Math.min(Math.round(ram), 1_048_576)
    if (pressure.notOnBattery === true) out.notOnBattery = true
    if (Object.keys(out).length > 0) limits.pressure = out
  }

  return limits
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

  /* The tab strip. Four words on one line at 520px, which is the window's width. */
  .tabs { display: flex; gap: 2px; margin: 14px 0 10px; border-bottom: 1px solid var(--line); }
  .tabs button {
    flex: 1; border: 0; background: none; border-radius: 0; padding: 8px 4px;
    color: var(--dim); font-size: 12.5px; border-bottom: 2px solid transparent;
  }
  .tabs button:hover:not(.on) { color: var(--ink); }
  .tabs button.on { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }

  .field { padding: 11px 0; border-top: 1px solid var(--line); }
  .field:first-child { border-top: 0; padding-top: 2px; }
  .field > .label { font-size: 13px; }
  .field > .hint { color: var(--dim); font-size: 11.5px; margin-top: 1px; }
  .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  input[type=number], input[type=time], select {
    font: inherit; font-size: 12.5px; padding: 5px 7px; border-radius: 7px;
    border: 1px solid var(--line); background: var(--bg); color: var(--ink);
  }
  input[type=number] { width: 5.5rem; }
  .choices { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .choices label {
    display: inline-flex; align-items: center; gap: 5px; font-size: 12px;
    border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; cursor: pointer;
  }
  .choices label.on { border-color: var(--accent); color: var(--accent); }
  .meter { height: 6px; border-radius: 3px; background: var(--line); overflow: hidden; margin-top: 8px; }
  .meter > div { height: 100%; background: var(--accent); }
  .why { font-size: 12px; color: var(--warn); margin-top: 6px; }
  .saved { font-size: 12px; color: var(--ok); }
  input, textarea {
    font: inherit; font-size: 13px; width: 100%; padding: 9px 11px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--bg); color: var(--ink); resize: vertical;
  }
  label.field { display: block; font-size: 12px; color: var(--dim); margin: 12px 0 5px; }
  .check { display: flex; align-items: flex-start; gap: 9px; margin-top: 14px; cursor: pointer; }
  .check input { width: auto; flex: none; margin: 2px 0 0; }
  .check .label { display: block; font-size: 13px; }
  .check .hint { display: block; color: var(--dim); font-size: 11.5px; margin-top: 2px; }
  .err { color: var(--bad); font-size: 12.5px; margin-top: 10px; white-space: pre-wrap; }
  .task { font-size: 12.5px; color: var(--dim); margin-top: 6px; white-space: pre-wrap; }
  .task code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink); }
  /* Recent work. The strip lets flex shrink the bars, with a 4px floor, so thirty runs
     always fit the card however long the longest one was. */
  .strip { display: flex; align-items: stretch; gap: 2px; height: 24px; margin-top: 13px; overflow: hidden; }
  .strip > div { flex: 0 1 auto; min-width: 4px; border-radius: 2px; background: var(--dim); }
  .strip > div.ok { background: var(--ok); }
  .strip > div.warn { background: var(--warn); }
  .strip > div.bad { background: var(--bad); }
  .runs { margin-top: 12px; font-size: 12.5px; }
  .run { display: flex; align-items: baseline; gap: 8px; padding: 6px 0; border-top: 1px solid var(--line); }
  .run:first-child { border-top: 0; }
  .run code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink); }
  .run .cost { color: var(--dim); overflow-wrap: anywhere; }
  .run .when { color: var(--dim); margin-left: auto; flex: none; }
  .run .mark { font-weight: 600; flex: none; }
  .run .mark.ok { color: var(--ok); }
  .run .mark.warn { color: var(--warn); }
  .run .mark.bad { color: var(--bad); }
  .adapters { margin-top: 11px; }
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
      <label class="check" for="runAtLogin">
        <input type="checkbox" id="runAtLogin" checked>
        <span>
          <span class="label">Rejoin automatically after a restart</span>
          <span class="hint">Otherwise this computer is only on the network while this app
          is open — close it, log out or restart, and it stops contributing.</span>
        </span>
      </label>
      <div style="margin-top:14px"><button class="primary" id="joinBtn">Join</button></div>
      <div class="err" id="joinErr" hidden></div>
    </div>
  </section>

  <!-- Paired -->
  <section id="main" hidden>
    <nav class="tabs">
      <button data-tab="status" class="on">Status</button>
      <button data-tab="work">Work</button>
      <button data-tab="limits">Limits</button>
      <button data-tab="settings">Settings</button>
    </nav>

    <div data-panel="status">
    <div class="card">
      <div class="state">
        <span class="dot" id="dot"></span>
        <span class="headline" id="headline">…</span>
      </div>
      <div class="task" id="tasks" hidden></div>
      <div class="advice" id="advice" hidden></div>
      <!-- Why nothing is running, when the reason is a rule the owner set rather than
           an empty queue. Without it the two are indistinguishable. -->
      <div class="why" id="whyIdle" hidden></div>
    </div>

    <div class="card">
      <dl>
        <dt>This computer</dt><dd id="label">—</dd>
        <dt>Network</dt><dd id="server">—</dd>
        <dt>Can run</dt><dd id="adapters">—</dd>
        <dt>Version</dt><dd id="version">—</dd>
        <dt id="runsInLabel" hidden>Runs in</dt><dd id="runsIn" hidden>—</dd>
      </dl>
    </div>
    </div>

    <div data-panel="work" hidden>
    <div class="card">
      <div class="headline">Recent work</div>
      <div class="note" id="histSummary">Nothing has run on this computer yet.</div>
      <div class="strip" id="histStrip" hidden></div>
      <div class="runs" id="histRuns" hidden></div>
      <div class="note adapters" id="histAdapters" hidden></div>
    </div>
    </div>

    <div data-panel="limits" hidden>
    <div class="card">
      <div class="headline">What this machine will run</div>
      <div class="note">Turned-off work is never offered to this machine, so it is not
      declined over and over — the network is told what you chose.</div>
      <div class="choices" id="workloadChoices"></div>
    </div>

    <div class="card">
      <div class="field">
        <div class="label">Daily limit on compute time</div>
        <div class="hint">Once spent, this machine turns work down until the window rolls
        over. Zero means no limit.</div>
        <div class="controls">
          <input type="number" id="budgetMinutes" min="0" max="10080" step="5" value="0">
          <span class="hint">minutes per</span>
          <select id="budgetPer">
            <option value="day">day</option>
            <option value="hour">hour</option>
          </select>
        </div>
        <div class="meter" id="budgetMeter" hidden><div id="budgetFill"></div></div>
        <div class="hint" id="budgetUsed" hidden></div>
      </div>

      <div class="field">
        <div class="label">Only work between these hours</div>
        <div class="hint">Your machine's local time. Leave both the same for any hour.</div>
        <div class="controls">
          <input type="time" id="fromTime" value="00:00">
          <span class="hint">to</span>
          <input type="time" id="toTime" value="00:00">
        </div>
        <div class="choices" id="dayChoices"></div>
      </div>
    </div>

    <div class="card">
      <div class="headline">Leave the machine usable</div>
      <div class="note">Checked before every task. A reading this machine cannot take is
      ignored rather than guessed at.</div>
      <div class="field">
        <div class="label">Skip work when already busy</div>
        <div class="hint">System load per core. 0 turns this off; 0.7 leaves headroom.</div>
        <div class="controls">
          <input type="number" id="maxLoad" min="0" max="16" step="0.1" value="0">
          <span class="hint" id="loadNow"></span>
        </div>
      </div>
      <div class="field">
        <div class="label">Always keep this much memory free</div>
        <div class="controls">
          <input type="number" id="minFreeRam" min="0" max="1048576" step="128" value="0">
          <span class="hint">MB</span>
          <span class="hint" id="ramNow"></span>
        </div>
      </div>
      <div class="field" id="batteryField">
        <div class="row">
          <div>
            <div class="label">Stop when on battery</div>
            <div class="hint" id="batteryHint">—</div>
          </div>
          <button id="batteryBtn">Off</button>
        </div>
      </div>
    </div>

    <div class="card">
      <button class="primary" id="saveLimits">Save limits</button>
      <div class="note" id="limitsNote">Saved on this machine. They work even when the
      network cannot be reached.</div>
      <div class="err" id="limitsErr" hidden></div>
    </div>
    </div>

    <div data-panel="settings" hidden>
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
          <div class="label" id="loginLabel">Start automatically when I log in</div>
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
          <div class="label">Leave this network</div>
          <div class="hint" id="leaveHint">Disconnects and forgets this network, so you can
          join a different one. Your computer keeps its identity; nothing else is removed.</div>
        </div>
        <button class="danger" id="leaveBtn">Leave</button>
      </div>
      <div class="row">
        <div>
          <div class="label">Stop the agent</div>
          <div class="hint" id="quitHint">Closing this window leaves it running. This stops it until next login.</div>
        </div>
        <button class="danger" id="quitBtn">Quit</button>
      </div>
    </div>
    <div class="err" id="actionErr" hidden></div>
    </div>
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
/**
 * Careful with backslash escapes below this point.
 *
 * Everything from here to the end of the page script is inside a TypeScript template
 * literal, so a \n written here is consumed at build time and emitted as a real
 * newline in the page. In an ordinary JS string literal that is a syntax error, and one
 * is enough to stop the entire inline script parsing -- which does not look like a
 * syntax error to a user. It
 * looks like an app frozen on "starting…", because no script ran at all to replace the
 * placeholder. Write \\n to emit an escape rather than a line break.
 */
function stoodDownHelp(platform) {
  if (platform === 'win32') return 'It is most likely one installed from PowerShell, at ' +
    '%USERPROFILE%\\.dwp\\bin\\dwp-agent.exe, started by a logon task. To hand over to this app, ' +
    'run these two lines in PowerShell and open this app again:\\n\\n' +
    '    schtasks /Delete /F /TN "DWP Agent"\\n' +
    '    Stop-Process -Name dwp-agent -Force'
  if (platform === 'darwin') return 'It is most likely one installed from a terminal, at ' +
    '~/.dwp/bin/dwp-agent. To hand over to this app, run this and open this app again:\\n\\n' +
    '    ~/.dwp/bin/dwp-agent uninstall-service; pkill -f dwp-agent'
  return 'It is most likely one installed from a terminal, at ~/.dwp/bin/dwp-agent. ' +
    'Stop that one, then open this app again.'
}

function describe(s) {
  if (s.stoodDown) return ['warn', 'Stopped — another copy is running',
    'Another agent on this computer already has this identity, so this one stood down ' +
    'rather than fight it for the connection. Your computer is still doing the work — ' +
    'the other copy is doing it, so nothing is broken.\\n\\n' + stoodDownHelp(s.platform)]
  if (s.paused) return ['warn', 'Paused', 'No work will be accepted until you resume.']
  if (s.connection === 'online' && s.running.length > 0) return ['busy', 'Working', null]
  if (s.connection === 'online') return ['ok', 'Connected', 'Waiting for work. Nothing to do right now.']
  if (s.connection === 'connecting') return ['wait', s.attempt > 1 ? 'Reconnecting…' : 'Connecting…', null]
  return ['bad', 'Offline', s.lastLostReason ? 'Last seen: ' + s.lastLostReason : 'Not connected.']
}

/** Durations people read at a glance: milliseconds, seconds, then minutes. */
function dur(ms) {
  if (!(ms > 0)) return '0 s'
  if (ms < 1000) return Math.round(ms) + ' ms'
  if (ms < 60000) return (ms / 1000).toFixed(1) + ' s'
  if (ms < 3600000) return Math.round(ms / 60000) + ' min'
  return (ms / 3600000).toFixed(1) + ' h'
}

function ago(at) {
  const secs = Math.max(0, Math.round((Date.now() - at) / 1000))
  if (secs < 60) return secs + 's ago'
  if (secs < 3600) return Math.round(secs / 60) + ' min ago'
  if (secs < 86400) return Math.round(secs / 3600) + ' h ago'
  return Math.round(secs / 86400) + ' d ago'
}

/** Reuse the status dot's three colours rather than invent a fourth vocabulary. */
function outcomeKind(outcome) {
  if (outcome === 'ok') return 'ok'
  if (outcome === 'error') return 'bad'
  return 'warn'
}

function outcomeWord(outcome) {
  if (outcome === 'result_too_large') return 'too large'
  return outcome
}

/**
 * Everything in here is built with createElement and textContent, never innerHTML.
 * Adapter names, task ids and adapter error messages all arrive from the network, and
 * this page's whole job is to display them.
 */
function mk(tag, cls, text) {
  const node = document.createElement(tag)
  if (cls) node.className = cls
  if (text !== undefined) node.textContent = text
  return node
}

function renderHistory(s) {
  const h = s.history || { recent: [], summary: null }
  const runs = h.recent || []
  const sum = h.summary || { runs: 0, failed: 0, busyMs: 0, byAdapter: {} }
  const strip = $('histStrip')
  const list = $('histRuns')
  const adapters = $('histAdapters')

  if (sum.runs === 0) {
    $('histSummary').textContent = 'Nothing has run on this computer yet.'
    show(strip, false)
    show(list, false)
    show(adapters, false)
    return
  }

  const parts = [sum.runs + (sum.runs === 1 ? ' run' : ' runs')]
  if (sum.failed > 0) parts.push(sum.failed + ' failed')
  parts.push(dur(sum.busyMs) + ' busy')
  // lastRunAt belongs to this process, so it is null for the whole of the first poll
  // after a restart even though the history is right there. Fall back to the newest
  // record rather than drop the clause until the next task arrives.
  const last = s.lastRunAt || (runs[0] ? Date.parse(runs[0].finishedAt) : 0)
  if (last) parts.push('last ' + ago(last))
  $('histSummary').textContent = parts.join(' · ')

  // Oldest on the left, newest on the right; recent() hands them over newest first.
  const ordered = runs.slice().reverse()
  let longest = 0
  for (const r of ordered) if (r.durationMs > longest) longest = r.durationMs
  strip.textContent = ''
  for (const r of ordered) {
    const bar = mk('div', outcomeKind(r.outcome))
    const share = longest > 0 ? Math.max(1, (r.durationMs / longest) * 25) : 1
    bar.style.width = share.toFixed(2) + '%'
    bar.title = r.adapter + ' · ' + dur(r.durationMs) + ' · ' + outcomeWord(r.outcome)
    strip.appendChild(bar)
  }
  show(strip, true)

  list.textContent = ''
  for (const r of runs.slice(0, 10)) {
    const row = mk('div', 'run')
    row.appendChild(mk('code', null, r.adapter))
    row.appendChild(mk('span', 'mark ' + outcomeKind(r.outcome), outcomeWord(r.outcome)))
    let cost = dur(r.durationMs) + ' · cpu ' + dur(r.cpuMs) + ' · ' + Math.round(r.rssMb) + ' MB'
    if (r.shared) cost += ' (shared)'
    row.appendChild(mk('span', 'cost', cost))
    row.appendChild(mk('span', 'when', ago(Date.parse(r.finishedAt))))
    if (r.message) row.title = r.message
    list.appendChild(row)
  }
  show(list, true)

  const names = Object.keys(sum.byAdapter)
  names.sort((a, b) => sum.byAdapter[b].runs - sum.byAdapter[a].runs)
  adapters.textContent = names.map(name =>
    name + ': ' + sum.byAdapter[name].runs + (sum.byAdapter[name].runs === 1 ? ' run, ' : ' runs, ') +
    dur(sum.byAdapter[name].busyMs)).join(' · ')
  show(adapters, names.length > 0)
}

// ------------------------------------------------------------------ tabs

var ADAPTER_NAMES = {
  echo: 'Connection test',
  walker_evolution: 'Walker simulation',
  cpu_inference_batch: 'Machine learning',
  remote_browser_session: 'Browser sessions',
}
var DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function showTab(name) {
  var tabs = document.querySelectorAll('.tabs button')
  for (var i = 0; i < tabs.length; i++) tabs[i].classList.toggle('on', tabs[i].dataset.tab === name)
  var panels = document.querySelectorAll('[data-panel]')
  for (var j = 0; j < panels.length; j++) panels[j].hidden = panels[j].dataset.panel !== name
}

document.querySelector('.tabs').addEventListener('click', function (e) {
  if (e.target.dataset && e.target.dataset.tab) showTab(e.target.dataset.tab)
})

// ------------------------------------------------------------------ limits

/**
 * Stop the one-second poll from overwriting what someone is in the middle of typing.
 *
 * The page re-renders from the server every second. Without this flag, changing a number
 * and pausing for a moment would have the field snap back to the saved value under the
 * cursor -- which reads as the app fighting you, and is exactly the bug that makes a
 * settings screen feel broken.
 */
var limitsDirty = false
var limitsReady = false

function markDirty() { limitsDirty = true; show($('limitsErr'), false) }

function dayChip(day, on) {
  var label = mk('label', on ? 'on' : '', '')
  var box = document.createElement('input')
  box.type = 'checkbox'
  box.checked = on
  box.dataset.day = String(day)
  box.onchange = function () { label.classList.toggle('on', box.checked); markDirty() }
  label.appendChild(box)
  label.appendChild(document.createTextNode(DAY_NAMES[day]))
  return label
}

function workloadChip(adapter, on) {
  var label = mk('label', on ? 'on' : '', '')
  var box = document.createElement('input')
  box.type = 'checkbox'
  box.checked = on
  box.dataset.adapter = adapter
  box.onchange = function () { label.classList.toggle('on', box.checked); markDirty() }
  label.appendChild(box)
  label.appendChild(document.createTextNode(ADAPTER_NAMES[adapter] || adapter))
  return label
}

function fillLimits(s) {
  var limits = s.limits || {}
  var chosen = limits.workloads || s.adapters
  var host = $('workloadChoices')
  host.textContent = ''
  for (var i = 0; i < s.adapters.length; i++) {
    host.appendChild(workloadChip(s.adapters[i], chosen.indexOf(s.adapters[i]) >= 0))
  }
  var budget = limits.budget || { minutes: 0, per: 'day' }
  $('budgetMinutes').value = String(budget.minutes || 0)
  $('budgetPer').value = budget.per || 'day'
  var schedule = limits.schedule || { from: '00:00', to: '00:00' }
  $('fromTime').value = schedule.from || '00:00'
  $('toTime').value = schedule.to || '00:00'
  var days = schedule.days || []
  var dayHost = $('dayChoices')
  dayHost.textContent = ''
  for (var d = 0; d < 7; d++) dayHost.appendChild(dayChip(d, days.length === 0 || days.indexOf(d) >= 0))
  var pressure = limits.pressure || {}
  $('maxLoad').value = String(pressure.maxLoadPerCore || 0)
  $('minFreeRam').value = String(pressure.minFreeRamMb || 0)
  $('batteryBtn').textContent = pressure.notOnBattery ? 'On' : 'Off'
  limitsReady = true
}

function readLimits(s) {
  var workloads = []
  var boxes = $('workloadChoices').querySelectorAll('input')
  for (var i = 0; i < boxes.length; i++) if (boxes[i].checked) workloads.push(boxes[i].dataset.adapter)

  var days = []
  var dayBoxes = $('dayChoices').querySelectorAll('input')
  for (var j = 0; j < dayBoxes.length; j++) if (dayBoxes[j].checked) days.push(Number(dayBoxes[j].dataset.day))

  var limits = {}
  // Only send what differs from "no limit", so a config stays readable and an unset
  // field is absent rather than present-and-zero.
  if (workloads.length > 0 && workloads.length < s.adapters.length) limits.workloads = workloads
  var minutes = Number($('budgetMinutes').value)
  if (minutes > 0) limits.budget = { minutes: minutes, per: $('budgetPer').value }
  var from = $('fromTime').value
  var to = $('toTime').value
  if (from && to && from !== to) {
    limits.schedule = { from: from, to: to }
    if (days.length > 0 && days.length < 7) limits.schedule.days = days
  }
  var pressure = {}
  var load = Number($('maxLoad').value)
  if (load > 0) pressure.maxLoadPerCore = load
  var ram = Number($('minFreeRam').value)
  if (ram > 0) pressure.minFreeRamMb = ram
  if ($('batteryBtn').textContent === 'On') pressure.notOnBattery = true
  if (Object.keys(pressure).length > 0) limits.pressure = pressure
  return limits
}

function renderLimits(s) {
  if (!limitsReady || !limitsDirty) fillLimits(s)

  var b = s.budgetState
  show($('budgetMeter'), Boolean(b))
  show($('budgetUsed'), Boolean(b))
  if (b) {
    var pct = Math.min(100, Math.round((b.usedMs / b.capMs) * 100))
    $('budgetFill').style.width = pct + '%'
    $('budgetUsed').textContent = dur(b.usedMs) + ' of ' + dur(b.capMs) + ' used this ' + b.per
  }
  var live = s.live || {}
  $('loadNow').textContent = live.loadPerCore === null || live.loadPerCore === undefined
    ? 'this machine does not report load'
    : 'now ' + live.loadPerCore.toFixed(2)
  $('ramNow').textContent = 'now ' + live.freeRamMb + ' MB free'
  // A container has no battery to look at, so the control says so rather than pretending.
  var known = live.onBattery !== null && live.onBattery !== undefined
  $('batteryBtn').disabled = !known
  $('batteryHint').textContent = known
    ? (live.onBattery ? 'On battery right now.' : 'On mains right now.')
    : 'This machine cannot see a battery, so this has no effect here.'
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
  /**
   * "updates verified" is a promise about a mechanism a container does not use, so in a
   * container the version says where the version comes from instead: the image tag.
   */
  var rtEarly = s.runtime || { container: false }
  $('version').textContent = 'v' + s.version + (rtEarly.container
    ? ' — from the image'
    : (s.pinnedKey ? ' — updates verified' : ' — updates unsigned, manual only'))

  renderHistory(s)
  renderLimits(s)

  /**
   * "Connected, nothing to do" and "connected, but I am turning work down" look the
   * same from outside, and only one of them is something the owner can act on.
   */
  var declined = s.lastDeclined
  var recent = declined && (Date.now() - declined.at) < 10 * 60 * 1000
  show($('whyIdle'), Boolean(recent) && s.running.length === 0)
  if (recent && s.running.length === 0) {
    $('whyIdle').textContent = 'Turned work down ' + ago(declined.at) + ' — ' + declined.detail + '.'
  }

  show($('retryRow'), s.stoodDown)
  // Concatenation, not a template literal: this whole script sits inside one, so a
  // substitution written here would be resolved by the compiler, not the browser.
  $('leaveHint').textContent = 'Disconnects and forgets ' + s.server +
    ', so you can join a different network. Your computer keeps its identity; nothing else is removed.'
  $('pauseLabel').textContent = s.paused ? 'Paused' : 'Accepting work'
  $('pauseBtn').textContent = s.paused ? 'Resume' : 'Pause'
  /**
   * Three of these rows ask a question a container cannot answer, so in a container they
   * state the answer instead. The alternative -- hiding them -- leaves an operator
   * hunting for controls that are simply somewhere else now.
   */
  var rt = rtEarly
  show($('runsInLabel'), Boolean(rt.container))
  show($('runsIn'), Boolean(rt.container))
  if (rt.container) $('runsIn').textContent = (rt.image ? rt.image + ' — ' : '') + 'container ' + (rt.id || '?')

  show($('loginBtn'), !rt.container)
  $('loginLabel').textContent = rt.container ? 'Restarting' : 'Start automatically when I log in'
  if (rt.container) {
    $('loginHint').textContent = rt.supervisor
  } else {
    $('loginBtn').textContent = s.runsAtLogin ? 'Turn off' : 'Turn on'
    $('loginHint').textContent = s.runsAtLogin
      ? 'On. It joins by itself after a restart, with no window open.'
      : 'Off. It only runs while this app is open.'
  }

  show($('updateBtn'), !rt.container)
  $('updateHint').textContent = s.updating ? 'Installing an update. It will restart itself.' : s.updateNote

  $('quitHint').textContent = rt.container
    ? 'Stops the agent inside this container. Whether it comes back is your restart policy.'
    : 'Closing this window leaves it running. This stops it until next login.'
}

let rendered = false
let misses = 0
async function tick() {
  if (quitting) return
  try {
    render(await api('state'))
    rendered = true
    misses = 0
    show($('footer'), true)
    $('footer').textContent = 'Closing this window does not stop the agent.'
  } catch {
    // A restart after an update lands here for a second or two, so the first few
    // failures are not worth alarming anyone about. But if render() has never run, the
    // page is still showing its "starting…" placeholder and will show it forever --
    // which is indistinguishable from a hang. Say what happened and what to do.
    misses += 1
    $('footer').textContent = 'Reconnecting to the agent…'
    if (!rendered && misses >= 5) {
      $('sub').textContent = 'cannot reach the agent'
      show($('footer'), true)
      $('footer').textContent =
        'The agent is not responding on this address. Close this window and open DWP Agent again.'
    }
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
    const r = await api('pair', {
      invite: $('invite').value,
      label: $('name').value,
      runAtLogin: $('runAtLogin').checked,
    })
    // Joined, but not durably. Say so once here rather than leave the difference to be
    // noticed the next time this computer is restarted and does not come back.
    if (r && r.serviceError) {
      $('actionErr').textContent = 'Joined, but could not set it to rejoin after a restart:\\n' +
        r.serviceError + '\\n\\nTurn it on below once that is sorted.'
      show($('actionErr'), true)
    }
  } catch (err) {
    $('joinErr').textContent = err.message
    show($('joinErr'), true)
  }
})
let leaveArmed = false
$('leaveBtn').onclick = () => {
  if (!leaveArmed) {
    leaveArmed = true
    $('leaveBtn').textContent = 'Really leave?'
    setTimeout(() => { leaveArmed = false; $('leaveBtn').textContent = 'Leave' }, 4000)
    return
  }
  leaveArmed = false
  $('leaveBtn').textContent = 'Leave'
  act($('leaveBtn'), () => api('leave', {}))
}
$('retryBtn').onclick = () => act($('retryBtn'), () => api('retry', {}))
$('batteryBtn').onclick = function () {
  $('batteryBtn').textContent = $('batteryBtn').textContent === 'On' ? 'Off' : 'On'
  markDirty()
}
var limitInputs = ['budgetMinutes', 'budgetPer', 'fromTime', 'toTime', 'maxLoad', 'minFreeRam']
for (var li = 0; li < limitInputs.length; li++) $(limitInputs[li]).oninput = markDirty
$('saveLimits').onclick = () => act($('saveLimits'), async () => {
  show($('limitsErr'), false)
  var current = await api('state')
  try {
    var saved = await api('limits', { limits: readLimits(current) })
    limitsDirty = false
    limitsReady = false
    $('limitsNote').textContent = saved && saved.reconnecting
      ? 'Saved. Reconnecting so the network knows what this machine now accepts.'
      : 'Saved on this machine. They work even when the network cannot be reached.'
  } catch (err) {
    $('limitsErr').textContent = err.message
    show($('limitsErr'), true)
    throw err
  }
})
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
    lastLostReason: null, advice: null, stoodDown: false, updating: false, lastRunAt: null,
    lastDeclined: null,
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
     * an address reaching this server would otherwise let a page in their browser drive
     * it. It matches on the name alone (see `hostAllowed`), because the port is not a
     * secret and requiring a particular one broke every published container whose host
     * port differs from its container port. The token in the path stops any other local
     * process or page that has not read a file only this user can read — and anything
     * that *can* read it could already read the agent's private key sitting beside it,
     * so this grants nothing new.
     */
    const port = (server.address() as { port: number } | null)?.port
    if (!hostAllowed(req.headers.host)) {
      send(res, 403, { error: 'wrong host' })
      return
    }

    const url = new URL(req.url ?? '/', `http://127.0.0.1:${port}`)
    const [, given, ...rest] = url.pathname.split('/')
    const supplied = Buffer.from(given ?? '')
    const authorised = supplied.length === tokenBuf.length && timingSafeEqual(supplied, tokenBuf)
    if (!authorised) {
      /**
       * A bare 404 here is a dead end, and it is reached by the ordinary route.
       *
       * Two agents on one machine is normal — a desktop install and a container, say —
       * and they hold different tokens on different ports. Opening the right token
       * against the wrong port then produces a blank "not found" that says nothing about
       * which of the two things is wrong, and the answer is invisible from the browser.
       * A container makes this the common case, because it prints its *internal* port.
       *
       * Only for a request that wants HTML, i.e. someone looking at it. Machine callers
       * keep the opaque JSON. This reveals nothing a TCP connect did not already: that
       * something is listening. It does not say whether the token was close.
       */
      if (req.method === 'GET' && (req.headers.accept ?? '').includes('text/html')) {
        res.writeHead(404, {
          'content-type': 'text/html; charset=utf-8',
          'cache-control': 'no-store',
          'content-security-policy': "default-src 'none'; style-src 'unsafe-inline'",
          'referrer-policy': 'no-referrer',
        })
        res.end('<!doctype html><meta charset="utf-8">'
          + '<style>body{font:16px/1.6 system-ui;max-width:34rem;margin:15vh auto;padding:0 1.5rem;'
          + 'color:#14181f;background:#f6f7f9}code{background:#e3e6eb;padding:.1em .3em;border-radius:3px}'
          + '@media(prefers-color-scheme:dark){body{color:#eef1f5;background:#14181f}'
          + 'code{background:#2b323d}}</style>'
          + '<h2>Wrong address for this agent</h2>'
          + '<p>A DWP agent is listening here, but not on this path. Every agent has its own '
          + 'single-use address, and more than one can run on a machine — a desktop install '
          + 'and a container, for instance, on different ports.</p>'
          + '<p>Ask the one you want for its address:</p>'
          + '<p><code>docker logs dwp-agent | grep Window</code><br>'
          + '<code>' + escapeHtml(invocation()) + ' gui-address</code></p>'
          + '<p>If that prints a port you did not publish, it is the container’s own. '
          + 'Use the host port from your <code>-p</code> flag and keep the token unchanged.</p>')
        return
      }
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
        /**
         * The installed *release*, falling back to the compile-time stamp.
         *
         * These differ, and showing only the stamp made "Check now" look broken: a
         * binary is built with package.json's version (0.4.0) while the release that
         * carries it is 0.4.0+<hash>, so a successful update left the screen unchanged
         * and the button appeared to do nothing. update.ts already compares the full
         * release string, so the agent knew -- only the display was behind.
         */
        version: config?.installedRelease ?? AGENT_VERSION,
        buildVersion: AGENT_VERSION,
        paired: config !== null,
        label: config?.label ?? null,
        hostId: config?.hostId ?? null,
        server: config?.server ?? null,
        pinnedKey: Boolean(config?.releaseKey),
        paused: isPaused(),
        platform: platform(),
        adapters: availableAdapters(),
        runsAtLogin: service.installed,
        /**
         * How this agent is packaged and who restarts it.
         *
         * The window has three rows that only make sense on a laptop — start at login,
         * install an update, quit until next login — and all three are actively
         * misleading in a container, where the answer to every one of them is "the
         * container runtime decides". Rather than hide them and leave an operator
         * wondering where the controls went, the page swaps in what is true here.
         */
        runtime: {
          container: isContainer(),
          supervisor: supervisorNote(),
          /** The container's own id and image, so one window is identifiably one agent. */
          ...(containerIdentity() ?? {}),
        },
        /**
         * Say which of the two it is, because they need different people to act.
         *
         * This used to read "This network offers no signed releases", which points at
         * the server when the truth is local: the network does publish signed releases,
         * and this computer simply joined before it did, so it pinned no key and refuses
         * every update. Blaming the network meant nobody ever ran the one command that
         * fixes it, on the one machine that can.
         */
        updateNote: isContainer()
          ? 'This agent is the image it was started from. Update it by pulling a newer '
            + 'image and recreating the container — nothing here rewrites itself, so what '
            + 'the registry holds and what is running can never drift apart.'
          : config?.releaseKey
            ? 'Installed automatically, verified against the key this computer pinned when it joined.'
            : 'This computer joined before the network signed its releases, so it pinned no key '
              + 'and cannot verify an update. Run  ' + invocation() + ' trust-updates  here to '
              + 'review the key and turn automatic updates back on.',
        connection: state.connection,
        attempt: state.attempt,
        connectedSince: state.connectedSince,
        running: state.running,
        lastLostReason: state.lastLostReason,
        advice: state.advice,
        stoodDown: state.stoodDown,
        updating: state.updating,
        lastRunAt: state.lastRunAt,
        /** Why the last offer was turned down, so an idle machine can explain itself. */
        lastDeclined: state.lastDeclined,
        /**
         * What this machine has run before now.
         *
         * Answered from the in-memory array, so polling this every second costs a JSON
         * encode and nothing else. Thirty records is more than the page draws in its
         * list, because the timeline strip shows the lot.
         */
        history: { recent: history.recent(30), summary: history.summary() },
        /** What the owner set, so the form shows the truth rather than its own defaults. */
        limits: config?.limits ?? {},
        /** Progress against the duty budget, or null when there is no budget. */
        budgetState: budgetState(config?.limits),
        /**
         * The same readings the rules are evaluated against, so the window can show what
         * a threshold is being compared to. A guard whose current value is invisible is
         * one nobody can set sensibly.
         */
        live: { loadPerCore: loadPerCore(), freeRamMb: freeRamMb(), onBattery: onBattery() },
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

      /**
       * Register the login task as part of joining, not as a setting to find later.
       *
       * The toggle below has always existed and was the only place this surfaced, which
       * meant the ordinary path through this window produced an agent that lived exactly
       * as long as the window did. Anyone who does not want that unticks the box; the
       * default is the one that makes the machine a worker rather than a visitor.
       *
       * A failure here does not fail the join. The computer has enrolled either way, and
       * losing a successful pairing over a Scheduled Task would be a far worse trade —
       * so it is reported as something that did not happen, and the toggle is left
       * showing the truth.
       */
      let serviceError: string | null = null
      if (body.runAtLogin !== false && config) {
        try {
          await installService('gui')
        } catch (err) {
          serviceError = err instanceof Error ? err.message : String(err)
          log.warn('gui.login_task_failed', { detail: serviceError })
        }
        await refreshService()
      }
      send(res, 200, { ok: true, runsAtLogin: service.installed, serviceError })
      return
    }

    /**
     * Save the owner's limits.
     *
     * Validated here rather than trusted from the page, because the page is not the only
     * thing that can POST to this port -- anything that can read the token can, and a
     * malformed limit would be persisted into the config and then applied to every offer
     * from then on. Unknown fields are dropped rather than rejected: a newer window
     * talking to an older agent should lose the setting it does not understand, not fail
     * to save the ones it does.
     */
    if (route === 'api/limits') {
      if (!config) { send(res, 400, { error: 'Join a network first.' }); return }
      let limits: Limits
      try {
        limits = sanitiseLimits(body.limits)
      } catch (err) {
        send(res, 400, { error: err instanceof Error ? err.message : String(err) })
        return
      }
      const before = JSON.stringify(config.limits?.workloads ?? null)
      /**
       * Mutated in place, not replaced.
       *
       * `connect()` closed over this exact object when the connection was established
       * and reads `cfg.limits` on every offer. Assigning a fresh object here updates
       * what the window shows and what is written to disk, and leaves the running agent
       * holding the previous one — so the limits saved, displayed correctly, persisted
       * across a restart, and did nothing at all until the process was restarted.
       * Measured: a task accepted by a machine whose load ceiling it was three hundred
       * times over.
       */
      config.limits = limits
      saveConfig(config)
      log.info('gui.limits_saved', { limits })
      /**
       * Only reconnect when the advertised set actually changed. Everything else is
       * enforced locally on the next offer, and dropping a healthy connection to apply
       * a memory threshold would interrupt work for no reason.
       */
      const reconnecting = JSON.stringify(limits.workloads ?? null) !== before
      if (reconnecting) agent?.refresh()
      send(res, 200, { ok: true, reconnecting })
      return
    }

    if (route === 'api/retry') {
      if (!agent) { send(res, 400, { error: 'Join a network first.' }); return }
      agent.retryNow()
      log.info('gui.retry_requested')
      send(res, 200, { ok: true })
      return
    }

    /**
     * Leave the network this computer joined, so it can join a different one.
     *
     * Without this the only way out was deleting ~/.dwp/config.json by hand, because
     * pairing refuses once a config exists — so a machine pointed at the wrong network
     * could not be moved by the person sitting in front of it. Changing networks is a
     * different operation from `set-server`, which only follows an address a network
     * already known to this machine has moved to: a different network has never seen
     * this host's key, so it must be joined from scratch.
     */
    if (route === 'api/leave') {
      if (!config) { send(res, 400, { error: 'This computer has not joined a network.' }); return }
      const was = config.server
      agent?.stop()
      agent = null
      connected = false
      clearConfig()
      config = null
      state = {
        connection: 'offline', attempt: 0, connectedSince: null, running: [],
        lastLostReason: null, advice: null, stoodDown: false, updating: false,
        // Leaving forgets a network, not what this computer has done.
        lastRunAt: state.lastRunAt,
        lastDeclined: null,
      }
      log.info('gui.left_network', { was })
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
      if (isContainer()) {
        send(res, 400, {
          error: 'There is no login inside a container. Whether this agent comes back is '
            + 'your container runtime\'s restart policy — set `restart: unless-stopped`.',
        })
        return
      }
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
      /**
       * Refuse here as well as in update.ts, and say the same thing.
       *
       * An update inside a container writes into a layer that the next `docker run`
       * throws away, so it would appear to work, survive a restart of the process, and
       * vanish on a recreate — leaving the running version and the image's version
       * permanently disagreeing with no way to tell which one you are looking at.
       */
      if (isContainer()) {
        send(res, 200, {
          message: 'Updates arrive as images here. Run  docker compose pull && docker compose up -d  '
            + 'on this host to move to a newer agent.',
        })
        return
      }
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
  const bindHost = guiBindHost()
  const listenOn = (port: number): Promise<boolean> => new Promise(resolve => {
    const onError = (err: NodeJS.ErrnoException): void => {
      server.removeListener('error', onError)
      if (err.code === 'EADDRINUSE') resolve(false)
      else resolve(false)
    }
    server.once('error', onError)
    server.listen(port, bindHost, () => {
      server.removeListener('error', onError)
      resolve(true)
    })
  })

  let bound = 0
  /**
   * A pinned port is taken or the agent stops, rather than quietly moving.
   *
   * Walking the range is right on a laptop, where the alternative is refusing to start
   * because something unrelated holds 43117. It is wrong in a container: the port is
   * already written down in the operator's `-p 43117:43117`, nothing else in that
   * namespace can be holding it, and landing on 43118 publishes a port with nothing
   * behind it — a window that shows "cannot reach the agent" forever while the agent is
   * perfectly healthy two ports away.
   */
  const pinned = guiPort()
  if (pinned !== null) {
    for (let i = 0; i < 20 && bound === 0; i += 1) {
      if (await listenOn(pinned)) bound = pinned
      else await sleep(250)
    }
    if (bound === 0) {
      console.error(`\n  Could not open port ${pinned} on ${bindHost}. Something else is holding it.\n`)
      process.exit(1)
    }
  } else {
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
  }

  writeLock({ port: bound, token, pid: process.pid, startedAt: new Date().toISOString(), version: AGENT_VERSION })
  /**
   * What to print. Inside a container the bound port is the *container's*, which is not
   * where anyone can reach it if the operator mapped it to a different host port.
   */
  const url = `${guiPublicOrigin() ?? `http://127.0.0.1:${bound}`}/${token}/`
  log.info('gui.listening', { port: bound, bindHost, container: isContainer(), hidden: opts.hidden, paired: config !== null })

  /**
   * Stop by handing work back, not by vanishing.
   *
   * This used to be a bare `process.exit(0)`, and because it is registered before the
   * connection exists it also pre-empted the agent's own handler — so `docker stop`
   * returned in under a fifth of a second with the server still believing this machine
   * held its task, which then sat idle until its 45-second lease expired. A second
   * signal still exits at once, for anyone who means it.
   */
  let stopping = false
  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.on(signal, () => {
      if (stopping) process.exit(0)
      stopping = true
      void (agent?.handOff() ?? Promise.resolve()).finally(() => process.exit(0))
    })
  }

  /**
   * Join from the environment before connecting, for a machine with nobody at it.
   *
   * Deliberately after the window is listening: if the invite turns out to be spent or
   * the address wrong, the operator can open the page and see exactly that, instead of
   * the container exiting and taking the explanation with it into a log they have to go
   * looking for.
   */
  if (!config) {
    const joined = await autoEnrol()
    if (joined?.ok) config = loadConfig()
  }

  if (config) await startConnection(config)

  if (opts.hidden) {
    console.log(`  DWP Agent running in the background. Window: ${url}`)
    if (isContainer() && !guiPublicOrigin()) {
      console.log(`  (that is this container's own port ${bound}; on the host it is whatever`
        + ` you mapped it to, e.g. -p 127.0.0.1:43118:${bound} means http://127.0.0.1:43118/${token}/)`)
    }
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
