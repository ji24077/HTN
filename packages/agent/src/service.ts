import { execFile } from 'node:child_process'
import { existsSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { homedir, platform } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { AGENT_HOME, isCompiledBinary } from './paths.ts'

const exec = promisify(execFile)

/**
 * Run at login, so nobody has to keep a terminal window open.
 *
 * This is the difference between "a thing I remember to start" and "a computer that
 * contributes" — and it is the complaint a non-technical person actually has, well
 * before they want a tray icon.
 *
 * Everything here is per-user: a LaunchAgent, a systemd *user* unit, a Scheduled Task
 * at logon. No sudo, no root daemon, nothing installed system-wide. An agent that runs
 * as the person who chose to install it can be stopped by that person, and cannot touch
 * anything they could not touch themselves.
 */

const LABEL = 'com.dwp.agent'

export type ServiceStatus =
  | { installed: false; platform: string }
  | { installed: true; platform: string; running: boolean; path: string }

/**
 * How the agent should start at login.
 *
 * `run` is the headless agent, for a terminal install. `gui` starts the desktop app's
 * process *without* opening a window — the window is opened on demand by launching the
 * app again, which hands off to this already-running process.
 *
 * The distinction matters because the two must never both be installed: they would be
 * two agents claiming one host identity, and the server hands the connection to whichever
 * connected last. Choosing per install rather than running both is what keeps that
 * impossible instead of merely unlikely.
 */
export type ServiceMode = 'run' | 'gui'

/** The command the service should run — the binary, or node plus this source tree. */
function launchCommand(mode: ServiceMode): string[] {
  const args = mode === 'gui' ? ['gui', '--hidden'] : ['run']
  if (isCompiledBinary()) return [process.execPath, ...args]
  return [process.execPath, join(process.cwd(), 'packages', 'agent', 'src', 'index.ts'), ...args]
}

// ------------------------------------------------------------------- macOS

const plistPath = (): string => join(homedir(), 'Library', 'LaunchAgents', `${LABEL}.plist`)

function macPlist(mode: ServiceMode): string {
  const args = launchCommand(mode)
  const argXml = args.map(a => `      <string>${a.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</string>`).join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
${argXml}
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key>
    <dict>
      <!-- Restart if it exits, but not if it was deliberately stopped. -->
      <key>SuccessfulExit</key><false/>
    </dict>
    <key>EnvironmentVariables</key>
    <dict>
      <key>DWP_HOME</key><string>${AGENT_HOME}</string>
      <key>DWP_DNS_FALLBACK</key><string>1</string>
    </dict>
    <key>StandardOutPath</key><string>${join(AGENT_HOME, 'service.log')}</string>
    <key>StandardErrorPath</key><string>${join(AGENT_HOME, 'service.log')}</string>
    <key>ProcessType</key><string>Background</string>
  </dict>
</plist>
`
}

// ------------------------------------------------------------------- Linux

const unitPath = (): string => join(homedir(), '.config', 'systemd', 'user', 'dwp-agent.service')

function linuxUnit(mode: ServiceMode): string {
  const args = launchCommand(mode).map(a => `"${a}"`).join(' ')
  return `[Unit]
Description=Distributed work agent
After=network-online.target

[Service]
Type=simple
ExecStart=${args}
Environment=DWP_HOME=${AGENT_HOME}
Environment=DWP_DNS_FALLBACK=1
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
`
}

// ----------------------------------------------------------------- Windows

const TASK_NAME = 'DWP Agent'

// ------------------------------------------------------------------- public

export async function installService(mode: ServiceMode = 'run'): Promise<string> {
  const os = platform()
  const [exe, ...rest] = launchCommand(mode)

  if (os === 'darwin') {
    mkdirSync(join(homedir(), 'Library', 'LaunchAgents'), { recursive: true })
    writeFileSync(plistPath(), macPlist(mode))
    // bootout first so a reinstall replaces rather than collides with the old definition.
    await exec('launchctl', ['bootout', `gui/${process.getuid?.() ?? 501}/${LABEL}`]).catch(() => {})
    await exec('launchctl', ['bootstrap', `gui/${process.getuid?.() ?? 501}`, plistPath()])
    return plistPath()
  }

  if (os === 'linux') {
    mkdirSync(join(homedir(), '.config', 'systemd', 'user'), { recursive: true })
    writeFileSync(unitPath(), linuxUnit(mode))
    await exec('systemctl', ['--user', 'daemon-reload'])
    await exec('systemctl', ['--user', 'enable', '--now', 'dwp-agent.service'])
    // Without lingering the unit stops when the user logs out, which defeats the point.
    await exec('loginctl', ['enable-linger', process.env.USER ?? '']).catch(() => {})
    return unitPath()
  }

  if (os === 'win32') {
    const command = rest.length > 0 ? `"${exe}" ${rest.map(a => `"${a}"`).join(' ')}` : `"${exe}"`
    await exec('schtasks', [
      '/Create', '/F', '/SC', 'ONLOGON', '/TN', TASK_NAME, '/TR', command, '/RL', 'LIMITED',
    ])
    return TASK_NAME
  }

  throw new Error(`no service support for ${os}`)
}

export async function uninstallService(): Promise<void> {
  const os = platform()
  if (os === 'darwin') {
    await exec('launchctl', ['bootout', `gui/${process.getuid?.() ?? 501}/${LABEL}`]).catch(() => {})
    rmSync(plistPath(), { force: true })
    return
  }
  if (os === 'linux') {
    await exec('systemctl', ['--user', 'disable', '--now', 'dwp-agent.service']).catch(() => {})
    rmSync(unitPath(), { force: true })
    await exec('systemctl', ['--user', 'daemon-reload']).catch(() => {})
    return
  }
  if (os === 'win32') {
    await exec('schtasks', ['/Delete', '/F', '/TN', TASK_NAME]).catch(() => {})
    return
  }
  throw new Error(`no service support for ${os}`)
}

export async function serviceStatus(): Promise<ServiceStatus> {
  const os = platform()
  if (os === 'darwin') {
    if (!existsSync(plistPath())) return { installed: false, platform: os }
    const running = await exec('launchctl', ['print', `gui/${process.getuid?.() ?? 501}/${LABEL}`])
      .then(() => true).catch(() => false)
    return { installed: true, platform: os, running, path: plistPath() }
  }
  if (os === 'linux') {
    if (!existsSync(unitPath())) return { installed: false, platform: os }
    const running = await exec('systemctl', ['--user', 'is-active', 'dwp-agent.service'])
      .then(r => r.stdout.trim() === 'active').catch(() => false)
    return { installed: true, platform: os, running, path: unitPath() }
  }
  if (os === 'win32') {
    const found = await exec('schtasks', ['/Query', '/TN', TASK_NAME]).then(() => true).catch(() => false)
    if (!found) return { installed: false, platform: os }
    return { installed: true, platform: os, running: true, path: TASK_NAME }
  }
  return { installed: false, platform: os }
}
