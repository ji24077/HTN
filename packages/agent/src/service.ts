import { execFile } from 'node:child_process'
import { existsSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { homedir, platform } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { AGENT_HOME, isCompiledBinary } from './paths.ts'
import { isContainer } from './runtime.ts'

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
const TASK_XML_PATH = (): string => join(AGENT_HOME, 'task.xml')

const xmlEscape = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

/**
 * Who the task runs as, as Task Scheduler spells it.
 *
 * Omitted entirely when the environment cannot say, rather than guessed: an XML naming
 * the wrong user is rejected outright, where an XML naming nobody lets schtasks fill in
 * the account that is creating the task, which is the one we want anyway.
 */
function currentUserId(): string | null {
  const user = process.env.USERNAME
  if (!user) return null
  const domain = process.env.USERDOMAIN
  return domain ? `${domain}\\${user}` : user
}

/**
 * The Scheduled Task, written out in full rather than left to `schtasks` defaults.
 *
 * This is the whole reason the Windows path is not two lines. A task created by
 * `schtasks /Create /SC ONLOGON` inherits the Task Scheduler schema's defaults, and
 * three of them are actively wrong for a laptop:
 *
 *   DisallowStartIfOnBatteries  true  — it will not start unless the machine is plugged in
 *   StopIfGoingOnBatteries      true  — it is killed the moment the charger comes out
 *   ExecutionTimeLimit          P3D   — it is killed after three days, plugged in or not
 *
 * A worker that quietly stops when someone unplugs their laptop is indistinguishable
 * from a broken one, and it is the failure this project actually hit: a Windows machine
 * that joined, ran a task, and was gone minutes later. `schtasks` has no flags for any
 * of these, so the definition has to arrive as XML.
 *
 * The repetition on the trigger is what makes this equivalent to the other two
 * platforms. macOS gets launchd's KeepAlive and Linux gets systemd's Restart=on-failure;
 * Task Scheduler has RestartOnFailure, but it only covers a task that *fails* and only a
 * few times. Re-firing the trigger every ten minutes with MultipleInstancesPolicy set to
 * IgnoreNew means a live agent is left alone — the extra start is dropped — while a dead
 * one is back within ten minutes, from any cause, indefinitely.
 */
function windowsTaskXml(mode: ServiceMode): string {
  const [exe = process.execPath, ...args] = launchCommand(mode)
  const userId = currentUserId()
  const userElement = userId ? `<UserId>${xmlEscape(userId)}</UserId>` : ''
  return `<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Joins this computer to the distributed work network at login.</Description>
    <URI>\\${TASK_NAME}</URI>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Repetition>
        <Interval>PT10M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <Enabled>true</Enabled>
      ${userElement}
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      ${userElement}
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>${xmlEscape(exe)}</Command>
      ${args.length > 0 ? `<Arguments>${xmlEscape(args.map(a => `"${a}"`).join(' '))}</Arguments>` : ''}
    </Exec>
  </Actions>
</Task>
`
}

/**
 * Register the task from XML, falling back to the flag form if that is refused.
 *
 * `schtasks /XML` is strict about the document it is handed and the ways it can object
 * are version-specific, so a rejection must not leave the machine with no task at all —
 * an agent that starts at login on battery-powered terms is still better than one that
 * never starts. The fallback says so in what it returns, because the difference decides
 * whether the machine can be relied on.
 */
async function installWindowsTask(mode: ServiceMode): Promise<string> {
  const xmlPath = TASK_XML_PATH()
  mkdirSync(AGENT_HOME, { recursive: true })
  // UTF-16LE with a BOM: what Task Scheduler itself exports, and the encoding schtasks
  // accepts without argument. UTF-8 is read as mojibake on some builds and rejected as
  // "incorrectly formatted", which is a confusing way to learn about an encoding.
  writeFileSync(xmlPath, '﻿' + windowsTaskXml(mode), 'utf16le')
  try {
    await exec('schtasks', ['/Create', '/F', '/TN', TASK_NAME, '/XML', xmlPath])
    return TASK_NAME
  } catch (err) {
    const detail = err instanceof Error ? err.message.split('\n')[0] : String(err)
    const [exe, ...rest] = launchCommand(mode)
    const command = rest.length > 0 ? `"${exe}" ${rest.map(a => `"${a}"`).join(' ')}` : `"${exe}"`
    await exec('schtasks', [
      '/Create', '/F', '/SC', 'ONLOGON', '/TN', TASK_NAME, '/TR', command, '/RL', 'LIMITED',
    ])
    return `${TASK_NAME} (basic — Windows refused the full definition: ${detail}. ` +
      `It will stop when this computer runs on battery.)`
  } finally {
    rmSync(xmlPath, { force: true })
  }
}

// ------------------------------------------------------------------- public

/**
 * What to say when asked to register a login service inside a container.
 *
 * Not an assertion failure and not silence: the request is reasonable, the answer is
 * that the question belongs to a different layer. A systemd unit written inside a
 * container is a file that nothing will ever read, and writing one anyway would report
 * success for a machine that does not in fact come back.
 */
const CONTAINER_REFUSAL =
  'This agent is running in a container, which has no login to start at. '
  + 'Whether it comes back is the container runtime\'s restart policy: '
  + 'use `restart: unless-stopped` in your compose file, or `--restart unless-stopped` with docker run.'

export async function installService(mode: ServiceMode = 'run'): Promise<string> {
  if (isContainer()) throw new Error(CONTAINER_REFUSAL)
  const os = platform()

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

  if (os === 'win32') return installWindowsTask(mode)

  throw new Error(`no service support for ${os}`)
}

export async function uninstallService(): Promise<void> {
  if (isContainer()) throw new Error(CONTAINER_REFUSAL)
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
  /**
   * A container is supervised by definition — something started it and something decides
   * whether to start it again. Reporting `installed: false` here made the window offer a
   * "start automatically at login" toggle that could only ever fail, on the one kind of
   * host where restarting is already handled.
   */
  if (isContainer()) {
    return { installed: true, platform: 'container', running: true, path: 'container runtime restart policy' }
  }
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
    /**
     * Ask the task what it is doing, rather than assuming a task that exists is running.
     *
     * The previous version reported `running: true` for any registered task, which meant
     * `service-status` was incapable of showing the one state worth seeing: installed and
     * dead. That is exactly the state a battery-stopped task sits in.
     *
     * The status word is localised, so a machine in another language falls through to
     * `true` rather than claiming the agent is down on the strength of not recognising a
     * word. Wrong in the direction that does not invent a problem.
     */
    const query = await exec('schtasks', ['/Query', '/TN', TASK_NAME, '/FO', 'LIST'])
      .then(r => r.stdout).catch(() => null)
    if (query === null) return { installed: false, platform: os }
    const status = /^\s*Status:\s*(.+)$/mi.exec(query)?.[1]?.trim()
    const known = status !== undefined && /^(running|ready|disabled)$/i.test(status)
    return { installed: true, platform: os, running: known ? /^running$/i.test(status!) : true, path: TASK_NAME }
  }
  return { installed: false, platform: os }
}
