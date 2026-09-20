/**
 * Platform-specific helpers for the operator scripts.
 *
 * Everything here is a no-op or an identity on macOS and Linux: the POSIX paths are the
 * ones that were already there, unchanged, and the Windows branches are additive. That
 * matters because the two are expected to run side by side — a Mac hosting the network
 * while Windows machines join it — so nothing in here may alter existing behaviour.
 */
import { execFileSync } from 'node:child_process'
import type { ChildProcess } from 'node:child_process'

export const IS_WINDOWS = process.platform === 'win32'

/** How to install cloudflared, in the words of whichever machine is asking. */
export const cloudflaredInstallHint = (): string =>
  IS_WINDOWS ? 'winget install --id Cloudflare.cloudflared'
    : process.platform === 'darwin' ? 'brew install cloudflared'
      : 'see https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/'

/** How to find whatever is already holding a port. */
export const portHolderHint = (port: number): string =>
  IS_WINDOWS ? `netstat -ano | findstr :${port}     (then: taskkill /PID <pid> /F)`
    : `lsof -nP -iTCP:${port} -sTCP:LISTEN`

/** How to stop a stray control service. */
export const stopServerHint = (): string =>
  IS_WINDOWS
    ? `Get-CimInstance Win32_Process -Filter "Name='node.exe'" | Where-Object CommandLine -like '*packages/control*' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`
    : `pkill -f 'packages/control/src/index.ts'`

/** How to copy the example environment file. */
export const copyEnvHint = (): string =>
  IS_WINDOWS ? 'copy .env.example .env' : 'cp .env.example .env'

/** How to set an environment variable for a single command. */
export const envPrefix = (key: string, value: string, command: string): string =>
  IS_WINDOWS ? `$env:${key}='${value}'; ${command}` : `${key}=${value} ${command}`

/**
 * Find the real executable behind a command name on Windows.
 *
 * Node spawns without a shell, and on Windows CreateProcess only ever appends `.exe`
 * while `.cmd`/`.bat` shims are refused outright. Tools installed through npm, corepack
 * or scoop are shims, so a bare spawn reports a working install as missing. Resolving to
 * an absolute path first keeps the spawned process a direct child, which a shell wrapper
 * would not — and a direct child is the only kind this script can reliably stop.
 *
 * Returns the command unchanged on every other platform.
 */
export function resolveCommand(command: string): string {
  if (!IS_WINDOWS) return command
  try {
    const found = execFileSync('where.exe', [command], { stdio: 'pipe' })
      .toString().split(/\r?\n/).map(l => l.trim()).filter(Boolean)
    // Prefer a real executable over a shim so the child can be signalled directly.
    return found.find(p => /\.exe$/i.test(p)) ?? found[0] ?? command
  } catch {
    return command
  }
}

/**
 * Stop a child process and everything it started.
 *
 * On POSIX this is exactly the SIGTERM that was here before. Windows has no SIGTERM to
 * deliver and Node kills only the process it spawned, so a tunnel's own children would
 * survive the script that started them; taskkill /T takes the tree. It is synchronous on
 * purpose — the caller exits immediately afterwards.
 */
export function stopChild(child: ChildProcess): void {
  if (!IS_WINDOWS) {
    try { child.kill('SIGTERM') } catch {}
    return
  }
  if (child.pid === undefined) return
  try {
    execFileSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' })
  } catch {
    try { child.kill() } catch {}
  }
}
