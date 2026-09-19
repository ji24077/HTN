import { existsSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { createHash } from 'node:crypto'

/**
 * The one-line installers, generated with the current binary hashes baked in.
 *
 * Piping a script from the internet into a shell deserves care, so the hashes are
 * embedded in the script itself rather than fetched separately: both arrive over the
 * same TLS connection from the same host, and the script refuses to run a download that
 * does not match. This is the model rustup and Homebrew use.
 */
const DIR = process.env.DWP_BINARIES_DIR ?? join('releases', 'binaries')
const APPS_DIR = process.env.DWP_APPS_DIR ?? join('releases', 'apps')

export type BinaryEntry = { target: string; os: string; arch: string; file: string; sha256: string; bytes: number }
/** A downloadable desktop app, as packaged by scripts/lib/apps.ts. */
export type AppEntry = { target: 'macos' | 'windows-x64'; file: string; sha256: string; bytes: number; opens: string }
export type BinaryIndex = {
  manifest: { version: string }; signature: string; publicKey: string
  binaries: BinaryEntry[]
  /** Absent on an index built before the apps existed, so every reader must allow for it. */
  apps?: AppEntry[]
}

export function binaryIndex(): BinaryIndex | null {
  const path = join(DIR, 'index.json')
  if (!existsSync(path)) return null
  try { return JSON.parse(readFileSync(path, 'utf8')) as BinaryIndex } catch { return null }
}

/** The desktop apps this server is offering, newest build only. */
export function appDownloads(): { version: string; apps: AppEntry[] } | null {
  const index = binaryIndex()
  if (!index?.apps?.length) return null
  return { version: index.manifest.version, apps: index.apps }
}

/**
 * Serve a file the signed index lists, and nothing else.
 *
 * Apps live in a different directory from the binaries they wrap, so the lookup has to
 * know which list matched. Everything else is unchanged: the name must appear in the
 * index exactly, which rules out traversal, and the bytes are re-hashed before they are
 * sent, so a corrupted or swapped file on this disk is refused here rather than being
 * distributed and rejected on every machine that downloads it.
 */
export function binaryFile(file: string): Buffer | null {
  const index = binaryIndex()
  const binary = index?.binaries.find(b => b.file === file)
  const app = index?.apps?.find(a => a.file === file)
  const entry = binary ?? app
  if (!entry) return null

  const path = join(binary ? DIR : APPS_DIR, file)
  if (!existsSync(path) || !statSync(path).isFile()) return null
  const bytes = readFileSync(path)
  if (createHash('sha256').update(bytes).digest('hex') !== entry.sha256) return null
  return bytes
}

/** POSIX installer: detect, download, verify, install, pair. */
export function installShell(origin: string): string {
  const index = binaryIndex()
  if (!index) return `#!/bin/sh\necho "This server is not offering binaries yet." >&2\nexit 1\n`

  const cases = index.binaries
    .filter(b => b.os !== 'win32')
    .map(b => `    ${b.os}-${b.arch}) FILE="${b.file}"; SUM="${b.sha256}" ;;`)
    .join('\n')

  return `#!/bin/sh
# Join a distributed work network.
#
#   curl -sSf ${origin}/install | sh -s -- YOUR-CODE
#
# Downloads a single executable, checks it against a hash delivered with this script,
# and pairs this computer. No Node, no package manager, nothing else to install.
set -eu

CODE="\${1:-}"
ORIGIN="${origin}"
VERSION="${index.manifest.version}"
BIN_DIR="\${DWP_BIN_DIR:-\$HOME/.dwp/bin}"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in
  darwin) OS=darwin ;;
  linux)  OS=linux ;;
  *) echo "  Unsupported system: $OS" >&2; exit 1 ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  arm64|aarch64) ARCH=arm64 ;;
  x86_64|amd64)  ARCH=x64 ;;
  *) echo "  Unsupported processor: $ARCH" >&2; exit 1 ;;
esac

case "$OS-$ARCH" in
${cases}
  *) echo "  No build for $OS-$ARCH." >&2; exit 1 ;;
esac

echo ""
echo "  Installing the agent for $OS-$ARCH ($VERSION)"
mkdir -p "$BIN_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "  Downloading..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$ORIGIN/download/$FILE" -o "$TMP/dwp-agent"
elif command -v wget >/dev/null 2>&1; then
  wget -q "$ORIGIN/download/$FILE" -O "$TMP/dwp-agent"
else
  echo "  Need curl or wget." >&2; exit 1
fi

# Verify before anything is made executable.
if command -v shasum >/dev/null 2>&1; then
  GOT="$(shasum -a 256 "$TMP/dwp-agent" | cut -d' ' -f1)"
elif command -v sha256sum >/dev/null 2>&1; then
  GOT="$(sha256sum "$TMP/dwp-agent" | cut -d' ' -f1)"
else
  echo "  Cannot verify the download: no shasum or sha256sum." >&2; exit 1
fi

if [ "$GOT" != "$SUM" ]; then
  echo "  The download does not match the expected hash. Refusing to install." >&2
  echo "    expected $SUM" >&2
  echo "    got      $GOT" >&2
  exit 1
fi
echo "  Verified."

chmod +x "$TMP/dwp-agent"
mv "$TMP/dwp-agent" "$BIN_DIR/dwp-agent"
echo "  Installed to $BIN_DIR/dwp-agent"

if [ -n "$CODE" ]; then
  echo ""
  "$BIN_DIR/dwp-agent" pair --server "$ORIGIN" --code "$CODE"
  echo ""
  echo "  Starting. Leave this window open; Ctrl-C stops it."
  echo ""
  exec "$BIN_DIR/dwp-agent" run
else
  echo ""
  echo "  Now pair it with the code you were given:"
  echo "      $BIN_DIR/dwp-agent pair --server $ORIGIN --code YOUR-CODE"
  echo "      $BIN_DIR/dwp-agent run"
  echo ""
fi
`
}

/** Windows installer, same steps in PowerShell. */
export function installPowerShell(origin: string): string {
  const index = binaryIndex()
  if (!index) return `Write-Host "This server is not offering binaries yet."; exit 1`
  const win = index.binaries.find(b => b.os === 'win32' && b.arch === 'x64')
  if (!win) return `Write-Host "No Windows build available."; exit 1`

  return `# Join a distributed work network.
#
#   irm ${origin}/install.ps1 | iex
#   # or, to pair in one step:
#   & ([scriptblock]::Create((irm ${origin}/install.ps1))) YOUR-CODE
#
# Downloads a single executable, checks it against a hash delivered with this script,
# and pairs this computer. Nothing else to install.
param([Parameter(Position = 0)] [string] $Code)

$ErrorActionPreference = 'Stop'

# Windows PowerShell 5.1 — still the default shell on Windows 10 and 11 — often defaults
# to a protocol set that excludes TLS 1.2, and then every download fails with "Could not
# create SSL/TLS secure channel", which says nothing useful about the cause. This cannot
# rescue the irm that fetched this script, but it fixes every request made inside it.
try {
  [Net.ServicePointManager]::SecurityProtocol =
    [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }
$Origin  = "${origin}"
$File    = "${win.file}"
$Sum     = "${win.sha256}"
$BinDir  = if ($env:DWP_BIN_DIR) { $env:DWP_BIN_DIR } else { Join-Path $env:USERPROFILE ".dwp\\bin" }

Write-Host ""
Write-Host "  Installing the agent for windows-x64 (${index.manifest.version})"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$Tmp = Join-Path $env:TEMP ("dwp-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

try {
  Write-Host "  Downloading..."
  Invoke-WebRequest -Uri "$Origin/download/$File" -OutFile "$Tmp\\dwp-agent.exe" -UseBasicParsing

  $Got = (Get-FileHash "$Tmp\\dwp-agent.exe" -Algorithm SHA256).Hash.ToLower()
  if ($Got -ne $Sum) {
    Write-Host "  The download does not match the expected hash. Refusing to install."
    Write-Host "    expected $Sum"
    Write-Host "    got      $Got"
    exit 1
  }
  Write-Host "  Verified."

  # Windows holds an exclusive handle on a running executable, so re-running this
  # installer while the agent is up fails on the move — and fails *after* the download
  # and hash check, which is the slow part. On POSIX the same operation quietly succeeds
  # through the inode, so this can only be caught on Windows.
  $Running = Get-Process dwp-agent -ErrorAction SilentlyContinue
  if ($Running) {
    Write-Host "  Stopping the running agent first..."
    $Running | Stop-Process -Force
    # Give Windows a moment to release the handle before moving over it.
    for ($i = 0; $i -lt 20; $i++) {
      if (-not (Get-Process dwp-agent -ErrorAction SilentlyContinue)) { break }
      Start-Sleep -Milliseconds 250
    }
  }

  $Dest = Join-Path $BinDir "dwp-agent.exe"
  try {
    Move-Item -Force "$Tmp\\dwp-agent.exe" $Dest
  } catch {
    # Still locked. Park the old one and put the new one in place; the stale file is
    # removed on the next run rather than leaving the install half-done.
    $Parked = "$Dest.old"
    Remove-Item -Force $Parked -ErrorAction SilentlyContinue
    Move-Item -Force $Dest $Parked
    Move-Item -Force "$Tmp\\dwp-agent.exe" $Dest
  }
  Remove-Item -Force "$Dest.old" -ErrorAction SilentlyContinue
  Write-Host "  Installed to $Dest"

  $Exe = Join-Path $BinDir "dwp-agent.exe"
  if ($Code) {
    Write-Host ""
    & $Exe pair --server $Origin --code $Code
    Write-Host ""
    Write-Host "  Starting. Leave this window open; Ctrl-C stops it."
    Write-Host ""
    & $Exe run
  } else {
    Write-Host ""
    Write-Host "  Now pair it with the code you were given:"
    Write-Host "      $Exe pair --server $Origin --code YOUR-CODE"
    Write-Host "      $Exe run"
    Write-Host ""
  }
} finally {
  Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
`
}
