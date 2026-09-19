# One command for someone joining a friend's network, from Windows.
#
#   .\scripts\join.ps1 https://their-address ABCD-1234
#
# The PowerShell counterpart of join.sh. Same checks, same order, same outcome — kept as
# a separate file rather than a translated one-liner so neither platform's path can break
# the other.
#
# If PowerShell refuses to run this, it is the execution policy, not the script:
#   powershell -ExecutionPolicy Bypass -File .\scripts\join.ps1 <server-url> <code>

param(
  [Parameter(Position = 0)] [string] $Server,
  [Parameter(Position = 1)] [string] $Code
)

$ErrorActionPreference = 'Stop'

if (-not $Server -or -not $Code) {
  Write-Host "Usage: .\scripts\join.ps1 <server-url> <pairing-code>"
  Write-Host "Both come from the invite link you were sent:"
  Write-Host "    https://SERVER/join?code=CODE"
  exit 1
}

Write-Host ""
Write-Host "  Joining $Server"
Write-Host ""

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  Write-Host "  Node.js is not installed."
  Write-Host "  Install Node 24 or newer from https://nodejs.org and run this again."
  Write-Host "  Or, if you have winget:  winget install OpenJS.NodeJS"
  exit 1
}

$nodeMajor = [int](node -p 'process.versions.node.split(".")[0]')
if ($nodeMajor -lt 24) {
  Write-Host "  Node.js $(node -v) is too old - version 24 or newer is required."
  Write-Host "  Update from https://nodejs.org and run this again."
  exit 1
}
Write-Host "  ok  Node.js $(node -v)"

if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
  Write-Host "  installing pnpm..."
  npm install -g pnpm | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "  Could not install pnpm automatically. Run:  npm install -g pnpm"
    exit 1
  }
}
Write-Host "  ok  pnpm $(pnpm --version)"

Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Test-Path 'node_modules')) {
  # --prod keeps this small. The heavy runtimes are opt-in via `pnpm agent enable`.
  Write-Host "  installing dependencies (one time, about 40 MB)..."
  pnpm install --prod --silent
  if ($LASTEXITCODE -ne 0) { Write-Host "  pnpm install failed."; exit 1 }
}
Write-Host "  ok  dependencies"
Write-Host ""

# Some networks resolve established domains but not freshly created ones; the agent
# falls back to public resolvers when that happens.
if (-not $env:DWP_DNS_FALLBACK) { $env:DWP_DNS_FALLBACK = '1' }

node packages/agent/src/index.ts pair --server $Server --code $Code
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "  Starting. Leave this window open - press Ctrl-C to stop at any time."
Write-Host "  To stop taking work without closing:  pnpm agent pause"
Write-Host "  To also run machine-learning work:     pnpm agent enable ml"
Write-Host ""
node packages/agent/src/index.ts run
exit $LASTEXITCODE
