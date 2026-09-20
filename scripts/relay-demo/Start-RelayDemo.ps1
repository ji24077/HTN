param([switch]$Live, [int]$FrontendPort = 5175, [int]$BackendPort = 8090, [int]$FleetPort = 8091)
$ErrorActionPreference = 'Stop'
# Some Windows launchers inherit both Path and PATH. PowerShell's child-process
# environment builder rejects those duplicate case-insensitive keys.
$relayPathValue = [Environment]::GetEnvironmentVariable('Path', 'Process')
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $relayPathValue, 'Process')
$workspace = $PSScriptRoot
$gpuRoot = Join-Path $workspace 'ji-review'
$frontRoot = Join-Path $workspace 'relay-demo/frontend'
$python = Join-Path $gpuRoot '.venv/Scripts/python.exe'
$fleetRoot = Join-Path $workspace 'HTN'
$fleetPython = Join-Path $fleetRoot 'backend/.venv/Scripts/python.exe'
$fleetWork = Join-Path $workspace '.relay-fleet-local'
$vite = Join-Path $frontRoot 'node_modules/vite/bin/vite.js'
$node = (Get-Command node.exe).Source
$logs = Join-Path $workspace '.relay-demo'
foreach ($required in @($python, $fleetPython, $vite, (Join-Path $fleetRoot 'backend/src/orchestrator/server/web/index.html'))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing dependency: $required" }
}
foreach ($port in @($FrontendPort, $BackendPort, $FleetPort)) {
    $probe = [Net.Sockets.TcpClient]::new()
    try {
        if ($probe.ConnectAsync('127.0.0.1', $port).Wait(300) -and $probe.Connected) {
            throw "Port $port is already in use. Keep that service running or choose another port."
        }
    } catch [AggregateException] { } finally { $probe.Dispose() }
}
New-Item -ItemType Directory -Path $logs -Force | Out-Null
New-Item -ItemType Directory -Path $fleetWork -Force | Out-Null
$settings = @{
    PYTHONPATH = (Join-Path $gpuRoot 'src')
    GPUSHARE_PORT = "$BackendPort"
    GPUSHARE_READ_ONLY_DEMO = $(if ($Live) { '0' } else { '1' })
    GPUSHARE_BACKEND_URL = "http://127.0.0.1:$BackendPort"
    VITE_GPUSHARE_URL = '/'
    ORCHESTRATOR_BACKEND_URL = "http://127.0.0.1:$FleetPort"
    PUBLIC_ORIGIN = "http://127.0.0.1:$FrontendPort"
    FRONTEND_TLS_CERT_FILE = ''
    FRONTEND_TLS_KEY_FILE = ''
}
$previous = @{}
$started = @()
try {
    # Use the existing loopback-only demo session flow and a separate local DB.
    # No .env file is loaded, and external agents/integrations are disabled.
    $fleetSettings = @{
        PYTHONPATH = (Join-Path $fleetRoot 'backend/src')
        PUBLIC_ORIGIN = ''
        OPENAI_API_KEY = ''
        SUPERVISOR_ENABLED = 'false'
        DATABASE_SCHEMA = 'public'
        SUPABASE_URL = ''
        SUPABASE_PUBLISHABLE_KEY = ''
        SUPABASE_ADMIN_IDS = ''
        SUPABASE_ADMIN_EMAILS = ''
        SENTRY_DSN = ''
        SENTRY_FRONTEND_DSN = ''
        TAILSCALE_OAUTH_CLIENT_ID = ''
        TAILSCALE_OAUTH_CLIENT_SECRET = ''
        WORKER_GATEWAY_URL = ''
        DWP_SELF_SERVE_JOIN = 'false'
    }
    $fleetPrevious = @{}
    try {
        foreach ($key in $fleetSettings.Keys) {
            $fleetPrevious[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
            [Environment]::SetEnvironmentVariable($key, $fleetSettings[$key], 'Process')
        }
        $fleet = Start-Process -FilePath $fleetPython -ArgumentList @('-m', 'orchestrator.demo', 'server', '--port', "$FleetPort") -WorkingDirectory $fleetWork -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'fleet.log') -RedirectStandardError (Join-Path $logs 'fleet.err.log')
        $started += $fleet
    } finally {
        foreach ($key in $fleetPrevious.Keys) { [Environment]::SetEnvironmentVariable($key, $fleetPrevious[$key], 'Process') }
    }
    $fleetReady = $false
    for ($attempt = 0; $attempt -lt 80; $attempt++) {
        if ($fleet.HasExited) { throw "Fleet backend exited; inspect $logs/fleet.err.log" }
        try {
            $session = Invoke-RestMethod "http://127.0.0.1:$FleetPort/auth/session" -TimeoutSec 1
            if ($session.mode -eq 'demo') { $fleetReady = $true; break }
        } catch { Start-Sleep -Milliseconds 300 }
    }
    if (-not $fleetReady) { throw "Fleet session did not become ready; inspect $logs/fleet.err.log" }
    foreach ($key in $settings.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    $backend = Start-Process -FilePath $python -ArgumentList @('-m', 'gpushare.dashboard.app') -WorkingDirectory $gpuRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'backend.log') -RedirectStandardError (Join-Path $logs 'backend.err.log')
    $started += $backend
    $frontend = Start-Process -FilePath $node -ArgumentList @(('"' + $vite + '"'), '--host', '127.0.0.1', '--port', "$FrontendPort", '--strictPort') -WorkingDirectory $frontRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'frontend.log') -RedirectStandardError (Join-Path $logs 'frontend.err.log')
    $started += $frontend
    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if ($backend.HasExited -or $frontend.HasExited) { throw "Demo process exited; inspect logs in $logs" }
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:$FrontendPort/api/health" -TimeoutSec 1
            $session = Invoke-RestMethod "http://127.0.0.1:$FrontendPort/auth/session" -SessionVariable relaySession -TimeoutSec 1
            $snapshot = Invoke-RestMethod "http://127.0.0.1:$FrontendPort/v1/snapshot" -WebSession $relaySession -TimeoutSec 2
            if ($health.ok -and $session.mode -eq 'demo' -and $null -ne $snapshot.workers -and $null -ne $snapshot.tasks) { $ready = $true; break }
        } catch { Start-Sleep -Milliseconds 300 }
    }
    if (-not $ready) { throw "Demo did not become ready; inspect logs in $logs" }
    $state = @{
        url = "http://127.0.0.1:$FrontendPort/"
        evidence_url = "http://127.0.0.1:$FrontendPort/gpu-lab"
        fleet_database = $true
        recorded_only = -not $Live
        processes = @($started | ForEach-Object { @{ id = $_.Id; started_at = $_.StartTime.ToUniversalTime().ToString('o'); executable = $_.Path } })
    }
    $state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $logs 'state.json') -Encoding UTF8
    Write-Output "Relay main app: $($state.url)"
    Write-Output "GPU Lab chat: $($state.evidence_url)"
    Write-Output $(if ($Live) { 'Live controls enabled for existing configured GPUs.' } else { 'Recorded hardware evidence. No GPU launches or workload changes.' })
    Write-Output "Logs: $logs"
} catch {
    foreach ($process in $started) {
        if (-not $process.HasExited) { $process.Kill() }
    }
    $stopDatabase = Join-Path $workspace 'Stop-RelayFleetDatabase.py'
    if (Test-Path -LiteralPath $stopDatabase) { & $fleetPython $stopDatabase }
    throw
} finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}
