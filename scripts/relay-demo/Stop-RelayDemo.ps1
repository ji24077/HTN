$ErrorActionPreference = 'Stop'
$statePath = Join-Path $PSScriptRoot '.relay-demo/state.json'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'No recorded Relay demo processes.'; exit }
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
foreach ($entry in $state.processes) {
    $process = Get-Process -Id $entry.id -ErrorAction SilentlyContinue
    if ($process -and $process.StartTime.ToUniversalTime().ToString('o') -eq $entry.started_at -and $process.Path -eq $entry.executable) {
        Stop-Process -Id $process.Id
    }
}
if ($state.fleet_database) {
    & (Join-Path $PSScriptRoot 'HTN/backend/.venv/Scripts/python.exe') (Join-Path $PSScriptRoot 'Stop-RelayFleetDatabase.py')
    if ($LASTEXITCODE -ne 0) { throw 'Demo database did not stop. Process state has been preserved.' }
}
Remove-Item -LiteralPath $statePath
Write-Output 'Stopped the Relay demo processes recorded by this launcher.'
