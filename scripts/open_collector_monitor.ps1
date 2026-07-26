param(
    [string]$OutputDir = (Join-Path $PSScriptRoot '..\output_collector_7lane_20260726'),
    [int]$RefreshSeconds = 2
)

$monitorPath = Join-Path $PSScriptRoot 'collector_monitor.ps1'
if (-not (Test-Path -LiteralPath $monitorPath)) {
    throw "Collector monitor script is missing: $monitorPath"
}

Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $monitorPath,
    '-OutputDir', $OutputDir, '-RefreshSeconds', $RefreshSeconds
)
