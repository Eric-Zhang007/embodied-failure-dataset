param(
    [int]$PollSeconds = 20,
    [switch]$Once,
    [switch]$NoStart
)

$RepoPath = '/home/zjc/embodied-failure-dataset'
$OutputDir = 'output_collector_7lane_20260726'
$WatchdogLog = Join-Path $PSScriptRoot "..\\$OutputDir\\watchdog.log"

function Test-CollectorProcessOutput {
    param([string[]]$Lines)

    return [bool]($Lines | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -First 1)
}

function Test-CollectorRunning {
    $pattern = '[.]venv/bin/python scripts/run_pipeline.py.*--task-lanes.*--output output_collector_7lane_20260726'
    $lines = & wsl.exe --distribution Ubuntu --exec bash -lc "pgrep -f '$pattern'" 2>$null
    return Test-CollectorProcessOutput $lines
}

function Write-WatchdogLog {
    param([string]$Message)

    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -LiteralPath $WatchdogLog -Value $line
}

function Start-Collector {
    $pipelineCommand = 'env EFD_API_TIMEOUT_S=20 EFD_API_MAX_ATTEMPTS=2 .venv/bin/python scripts/run_pipeline.py --config config.toml --max 0 --parallel 7 --task-lanes --worker-retries 2 --output output_collector_7lane_20260726 >> output_collector_7lane_20260726/collector.log 2>&1'
    $argumentLine = "--distribution Ubuntu --cd $RepoPath bash -lc `"$pipelineCommand`""
    Start-Process -FilePath 'wsl.exe' -ArgumentList $argumentLine -WindowStyle Hidden
    Write-WatchdogLog 'collector missing; started one seven-lane collector'
}

function Invoke-CollectorWatchdog {
    param([int]$IntervalSeconds, [switch]$RunOnce, [switch]$SkipStart)

    do {
        if (-not (Test-CollectorRunning)) {
            if ($SkipStart) {
                Write-WatchdogLog 'collector missing; launch suppressed by -NoStart'
            } else {
                Start-Collector
            }
        }

        if ($RunOnce) {
            return
        }
        Start-Sleep -Seconds $IntervalSeconds
    } while ($true)
}

if ($MyInvocation.InvocationName -ne '.') {
    $createdNew = $false
    $mutex = [System.Threading.Mutex]::new($false, 'Local\EFBenchCollectorWatchdog', [ref]$createdNew)
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        $mutex.Dispose()
        exit 0
    }

    try {
        Invoke-CollectorWatchdog -IntervalSeconds $PollSeconds -RunOnce:$Once -SkipStart:$NoStart
    } finally {
        if ($ownsMutex) {
            $mutex.ReleaseMutex()
        }
        $mutex.Dispose()
    }
}
