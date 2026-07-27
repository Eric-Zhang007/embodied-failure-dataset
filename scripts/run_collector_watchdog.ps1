param(
    [int]$PollSeconds = 20,
    [switch]$Once,
    [switch]$NoStart,
    [string]$OutputDir = 'output_collector_7lane_20260726'
)

function Test-CollectorProcessOutput {
    param([string[]]$Lines)

    return [bool]($Lines | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -First 1)
}

function Test-CollectorOutputDirectory {
    param([string]$OutputDir)

    return $OutputDir -match '^[A-Za-z0-9][A-Za-z0-9._-]*$'
}

function Get-CollectorPipelineCommand {
    param([string]$OutputDir)

    if (-not (Test-CollectorOutputDirectory -OutputDir $OutputDir)) {
        throw "Collector output directory must be a simple repository-relative name: $OutputDir"
    }
    return "flock -n /tmp/efbench-$OutputDir.lock env EFD_API_TIMEOUT_S=35 EFD_API_MAX_ATTEMPTS=2 EFD_FAIL_FAST_AUTH_ERRORS=1 .venv/bin/python scripts/run_pipeline.py --config config.toml --max 0 --parallel 7 --task-lanes --worker-retries 2 --output $OutputDir >> $OutputDir/collector.log 2>&1"
}

if (-not (Test-CollectorOutputDirectory -OutputDir $OutputDir)) {
    throw "Collector output directory must be a simple repository-relative name: $OutputDir"
}

$RepoPath = '/home/zjc/embodied-failure-dataset'
$WatchdogLog = Join-Path $PSScriptRoot "..\\$OutputDir\\watchdog.log"

function Test-CollectorWindowsFrontend {
    param(
        [string]$OutputDir,
        [object[]]$Processes = $null
    )

    if ($null -eq $Processes) {
        $Processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
    }
    $outputPattern = "(?:^|\s)--output\s+$([regex]::Escape($OutputDir))(?=\s|[^A-Za-z0-9._-]|$)"
    return [bool]($Processes | Where-Object {
        $_.Name -ieq 'wsl.exe' -and
        $_.CommandLine -match 'run_pipeline[.]py' -and
        $_.CommandLine -match '--task-lanes' -and
        $_.CommandLine -match $outputPattern
    } | Select-Object -First 1)
}

function Test-CollectorRunning {
    if (Test-CollectorWindowsFrontend -OutputDir $OutputDir) {
        return $true
    }

    $escapedOutputDir = [regex]::Escape($OutputDir)
    $pattern = "[.]venv/bin/python scripts/run_pipeline.py.*--task-lanes.*--output $escapedOutputDir"
    $lines = & wsl.exe --distribution Ubuntu --exec bash -lc "pgrep -f '$pattern'" 2>$null
    return Test-CollectorProcessOutput $lines
}

function Write-WatchdogLog {
    param([string]$Message)

    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WatchdogLog) | Out-Null
    Add-Content -LiteralPath $WatchdogLog -Value $line
}

function Start-Collector {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "..\$OutputDir") | Out-Null
    $pipelineCommand = Get-CollectorPipelineCommand -OutputDir $OutputDir
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
