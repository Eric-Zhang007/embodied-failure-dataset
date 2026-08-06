$ErrorActionPreference = 'Stop'

$monitorPath = Join-Path $PSScriptRoot '..\scripts\collector_monitor.ps1'
. $monitorPath

$procRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("efbench-proc-" + [guid]::NewGuid().ToString('N'))

try {
    foreach ($processId in @('5000', '5001', '5002')) {
        New-Item -ItemType Directory -Path (Join-Path $procRoot $processId) -Force | Out-Null
    }
    Set-Content -LiteralPath (Join-Path $procRoot '5000\cmdline') -NoNewline -Value "bash`0-lc`0flock -n /tmp/collector.lock .venv/bin/python scripts/run_pipeline.py --task-lanes"
    Set-Content -LiteralPath (Join-Path $procRoot '5000\status') -Value "Name:`tbash`nPPid:`t1"
    Set-Content -LiteralPath (Join-Path $procRoot '5001\cmdline') -NoNewline -Value ".venv/bin/python`0scripts/run_pipeline.py`0--task-lanes`0--output`0fixture"
    Set-Content -LiteralPath (Join-Path $procRoot '5001\status') -Value "Name:`tpython`nPPid:`t1"
    Set-Content -LiteralPath (Join-Path $procRoot '5002\cmdline') -NoNewline -Value "/opt/thor-Linux64-release`0-screen-width`0300"
    Set-Content -LiteralPath (Join-Path $procRoot '5002\status') -Value "Name:`tthor-Linux64`nPPid:`t5001"
    Set-Content -LiteralPath (Join-Path $procRoot 'meminfo') -Value "MemTotal:       8388608 kB`nMemAvailable:   4194304 kB"

    $pipeline = Get-CollectorProcfsQuery -Kind pipeline -ProcRoot $procRoot
    if (-not $pipeline.Available -or @($pipeline.Lines).Count -ne 1 -or $pipeline.Lines[0] -notmatch '^5001\s+python') {
        throw 'Procfs pipeline query must identify the collector without launching wsl.exe.'
    }

    $unity = Get-CollectorProcfsQuery -Kind unity -PipelinePid 5001 -ProcRoot $procRoot
    $unityLines = @($unity.Lines)
    if (-not $unity.Available -or $unityLines.Count -ne 1 -or $unityLines[0] -ne '5002') {
        throw 'Procfs Unity query must return only direct collector workers.'
    }

    $memory = Get-CollectorProcfsQuery -Kind memory -ProcRoot $procRoot
    if (-not $memory.Available -or $memory.Lines[0] -notmatch '^8\.0Gi\s+4\.0Gi\s+4\.0Gi$') {
        throw 'Procfs memory query must preserve total, used, and available memory.'
    }
} finally {
    Remove-Item -LiteralPath $procRoot -Recurse -Force -ErrorAction SilentlyContinue
}
