param(
    [string]$OutputDir = (Join-Path $PSScriptRoot '..\output_collector_7lane_20260726'),
    [int]$RefreshSeconds = 2,
    [switch]$Once
)

$script:TaskTypes = @(
    'pick_and_place_simple',
    'pick_and_place_with_movable_recep',
    'pick_clean_then_place_in_recep',
    'pick_heat_then_place_in_recep',
    'pick_cool_then_place_in_recep',
    'look_at_obj_in_light',
    'pick_two_obj_and_place'
)

function Get-CollectorProcfsQuery {
    param(
        [ValidateSet('pipeline', 'unity', 'memory')][string]$Kind,
        [int]$PipelinePid = 0,
        [string]$ProcRoot = '\\wsl.localhost\Ubuntu\proc'
    )

    if (-not (Test-Path -LiteralPath $ProcRoot)) {
        return [pscustomobject]@{ Available = $false; Lines = @() }
    }

    try {
        if ($Kind -eq 'memory') {
            $memInfo = Get-Content -LiteralPath (Join-Path $ProcRoot 'meminfo') -Raw -ErrorAction Stop
            $totalMatch = [regex]::Match($memInfo, '(?m)^MemTotal:\s*(\d+)\s+kB')
            $availableMatch = [regex]::Match($memInfo, '(?m)^MemAvailable:\s*(\d+)\s+kB')
            if (-not $totalMatch.Success -or -not $availableMatch.Success) {
                return [pscustomobject]@{ Available = $true; Lines = @() }
            }
            $totalGiB = [math]::Round(([double]$totalMatch.Groups[1].Value) / 1MB, 1)
            $availableGiB = [math]::Round(([double]$availableMatch.Groups[1].Value) / 1MB, 1)
            $usedGiB = [math]::Round($totalGiB - $availableGiB, 1)
            $culture = [Globalization.CultureInfo]::InvariantCulture
            return [pscustomobject]@{
                Available = $true
                Lines = @(
                    "$($totalGiB.ToString('F1', $culture))Gi " +
                    "$($usedGiB.ToString('F1', $culture))Gi " +
                    "$($availableGiB.ToString('F1', $culture))Gi"
                )
            }
        }

        if ($Kind -eq 'unity' -and $PipelinePid -le 0) {
            return [pscustomobject]@{ Available = $true; Lines = @() }
        }

        $lines = @()
        foreach ($entry in Get-ChildItem -LiteralPath $ProcRoot -Directory -ErrorAction Stop) {
            if ($entry.Name -notmatch '^\d+$') {
                continue
            }
            $cmdlinePath = Join-Path $entry.FullName 'cmdline'
            if (-not (Test-Path -LiteralPath $cmdlinePath)) {
                continue
            }
            try {
                $cmdline = ((Get-Content -LiteralPath $cmdlinePath -Raw -ErrorAction Stop) -replace [string][char]0, ' ').Trim()
            } catch {
                continue
            }

            if ($Kind -eq 'pipeline') {
                if ($cmdline -match '^[.]venv/bin/python\s+scripts/run_pipeline[.]py' -and $cmdline -match '--task-lanes') {
                    $lines += "$($entry.Name) python $cmdline"
                }
                continue
            }

            if ($cmdline -notmatch '(^|\s)([^\s]*/)?thor-Linux64(-|\s|$)') {
                continue
            }
            try {
                $status = Get-Content -LiteralPath (Join-Path $entry.FullName 'status') -Raw -ErrorAction Stop
            } catch {
                continue
            }
            $parentMatch = [regex]::Match($status, '(?m)^PPid:\s*(\d+)')
            if ($parentMatch.Success -and [int]$parentMatch.Groups[1].Value -eq $PipelinePid) {
                $lines += $entry.Name
            }
        }
        return [pscustomobject]@{ Available = $true; Lines = @($lines) }
    } catch {
        # Procfs is present but momentarily unreadable; keep the dashboard responsive.
        return [pscustomobject]@{ Available = $true; Lines = @() }
    }
}

function Invoke-CollectorWslQuery {
    param(
        [ValidateSet('pipeline', 'unity', 'memory')][string]$Kind,
        [int]$PipelinePid = 0
    )

    $procfs = Get-CollectorProcfsQuery -Kind $Kind -PipelinePid $PipelinePid
    if ($procfs.Available) {
        return @($procfs.Lines)
    }

    switch ($Kind) {
        'pipeline' {
            $command = 'ps -eo pid=,comm=,args= | awk ''/[.]venv\/bin\/python scripts\/run_pipeline\.py/ && /--task-lanes/ && /--output/ {print}'''
        }
        'unity' {
            if ($PipelinePid -le 0) {
                return @()
            }
            $command = "ps -eo pid=,ppid=,comm= | awk -v parent=$PipelinePid '`$2 == parent && /thor-Linux64/ {print `$1}'"
        }
        'memory' {
            $command = 'free -h | awk ''/^Mem:/ {print $2, $3, $7}'''
        }
    }

    try {
        return @(& wsl.exe --distribution Ubuntu --exec bash -lc $command 2>$null)
    } catch {
        return @()
    }
}

function Select-CollectorPipelinePid {
    param([string[]]$ProcessLines)

    foreach ($line in $ProcessLines) {
        if ($line -match '^\s*(?<pid>\d+)\s+python(?:3(?:\.\d+)?)?\s+.*[.]venv/bin/python scripts/run_pipeline\.py') {
            return [string]$Matches.pid
        }
    }
    return $null
}

function Get-VisibleAi2ThorWindowCount {
    return @(
        Get-Process -ErrorAction SilentlyContinue | Where-Object {
            $_.MainWindowTitle -match 'AI2-THOR|Unity|ai2thor'
        }
    ).Count
}

function Get-CollectorPipelinePids {
    param([scriptblock]$WslQuery = ${function:Invoke-CollectorWslQuery})

    $collectorPid = Select-CollectorPipelinePid -ProcessLines @(& $WslQuery 'pipeline' 0)
    if ($collectorPid) {
        return @($collectorPid)
    }
    return @()
}

function Get-MonitorEpisodeRecords {
    param([string]$OutputDir)

    if (-not (Test-Path -LiteralPath $OutputDir)) {
        return @()
    }

    $records = @()
    foreach ($file in Get-ChildItem -LiteralPath $OutputDir -Filter 'trial_*.json' -File -ErrorAction SilentlyContinue) {
        try {
            $episode = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            $branchSteps = @{}
            foreach ($step in @($episode.steps)) {
                $branchId = [string]$step.branch_id
                if (-not $branchId) {
                    continue
                }
                if (-not $branchSteps.ContainsKey($branchId)) {
                    $branchSteps[$branchId] = 0
                }
                $branchSteps[$branchId]++
            }
            $records += [pscustomobject]@{
                EpisodeId = [string]$episode.episode_id
                TaskType = [string]$episode.alfred_task_type
                Status = [string]$episode.status
                Pid = [int]$episode.pid
                Steps = @($episode.steps).Count
                BranchSteps = $branchSteps
                Modified = $file.LastWriteTime
            }
        } catch {
            # A worker can be replacing JSON while the monitor reads it.
        }
    }
    return $records
}

function Get-ActiveForks {
    param([string]$LogPath)

    if (-not (Test-Path -LiteralPath $LogPath)) {
        return @()
    }

    $active = @{}
    $lines = Get-Content -LiteralPath $LogPath -Tail 20000 -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
        if ($line -match '^\[fork\] Starting (?<branch>\S+) \(parent=(?<parent>[^)]+)\) on (?<episode>\S+)(?: lane=(?<lane>\S+))?') {
            $key = "$($Matches.episode)|$($Matches.branch)"
            $lane = [string]$Matches.lane
            if (-not $lane) {
                continue
            }
            if ($lane) {
                $active[$key] = [pscustomobject]@{
                    EpisodeId = [string]$Matches.episode
                    Branch = [string]$Matches.branch
                    TaskType = $lane
                }
            }
            continue
        }
        if ($line -match '^\[fork\] Finished (?<branch>\S+) \(parent=[^)]+\) on (?<episode>\S+)') {
            $active.Remove("$($Matches.episode)|$($Matches.branch)")
        }
    }
    return @($active.Values)
}

function Get-HostMemoryText {
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        $total = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        $free = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
        return "$free GiB free / $total GiB host"
    } catch {
        return 'host memory unavailable'
    }
}

function Get-CollectorSnapshot {
    param(
        [string]$OutputDir,
        [int]$PipelinePid = 0,
        [scriptblock]$WslQuery = ${function:Invoke-CollectorWslQuery},
        [scriptblock]$VisibleWindowQuery = ${function:Get-VisibleAi2ThorWindowCount}
    )

    $pipelinePids = @(Get-CollectorPipelinePids -WslQuery $WslQuery)
    if ($PipelinePid -le 0 -and $pipelinePids.Count -gt 0) {
        $PipelinePid = [int]$pipelinePids[0]
    }
    $records = Get-MonitorEpisodeRecords -OutputDir $OutputDir
    $lanes = [ordered]@{}
    foreach ($taskType in $script:TaskTypes) {
        $lanes[$taskType] = [pscustomobject]@{
            State = 'idle'
            EpisodeId = ''
            Branch = ''
            Steps = 0
        }
    }

    $branchStepsByEpisode = @{}
    foreach ($record in $records) {
        if ($record.EpisodeId -and $record.TaskType) {
            $branchStepsByEpisode[$record.EpisodeId] = $record.BranchSteps
        }
    }
    foreach ($record in $records | Where-Object {
        $_.Status -eq 'running' -and $_.Pid -eq $PipelinePid -and $lanes.Contains($_.TaskType)
    }) {
        $lanes[$record.TaskType] = [pscustomobject]@{
            State = 'main'
            EpisodeId = $record.EpisodeId
            Branch = 'main'
            Steps = $record.Steps
        }
    }

    $logPath = Join-Path $OutputDir 'collector.log'
    $forks = Get-ActiveForks -LogPath $logPath
    foreach ($fork in $forks) {
        if ($lanes.Contains($fork.TaskType) -and $lanes[$fork.TaskType].State -eq 'idle') {
            $forkSteps = 0
            if ($branchStepsByEpisode.ContainsKey($fork.EpisodeId)) {
                $episodeBranchSteps = $branchStepsByEpisode[$fork.EpisodeId]
                if ($episodeBranchSteps.ContainsKey($fork.Branch)) {
                    $forkSteps = [int]$episodeBranchSteps[$fork.Branch]
                }
            }
            $lanes[$fork.TaskType] = [pscustomobject]@{
                State = 'fork'
                EpisodeId = $fork.EpisodeId
                Branch = $fork.Branch
                Steps = $forkSteps
            }
        }
    }

    $unityPids = @(& $WslQuery 'unity' $PipelinePid | ForEach-Object {
        $value = ([string]$_).Trim()
        if ($value -match '^\d+$') { $value }
    })
    $namedForkCount = @($lanes.Values | Where-Object { $_.State -eq 'fork' }).Count
    $mainLaneCount = @($lanes.Values | Where-Object { $_.State -eq 'main' }).Count
    $inferredForkCount = [math]::Max(0, $unityPids.Count - $mainLaneCount - $namedForkCount)
    $remainingInferredForks = $inferredForkCount
    foreach ($taskType in $script:TaskTypes) {
        if ($remainingInferredForks -le 0) {
            break
        }
        if ($lanes[$taskType].State -eq 'idle') {
            $lanes[$taskType] = [pscustomobject]@{
                State = 'fork?'
                EpisodeId = '-'
                Branch = 'inferred from Unity'
                Steps = 0
            }
            $remainingInferredForks--
        }
    }
    $memoryLine = (@(& $WslQuery 'memory' $PipelinePid) | Select-Object -First 1) -join ' '
    $recentEvents = @()
    if (Test-Path -LiteralPath $logPath) {
        $recentEvents = @(
            Get-Content -LiteralPath $logPath -Tail 1000 -ErrorAction SilentlyContinue |
                Where-Object { $_ -match '\[fork\]|RETRY |CRASH |Worker for |TIMEOUT_EXHAUSTED|API_OUTAGE' } |
                Select-Object -Last 6
        )
    }

    return [pscustomobject]@{
        Timestamp = Get-Date
        PipelinePid = $PipelinePid
        CollectorPids = $pipelinePids
        UnityCount = $unityPids.Count
        VisibleWindowCount = [int](& $VisibleWindowQuery)
        HostMemory = Get-HostMemoryText
        WslMemory = if ($memoryLine) { "WSL used/available: $memoryLine" } else { 'WSL memory unavailable' }
        Lanes = $lanes
        ForkCount = $namedForkCount + $inferredForkCount
        RecentEvents = $recentEvents
    }
}

function Format-CollectorDashboardLines {
    param([pscustomobject]$Snapshot)

    $collector = if ($Snapshot.PipelinePid -gt 0) { "RUNNING (PID $($Snapshot.PipelinePid))" } else { 'NOT RUNNING' }
    $lines = @(
        'EF-Bench Collector Monitor',
        "Updated: $($Snapshot.Timestamp.ToString('yyyy-MM-dd HH:mm:ss'))    Collector: $collector",
        "Unity workers: $($Snapshot.UnityCount)    Visible AI2-THOR windows: $($Snapshot.VisibleWindowCount)    Active forks: $($Snapshot.ForkCount)",
        "Memory: $($Snapshot.HostMemory)    $($Snapshot.WslMemory)",
        '',
        ('{0,-36} {1,-6} {2,-29} {3,-25} {4,5}' -f 'TASK TYPE', 'STATE', 'EPISODE', 'BRANCH', 'STEPS'),
        ('-' * 108)
    )
    foreach ($taskType in $script:TaskTypes) {
        $lane = $Snapshot.Lanes[$taskType]
        $state = $lane.State.ToUpperInvariant()
        $episode = if ($lane.EpisodeId) { $lane.EpisodeId } else { '-' }
        $branch = if ($lane.Branch) { $lane.Branch } else { '-' }
        $lines += ('{0,-36} {1,-6} {2,-29} {3,-25} {4,5}' -f $taskType, $state, $episode, $branch, $lane.Steps)
    }
    $lines += ''
    $lines += 'Recent events:'
    if ($Snapshot.RecentEvents.Count -eq 0) {
        $lines += '  (no retry, crash, timeout-exhausted, or fork event in current log tail)'
    } else {
        foreach ($event in $Snapshot.RecentEvents) {
            $lines += "  $event"
        }
    }
    $lines += ''
    $lines += '[P] Pause collector   [R] Restart collector   [L] Tail log   [Q] Close monitor'
    return $lines
}

function Show-CollectorDashboard {
    param([pscustomobject]$Snapshot)

    Clear-Host
    Format-CollectorDashboardLines -Snapshot $Snapshot | ForEach-Object { Write-Host $_ }
}

function Test-CollectorPidTarget {
    param(
        [int]$TargetPid,
        [string[]]$CollectorPids
    )

    return $CollectorPids -contains [string]$TargetPid
}

function Get-CollectorWatchdogPids {
    return @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $_.CommandLine -match 'run_collector_watchdog[.]ps1'
        } | ForEach-Object { [string]$_.ProcessId }
    )
}

function Confirm-CollectorAction {
    param([string]$Prompt)

    return (Read-Host "$Prompt Type YES to continue") -ceq 'YES'
}

function Invoke-CollectorControl {
    param(
        [ValidateSet('Pause', 'Restart', 'Log')][string]$Action,
        [string]$OutputDir,
        [scriptblock]$WslQuery = ${function:Invoke-CollectorWslQuery}
    )

    $watchdogPath = Join-Path $PSScriptRoot 'run_collector_watchdog.ps1'
    $logPath = Join-Path $OutputDir 'collector.log'
    if ($Action -eq 'Log') {
        $quotedPath = $logPath.Replace("'", "''")
        Start-Process -FilePath 'powershell.exe' -ArgumentList @(
            '-NoExit', '-NoProfile', '-Command',
            "Get-Content -LiteralPath '$quotedPath' -Tail 120 -Wait"
        )
        return
    }

    $collectorPids = Get-CollectorPipelinePids -WslQuery $WslQuery
    if ($collectorPids.Count -eq 0) {
        Write-Host 'Collector is already stopped.' -ForegroundColor Yellow
    } elseif (-not (Confirm-CollectorAction -Prompt "$Action collector PID(s): $($collectorPids -join ',').")) {
        Write-Host 'Control action cancelled.' -ForegroundColor Yellow
        return
    } else {
        foreach ($pidText in $collectorPids) {
            $pid = [int]$pidText
            if (Test-CollectorPidTarget -TargetPid $pid -CollectorPids $collectorPids) {
                & wsl.exe --distribution Ubuntu --exec bash -lc "kill -TERM $pid" 2>$null
            }
        }
    }

    if ($Action -eq 'Pause') {
        foreach ($watchdogPid in Get-CollectorWatchdogPids) {
            Stop-Process -Id ([int]$watchdogPid) -ErrorAction SilentlyContinue
        }
        Write-Host 'Collector and watchdog paused. Use R to restart.' -ForegroundColor Yellow
        return
    }

    Start-Sleep -Seconds 1
    Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $watchdogPath, '-Once',
        '-OutputDir', (Split-Path -Leaf $OutputDir)
    )
    Write-Host 'Collector restart requested through the watchdog.' -ForegroundColor Green
}

function Start-CollectorMonitor {
    param([string]$OutputDir, [int]$RefreshSeconds, [switch]$Once)

    do {
        $snapshot = Get-CollectorSnapshot -OutputDir $OutputDir
        Show-CollectorDashboard -Snapshot $snapshot
        if ($Once) {
            return
        }

        $until = (Get-Date).AddSeconds([math]::Max(1, $RefreshSeconds))
        while ((Get-Date) -lt $until) {
            try {
                if ([Console]::KeyAvailable) {
                    $key = [Console]::ReadKey($true).Key
                    switch ($key) {
                        'P' { Invoke-CollectorControl -Action Pause -OutputDir $OutputDir }
                        'R' { Invoke-CollectorControl -Action Restart -OutputDir $OutputDir }
                        'L' { Invoke-CollectorControl -Action Log -OutputDir $OutputDir }
                        'Q' { return }
                    }
                }
            } catch {
                # Non-console hosts do not expose keyboard polling.
            }
            Start-Sleep -Milliseconds 100
        }
    } while ($true)
}

if ($MyInvocation.InvocationName -ne '.') {
    Start-CollectorMonitor -OutputDir $OutputDir -RefreshSeconds $RefreshSeconds -Once:$Once
}
