$ErrorActionPreference = 'Stop'

$monitorPath = Join-Path $PSScriptRoot '..\scripts\collector_monitor.ps1'

if (-not (Test-Path -LiteralPath $monitorPath)) {
    throw "Collector monitor script is missing: $monitorPath"
}

. $monitorPath

$collectorPid = Select-CollectorPipelinePid -ProcessLines @(
    '1199 bash bash -lc .venv/bin/python scripts/run_pipeline.py --task-lanes',
    '1205 python .venv/bin/python scripts/run_pipeline.py --task-lanes'
)
if ($collectorPid -ne '1205') {
    throw "The Python collector PID must be selected instead of its bash parent; got $collectorPid."
}

$fixtureDir = Join-Path ([System.IO.Path]::GetTempPath()) ("efbench-monitor-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixtureDir | Out-Null

try {
    $mainEpisode = [ordered]@{
        episode_id = 'trial_main'
        alfred_task_type = 'pick_and_place_simple'
        status = 'running'
        pid = 1205
        steps = @(
            @{ step_id = 'main__s0'; branch_id = 'main' },
            @{ step_id = 'main__s1'; branch_id = 'main' },
            @{ step_id = 'fork_old_main__s0'; branch_id = 'fork_old_main' }
        )
    }
    $forkEpisode = [ordered]@{
        episode_id = 'trial_fork'
        alfred_task_type = 'look_at_obj_in_light'
        status = 'completed'
        pid = 0
        steps = @(
            @{ step_id = 'main__s0'; branch_id = 'main' },
            @{ step_id = 'fork_s4_main__s0'; branch_id = 'fork_s4_main' },
            @{ step_id = 'fork_s4_main__s1'; branch_id = 'fork_s4_main' }
        )
    }
    $mainEpisode | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $fixtureDir 'trial_main.json')
    $forkEpisode | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $fixtureDir 'trial_fork.json')
    @(
        '[fork] Starting fork_stale_run (parent=main) on trial_old lane=pick_cool_then_place_in_recep',
        'Running 7 task lanes: new collector run',
        '[fork] Starting fork_s4_main (parent=main) on trial_fork lane=look_at_obj_in_light',
        '[fork] Starting fork_stale_main (parent=main) on trial_main',
        '[fork] Finished fork_old_main (parent=main) on trial_old lane=pick_cool_then_place_in_recep'
    ) | Set-Content -LiteralPath (Join-Path $fixtureDir 'collector.log')

    $wslQuery = {
        param([string]$Kind, [int]$PipelinePid)
        switch ($Kind) {
            'pipeline' { return @('1205 python .venv/bin/python scripts/run_pipeline.py --task-lanes') }
            'unity' { return @('7001', '7002', '7003') }
            'memory' { return @('Mem: 8Gi 6Gi 2Gi') }
            default { throw "Unexpected WSL query kind: $Kind" }
        }
    }

    $snapshot = Get-CollectorSnapshot -OutputDir $fixtureDir `
        -WslQuery $wslQuery -VisibleWindowQuery { return 0 }

    if ($snapshot.UnityCount -ne 3) {
        throw "Expected three Unity processes, got $($snapshot.UnityCount)."
    }
    if ($snapshot.VisibleWindowCount -ne 0) {
        throw "Expected zero visible AI2-THOR windows, got $($snapshot.VisibleWindowCount)."
    }
    if ($snapshot.Lanes.Count -ne 7) {
        throw "Expected seven stable task lanes, got $($snapshot.Lanes.Count)."
    }
    if ($snapshot.Lanes['pick_and_place_simple'].State -ne 'main') {
        throw 'A running main episode must populate its ALFRED task lane.'
    }
    if ($snapshot.Lanes['pick_and_place_simple'].Steps -ne 2) {
        throw 'The main lane must report its current step count.'
    }
    if ($snapshot.Lanes['look_at_obj_in_light'].State -ne 'fork') {
        throw 'An unfinished fork must populate an otherwise empty task lane.'
    }
    if ($snapshot.Lanes['look_at_obj_in_light'].Steps -ne 2) {
        throw 'A fork lane must report its branch-local persisted step count.'
    }
    if ($snapshot.Lanes['pick_cool_then_place_in_recep'].State -ne 'idle') {
        throw 'A finished fork must not be displayed as an active lane.'
    }
    if ($snapshot.ForkCount -ne 1) {
        throw 'A legacy fork-start line without a lane marker must not be counted as active.'
    }
    if ($snapshot.UnattributedUnityCount -ne 1) {
        throw 'A transient extra Unity process must be reported without inventing a fork lane.'
    }
    if (@($snapshot.Lanes.Values | Where-Object { $_.State -eq 'fork?' }).Count -ne 0) {
        throw 'The monitor must never assign an inferred Unity process to an arbitrary task lane.'
    }

    $lines = Format-CollectorDashboardLines -Snapshot $snapshot
    if (($lines -join "`n") -notmatch 'Unity workers: 3') {
        throw 'Dashboard output must report the Unity worker count.'
    }
    if (($lines -join "`n") -notmatch 'look_at_obj_in_light.*FORK') {
        throw 'Dashboard output must visibly label fork work.'
    }

    if (-not (Test-CollectorPidTarget -TargetPid 1205 -CollectorPids @('1199', '1205'))) {
        throw 'A verified live collector PID must be accepted as a control target.'
    }
    if (Test-CollectorPidTarget -TargetPid 9999 -CollectorPids @('1199', '1205')) {
        throw 'An unrelated PID must never be accepted as a control target.'
    }

    $monitorText = Get-Content -LiteralPath $monitorPath -Raw
    if ($monitorText -notmatch 'kill -TERM \$pid') {
        throw 'Pause and restart must terminate a collector that is waiting for API recovery.'
    }
    if ($monitorText -match 'output_collector_7lane_20260726/ \{print\}') {
        throw 'The monitor must not be pinned to the legacy collector output directory.'
    }

    $launcherPath = Join-Path $PSScriptRoot '..\scripts\open_collector_monitor.ps1'
    if (-not (Test-Path -LiteralPath $launcherPath)) {
        throw "Collector monitor launcher is missing: $launcherPath"
    }
    $launcherText = Get-Content -LiteralPath $launcherPath -Raw
    if ($launcherText -notmatch 'Start-Process') {
        throw 'The launcher must open a separate foreground PowerShell window.'
    }
} finally {
    Remove-Item -LiteralPath $fixtureDir -Recurse -Force -ErrorAction SilentlyContinue
}
