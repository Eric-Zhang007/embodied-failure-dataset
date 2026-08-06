$scriptPath = Join-Path $PSScriptRoot '..\scripts\run_collector_watchdog.ps1'

if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Collector watchdog script is missing: $scriptPath"
}

. $scriptPath

if (Test-CollectorProcessOutput @()) {
    throw 'An empty process query must not be treated as a running collector.'
}

if (-not (Test-CollectorProcessOutput @('  12345  '))) {
    throw 'A numeric collector PID must be treated as running.'
}

if (Test-CollectorProcessOutput @('not a pid')) {
    throw 'Non-PID process output must not be treated as running.'
}

$cleanOutputDir = 'output_collector_7lane_clean_20260726'
if (-not (Test-CollectorOutputDirectory -OutputDir $cleanOutputDir)) {
    throw 'A simple relative collector output directory must be accepted.'
}
if (Test-CollectorOutputDirectory -OutputDir '../outside-workspace') {
    throw 'A collector output directory must not escape the repository.'
}

$pipelineCommand = Get-CollectorPipelineCommand -OutputDir $cleanOutputDir
if ($pipelineCommand -notmatch 'EFD_API_TIMEOUT_S=35') {
    throw 'The collector command must allow a 35-second API response before timing out.'
}
if ($pipelineCommand -notmatch "--output $cleanOutputDir") {
    throw 'The collector command must write episode data to the requested output directory.'
}
if ($pipelineCommand -notmatch ">> $cleanOutputDir/collector[.]log") {
    throw 'The collector log must live beside the requested episode data.'
}
if ($pipelineCommand -notmatch "flock -n /tmp/efbench-$cleanOutputDir[.]lock") {
    throw 'The collector command must hold an output-directory-specific process lock.'
}

$watchdogSource = Get-Content -LiteralPath $scriptPath -Raw
$startFunction = [regex]::Match(
    $watchdogSource,
    'function Start-Collector \{(?<body>[\s\S]*?)\r?\n\}\r?\n\r?\nfunction Invoke-CollectorWatchdog'
).Groups['body'].Value
$createIndex = $startFunction.IndexOf('New-Item -ItemType Directory')
$launchIndex = $startFunction.IndexOf("Start-Process -FilePath 'wsl.exe'")
if ($createIndex -lt 0 -or $launchIndex -lt 0 -or $createIndex -gt $launchIndex) {
    throw 'Start-Collector must create its output directory before the shell redirects collector.log.'
}

$matchingFrontend = [pscustomobject]@{
    Name = 'wsl.exe'
    CommandLine = "wsl.exe --cd /home/zjc/embodied-failure-dataset bash -lc '.venv/bin/python scripts/run_pipeline.py --task-lanes --output $cleanOutputDir'"
}
if (-not (Test-CollectorWindowsFrontend -OutputDir $cleanOutputDir -Processes @($matchingFrontend))) {
    throw 'The watchdog must recognize its own Windows WSL pipeline frontend.'
}
$otherOutputFrontend = [pscustomobject]@{
    Name = 'wsl.exe'
    CommandLine = "wsl.exe --cd /home/zjc/embodied-failure-dataset bash -lc '.venv/bin/python scripts/run_pipeline.py --task-lanes --output other-output'"
}
if (Test-CollectorWindowsFrontend -OutputDir $cleanOutputDir -Processes @($otherOutputFrontend)) {
    throw 'A frontend for another output directory must not be treated as this collector.'
}

$onceOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $scriptPath -Once -NoStart 2>&1 | Out-String
if ($onceOutput -match 'ReleaseMutex|unsynchronized block') {
    throw "A one-shot watchdog run must release its mutex cleanly. Output: $onceOutput"
}
