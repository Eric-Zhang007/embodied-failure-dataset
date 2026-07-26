# Collector Monitor Design

## Goal

Provide a separate foreground PowerShell window for observing and controlling
the continuously running seven-lane collector. The monitor is operational only:
it does not modify episode JSON files or collector logic.

## Interface

`scripts/open_collector_monitor.ps1` opens a dedicated PowerShell window and
refreshes every two seconds. Its fixed-size dashboard shows:

- collector and watchdog process health;
- active Unity process count and visible AI2-THOR window count;
- host and WSL memory availability;
- one row for each ALFRED task type, identifying either an active main episode
  or a fork/replay that currently occupies that lane;
- current episode, branch, steps, recent retry/crash events, and recent API
  timeout counters.

The display distinguishes seven occupied lanes from main-episode JSON records.
Forks do not write a second `running` main-episode record, so current fork
starts from `collector.log` supplement the JSON view.

## Controls

The monitor handles single-key commands without stopping refresh:

- `P`: pause the collector with an interrupt signal after prompting for
  confirmation;
- `R`: restart the collector through the existing watchdog after prompting for
  confirmation;
- `L`: open `collector.log` in a separate PowerShell tail window;
- `Q`: close the monitor only.

Pause and restart target only the task-lane pipeline matching the fixed output
directory. The watchdog itself remains responsible for recovery after an
unexpected collector exit.

## Data Flow

The monitor reads `output_collector_7lane_20260726/*.json`,
`collector.log`, `watchdog.log`, Windows process window titles, and WSL process
tables. It never relies solely on `status=running`: the source of truth for
actual occupied slots is the number of child `thor-Linux64` processes owned by
the live pipeline. This prevents active fork/replay work from being displayed
as a false idle lane.

## Failure Handling

Transient unreadable JSON and missing log lines render as `updating` rather
than terminating the monitor. If the collector is absent, the screen states
that clearly and the restart command is available. A failed WSL query shows an
inline unavailable marker and the next refresh retries it.

## Verification

A PowerShell test will use fixture JSON, a fixture log, and mocked WSL/process
commands to verify lane aggregation, fork display, safe command targeting, and
the four keyboard command routes. A manual smoke test opens the monitor against
the live collector and checks that it reports seven Unity workers and zero
visible AI2-THOR windows.
