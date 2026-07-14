# Codebase Structure

**Analysis Date:** 2026-06-12

## Directory Layout

```
embodied-failure-dataset/
│
├── pyproject.toml                  # Project config (uv-based Python 3.10, single dep: ai2thor)
├── uv.lock                         # Lockfile (183K)
├── .python-version                 # "3.10"
├── .gitignore
├── CLAUDE.md                       # Project guidance for Claude
│
├── src/                            # All Python source (17 files, 4083 lines total)
│   ├── __init__.py                 # Empty init
│   ├── branch_runner.py            # Main loop: Phase 1-4 orchestration (1486 lines)
│   ├── eb_agent.py                 # Planner: intent, review, diagnose (759 lines)
│   ├── executor.py                 # Executor: intent -> actions (214 lines)
│   ├── oracle_agent.py             # Oracle: injection + evaluation (288 lines)
│   ├── vlm_client.py               # VLM API client (512 lines)
│   ├── egocentric_memory.py        # Spatial memory (397 lines)
│   ├── env_controller.py           # AI2-THOR wrapper (124 lines)
│   ├── env_injector.py             # 6 injection methods (217 lines)
│   ├── task_conditions.py          # 7 task checkers + dead loop (322 lines)
│   ├── scheduler.py                # Parallel scheduler (309 lines)
│   ├── action_adapter.py           # objectId resolution + param cleanup (123 lines)
│   ├── context_builder.py          # History rendering (125 lines)
│   ├── episode_manager.py          # Episode JSON persistence (82 lines)
│   ├── trap_planner.py             # Trap selection from library (160 lines)
│   ├── fork_manager.py             # Fork reasoning rewrite (56 lines)
│   ├── step_recorder.py            # Step entry + image save (55 lines)
│   ├── alfred_scene.py             # Scene restore + task state (192 lines)
│   └── alfred_parser.py            # traj_data.json parser (148 lines)
│
├── scripts/                        # Entry point scripts (4 files)
│   ├── e2e_test.py                 # Single/episode test runner
│   ├── run_pipeline.py             # Scheduler pipeline runner
│   ├── resume_episode.py           # Episode resume from JSON
│   ├── alfred_replay.py            # ALFRED trajectory replay
│   ├── bench_api.py                # API benchmark
│   ├── download_alfred.py          # ALFRED dataset downloader
│   ├── e2e_test_output.py          # Output analysis
│   ├── finetune_qwen3vl.py         # Fine-tuning script
│   ├── generate_ft_data.py         # FT data generator
│   ├── regenerate_screenshots.py   # Screenshot regeneration
│   └── replay_demo.py              # Demo replay
│
├── data/                           # Static data assets
│   ├── json_2.1.0/                 # ALFRED trajectory data (not in repo)
│   └── failure_type_library.json   # 66 trap entries (8 types)
│
├── logs/                           # API call logs generated at runtime
│   ├── api_calls.jsonl             # Full API exchange log
│   └── api_calls.log               # Formatted log
│
├── output_e2e_*/                   # Episode output directories (timestamped)
│   └── {episode_id}/
│       ├── {episode_id}.json       # Episode data (incremental)
│       ├── s0.png, s1.png, ...     # Step screenshots
│       ├── s0_look_ahead.png, ...  # LookAround views
│       └── failures_main.jsonl     # Failure event log
│
├── tests/                          # Test directory (empty/minimal)
│
├── docs/
│
├── .planning/                      # GSD planning data
│   └── codebase/                   # Codebase analysis docs
│
├── DESIGN.md                       # Original design doc (20K)
├── _ORIGINAL_DESIGN.md             # Earlier design doc (29K)
├── PLANNER_EXECUTOR_RESEARCH.md    # Research notes (9.5K)
│
├── cot.html                        # Chain-of-thought HTML viewer (20K)
├── im.html                         # Image viewer (21K)
│
├── smoke_test.py                   # Quick smoke test
├── check_task.py                   # Task condition checker
├── show_steps.py                   # Step viewer
├── _check_syntax.py                # Syntax checker
├── _check_traj.py                  # Trajectory checker
├── _inspect.py                     # Inspection tool
├── _scan.py                        # Scan utility
└── _stats.py                       # Statistics generator
```

## Directory Purposes

**`src/` (Source):**
- Purpose: All Python application logic
- Contains: 17 Python modules + `__init__.py`
- Key constraint: No subdirectories within `src/` -- flat package structure
- Import convention: All modules use `from src.module import ...` pattern

**`scripts/` (Entry Points):**
- Purpose: CLI scripts that import from `src/` and handle user-facing args
- Contains: 11 Python scripts
- Key files:
  - `e2e_test.py`: Primary CLI for running one or all tasks with `--all` parallel mode
  - `run_pipeline.py`: Uses `Scheduler` for production runs
  - `resume_episode.py`: Resumes interrupted episode from JSON state

**`data/` (Data Assets):**
- Purpose: ALFRED trajectory data and failure type library
- Contains:
  - `failure_type_library.json`: 66 trap entries across 8 failure types
  - `json_2.1.0/`: ALFRED dataset (train/valid_seen/valid_unseen splits), not committed
- Trajectory path pattern: `data/json_2.1.0/{split}/{episode_id}/traj_data.json`

**`output_e2e_*/` (Output):**
- Purpose: Generated episode output directories (30+ directories observed)
- Naming: `output_e2e_YYYYMMDD_HHMMSS/` (timestamped) or `output_YYYYMMDD_HHMMSS/`
- Structure per episode:
  - `{episode_id}.json`: Full episode data with all steps
  - `s{N}.png`: Step screenshots
  - `s{N}_look_{direction}.png`: LookAround view captures
  - `failures_{branch_id}.jsonl`: Failure events

**`logs/` (Runtime Logs):**
- Purpose: API call logs generated at runtime
- Created automatically by `VLMClient` on first use
- `api_calls.jsonl`: Full request/response bodies (controlled by `EFD_LOG_FULL_API` env var)
- `api_calls.log`: Formatted DEBUG-level logs

## Key File Locations

**Entry Points:**
- `scripts/e2e_test.py`: Task runner for single/parallel episodes
- `scripts/run_pipeline.py`: Scheduler-based batch pipeline
- `scripts/resume_episode.py`: Episode resume
- `src/branch_runner.py:run_single_branch()`: Module-level function for full episode lifecycle (also acts as entry point for Scheduler workers)

**Configuration:**
- `pyproject.toml`: Project metadata, Python 3.10 requirement, ai2thor dependency
- `.python-version`: Python version pin ("3.10")
- `CLAUDE.md`: Project guidance (WSL2 environment, model configs, architecture summary)

**Core Logic (Phase 1-4 Loop):**
- `src/branch_runner.py:54` (class `BranchRunner`): Main orchestrator with `run()` method (~200 lines for the core loop)
- `src/branch_runner.py:1406` (`run_single_branch`): Episode lifecycle wrapper
- `src/eb_agent.py:420` (class `EBAgent`): Planner functions
- `src/executor.py:56` (class `ExecutorAgent`): Executor functions
- `src/oracle_agent.py:223` (class `OracleAgent`): Oracle functions

**Environment:**
- `src/env_controller.py:19` (class `EnvController`): AI2-THOR Controller wrapper
- `src/alfred_scene.py`: ALFRED scene restore utilities
- `src/env_injector.py`: 6 injection methods
- `src/trap_planner.py`: Trap selection from library

**Memory and State:**
- `src/egocentric_memory.py:67` (class `EgocentricMemory`): Spatial memory
- `src/episode_manager.py:377` (class `EpisodeManager`): Episode JSON persistence
- `src/alfred_scene.py:1591` (`empty_task_state`/`update_alfred_task_state`): ALFRED task tracking

**VLM Integration:**
- `src/vlm_client.py:59` (class `VLMClient`): API abstraction with JSON retry
- `src/context_builder.py`: History rendering with field-level permission filtering

**Auxiliary:**
- `src/scheduler.py:606` (class `Scheduler`): Parallel scheduling
- `src/action_adapter.py`: Object resolution and action adaptation
- `src/task_conditions.py`: 7 task completion checkers
- `src/fork_manager.py:521` (class `ForkManager`): Reasoning rewriting for forks
- `src/step_recorder.py:461` (class `StepRecorder`): Step building and frame saving
- `src/alfred_parser.py`: ALFRED trajectory parsing

## Naming Conventions

**Files:**
- All Python modules: `snake_case.py` (e.g., `branch_runner.py`, `env_controller.py`)
- Entry point scripts: `snake_case.py` (e.g., `e2e_test.py`, `run_pipeline.py`)
- Utility scripts: `_snake_case.py` prefix (e.g., `_check_syntax.py`, `_stats.py`)

**Classes:**
- `PascalCase` (e.g., `BranchRunner`, `EBAgent`, `EnvController`, `OracleAgent`, `ExecutorAgent`, `EgocentricMemory`, `EpisodeManager`, `VLMClient`, `TrapPlanner`, `ForkManager`, `StepRecorder`, `BranchConfig`, `BranchResult`, `SchedulerConfig`, `Scheduler`)

**Dataclasses:**
- Same as classes: `PascalCase`
- `BranchConfig` (`src/branch_runner.py:36`), `BranchResult` (`src/branch_runner.py:46`), `SchedulerConfig` (`src/scheduler.py:592`)
- Private ones: `_ObjectEntry` (`src/egocentric_memory.py:19`), `_ObstacleEntry` (`src/egocentric_memory.py:31`)

**Functions/Methods:**
- `snake_case` (e.g., `plan_intent()`, `execute_intent()`, `diagnose_failure()`, `_execute_move_sequence()`)
- Private methods/helpers: `_snake_case` prefix (e.g., `_fix_visible_bounds()`, `_sanitize_actions()`, `_render_objects()`)
- Module-level functions: `snake_case` (e.g., `run_single_branch()`, `replay_steps()`, `inject()`)

**Constants:**
- `UPPER_SNAKE_CASE` (e.g., `_VALID_ACTIONS`, `_META_ACTIONS`, `PHASE1_SYSTEM`, `MAX_OBJECTS=15`, `AGING_THRESHOLD=20`)

**Directories:**
- `src/`, `scripts/`, `data/`, `logs/`, `docs/`, `tests/`
- Output directories: `output_e2e_YYYYMMDD_HHMMSS/` (timestamped)

## Where to Add New Code

**New Feature (e.g., new trap type, new task type):**
- Primary code: `src/` directory (flat, no subdirectories)
- Logic in existing module if it fits (e.g., new injection method in `src/env_injector.py`, new task checker in `src/task_conditions.py`)
- New module as separate `.py` file if crossing concerns (import pattern: `from src.new_module import ...`)
- Tests: `tests/` directory (currently minimal, add test files with `test_` prefix)

**New Component/Module:**
- Implementation: `src/{name}.py` (flat structure, no nested packages)
- Import pattern: `from src.{module} import {ClassName}`
- No `__init__.py` re-exports needed (all imports use direct path)

**New Script/Entry Point:**
- File: `scripts/{name}.py`
- Pattern: argparse-based CLI, imports from `src/` modules
- If it's a utility not meant for direct use: prefix with `_` (e.g., `_check_traj.py`)

**Utilities:**
- Shared helpers go in existing module if tightly coupled
- New utility module: `src/{name}.py` (e.g., not yet: `src/geometry.py`, `src/validation.py`)
- Action adapters: `src/action_adapter.py`
- Context rendering: `src/context_builder.py`

**Configuration:**
- Add env vars to `CLAUDE.md` (project instructions)
- Add config fields to dataclasses like `SchedulerConfig` (`src/scheduler.py:592`)
- API config: `VLMClient` factory methods in `src/vlm_client.py:69-88`

## Special Directories

**`__pycache__/` directories:**
- Purpose: Python bytecode cache
- Generated: Yes (by Python runtime)
- Committed: No (in `.gitignore`)

**`.venv/`:**
- Purpose: Virtual environment (uv-managed)
- Generated: Yes (by `uv venv`)
- Committed: No

**`.planning/`:**
- Purpose: GSD planning data (codebase maps, phase plans)
- Generated: Yes (by GSD tools)
- Committed: Yes

**`output_e2e_*/` and `output_*/`:**
- Purpose: Episode run output directories
- Generated: Yes (at runtime)
- Committed: No (typically)

---

*Structure analysis: 2026-06-12*
