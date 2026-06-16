# Architecture
**Analysis Date:** 2026-06-16

## Overview

The `embodied-failure-dataset` project generates a dataset of embodied agent failures with counterfactual reasoning. It runs an Embodied Agent (EB) inside the AI2-THOR 3D household simulator, using ALFRED task trajectories as ground-truth seeds. An Oracle supervisor injects failures at runtime, evaluates the agent's recovery, and spawns counterfactual "fork" branches to explore alternative action paths.

The system operates as a **multi-phase, multi-branch pipeline** orchestrated by a global `Scheduler`.

## Architectural Pattern

**Pipeline + Agent Loop + Fork Tree**

1. **Data Loading Layer** — Parses ALFRED `traj_data.json` files to extract tasks, metadata, and scene states.
2. **Environment Layer** — Wraps AI2-THOR `Controller` for scene reset, step execution, state snapshot, and Xvfb display management.
3. **Agent Layer** — Two VLM-powered agents (EB Agent + Oracle Agent) with distinct roles across four phases.
4. **Orchestration Layer** — `Scheduler` manages episode queues, parallel main-branch execution, and serial fork-branch execution. `BranchRunner` runs the Phase 1–4 loop for a single branch.
5. **Persistence Layer** — `EpisodeManager` writes incremental JSON per episode; `StepRecorder` builds step entries and saves PNG screenshots; failure events go to per-branch `.jsonl` logs.

```
  scripts/run_pipeline.py
        │
        ▼
   Scheduler (scheduler.py)
        │
        ├── load_tasks() ──► alfred_parser.py (parse ALFRED traj_data.json)
        │
        ├── Phase 1: ThreadPoolExecutor → run_single_branch() ──► BranchRunner.run()
        │                                                              │
        │                                    ┌─────────────────────────┤
        │                                    │  Per-step loop:         │
        │                                    │                         │
        │                                    │  Phase 1: Planner       │  eb_agent.py
        │                                    │      → plan_intent()    │  (EBAgent)
        │                                    │      → Executor         │  executor.py
        │                                    │      → Planner review   │  (ExecutorAgent)
        │                                    │                         │
        │                                    │  Phase 2: Oracle        │  oracle_agent.py
        │                                    │      → decide_injection │  (OracleAgent)
        │                                    │      → env_injector.py  │
        │                                    │                         │
        │                                    │  [if action fails:]     │
        │                                    │  Phase 3: EB diagnosis  │  eb_agent.py
        │                                    │  Phase 4: Oracle eval   │  oracle_agent.py
        │                                    │                         │
        │                                    │  Fork tasks returned ───┤
        │                                    └─────────────────────────┘
        │
        └── Phase 2: serial fork queue → _run_fork() → replay + BranchRunner.run()
```

## Layers

### 1. Entry Points Layer (`scripts/`)

| Script | Role |
|--------|------|
| `scripts/run_pipeline.py` | **Main CLI entry point.** Parses args, creates `SchedulerConfig`, runs `Scheduler.run()`. |
| `scripts/e2e_test.py` | End-to-end test harness; calls `run_single_branch()` directly. |
| `scripts/replay_demo.py` | Replays recorded episodes with screenshots. |
| `scripts/alfred_replay.py` | Replays raw ALFRED expert trajectories. |
| `scripts/bench_api.py` | Benchmarks VLM API latency/throughput. |
| `scripts/generate_ft_data.py` | Converts episode JSON outputs into fine-tuning data. |
| `scripts/finetune_qwen3vl.py` | Fine-tunes Qwen3-VL on generated data. |
| `scripts/regenerate_screenshots.py` | Regenerates PNG screenshots from existing episode data. |
| `scripts/download_alfred.py` | Downloads ALFRED dataset `json_2.1.0`. |

### 2. Orchestration Layer (`src/scheduler.py`, `src/branch_runner.py`)

- **`Scheduler`** — Top-level orchestrator. Loads ALFRED tasks into a queue, runs main branches in parallel via `ThreadPoolExecutor`, then processes fork branches serially.
- **`SchedulerConfig`** — Dataclass: `data_dir`, `output_dir`, `max_episodes`, `task_filter`, `splits`, model names, fork/parallel flags.
- **`BranchRunner`** — Runs the Phase 1–4 loop for a single branch. Manages step counting, cascade tracking, injection quota, egocentric memory, intent history, and dead-loop detection.
- **`BranchConfig`** / **`BranchResult`** — Dataclasses for branch identity and termination status.
- **`run_single_branch()`** — Free function: loads ALFRED trajectory, initializes environment, applies initial traps (via `TrapPlanner`), creates `EpisodeManager`, and delegates to `BranchRunner.run()`.

### 3. Agent Layer (`src/eb_agent.py`, `src/oracle_agent.py`, `src/executor.py`)

Three distinct agent personas, each using a VLM:

| Agent | Model | Role |
|-------|-------|------|
| **`EBAgent`** (Planner) | 32B VLM | Phase 1: proposes high-level intents (`plan_intent()`), 4-view scan analysis (`analyze_scan_room()`), reviews Executor actions (`review_actions()`). Phase 3: diagnoses failures (`diagnose_failure()`). Legacy: single-step `propose_action()` and `propose_action_lookaround()`. |
| **`ExecutorAgent`** | 8B VLM | Receives a Planner intent and outputs a concrete action sequence (1–12 actions with batching via `repeat`). Has rich context: spatial memory, visible objects with directions, hand status, task criteria. |
| **`OracleAgent`** | 32B VLM | Phase 2: decides whether to inject failures (`decide_injection()`). Phase 4: evaluates EB's failure diagnosis and counterfactual reasoning (`evaluate_failure()`), assigns grades (WA/PA/AC), decides whether to fork. |

### 4. Environment Layer (`src/env_controller.py`, `src/env_injector.py`, `src/alfred_scene.py`)

- **`EnvController`** — Thin wrapper around `ai2thor.controller.Controller`. Manages scene reset, step execution with metadata capture, ALFRED scene restoration, Xvfb headless display management. Also fixes an AI2-THOR 5.0.0 bug in `visibleBounds2D`.
- **`env_injector.py`** — Applies Oracle-decided injections: `set_object_property`, `swap_object`, `occlude_object`, `close_container`, `remove_object`, `hide_object`. Includes guards to prevent modifying task-critical objects.
- **`alfred_scene.py`** — Restores ALFRED scene state (object poses, toggles, dirt), manages `task_state` (cleaned/heated/cooled objects), handles `init_action`.

### 5. Data & Persistence Layer

| Module | Purpose |
|--------|---------|
| `src/alfred_parser.py` | Parses ALFRED `traj_data.json`: extracts task metadata, scene state, low-level actions; generates human-readable task goals. |
| `src/episode_manager.py` | Manages per-episode JSON output. Supports incremental `add_step()`, `add_runtime_trap()`, `set_final_outcome()` with auto-flush. Has static `load()` for resumption. |
| `src/step_recorder.py` | Builds step dict entries with all fields; saves PNG screenshots via PIL. |
| `src/context_builder.py` | Renders compact JSONL history for EB and Oracle prompts. `build_branch_history()` reconstructs shared+current-branch step lineage. |
| `src/task_conditions.py` | Hard-coded task completion checkers for all ALFRED task types. Also: `check_unrecoverable()`, `detect_dead_loop()`, `get_completion_criteria_text()`. |

### 6. Supporting Modules

| Module | Purpose |
|--------|---------|
| `src/action_adapter.py` | Adapts VLM-output actions to AI2-THOR format. Resolves `objectType`→`objectId` and `receptacleType`→`receptacleId`. Cleans deprecated params. |
| `src/trap_planner.py` | Selects initial traps from `data/failure_type_library.json` based on task type and scene objects. Resolves placeholders (`<target>`, `<container>`, etc.) to actual object types. |
| `src/fork_manager.py` | Rewrites EB reasoning text for fork branches to maintain self-consistency. |
| `src/egocentric_memory.py` | Accumulates objects the agent has ever seen into a compressed spatial memory. Tracks statuses (visible/held/placed/remembered), obstacles, area labels. Ages out stale entries. |
| `src/vlm_client.py` | VLM API client. Supports local Ollama and remote OpenAI-compatible APIs (SiliconFlow, OpenRouter). Handles image encoding, JSON mode, multi-image chat, structured output parsing. |

## Data Flow: A Typical Episode

1. **Load** — `run_pipeline.py` → `SchedulerConfig` → `Scheduler.load_tasks()` reads ALFRED JSON files via `alfred_parser.py`.
2. **Init Environment** — `run_single_branch()` creates `EnvController`, resets to ALFRED scene via `alfred_scene.py`.
3. **Initial Traps** — `TrapPlanner` selects and applies traps from the failure type library.
4. **Episode Manager** — `EpisodeManager` is created, writes initial metadata to `{episode_id}.json`.
5. **BranchRunner.run()** begins the Phase 1–4 loop:

   **Phase 1 — Plan:**
   - `env.get_state_snapshot()` captures current frame + metadata.
   - `EBAgent.plan_intent()` proposes high-level intent (e.g., "approach CounterTop").
   - `ExecutorAgent.execute_intent()` generates action sequence with batching.
   - `EBAgent.review_actions()` approves or corrects the sequence.

   **Phase 2 — Inject:**
   - `OracleAgent.decide_injection()` decides whether to inject a failure.
   - If yes, `env_injector.inject()` modifies the environment.
   - Runtime traps are recorded in the episode JSON.

   **Phase 3 — Diagnose (on failure):**
   - `EBAgent.diagnose_failure()` analyzes the error, proposes recovery and optional counterfactual.

   **Phase 4 — Evaluate (on failure):**
   - `OracleAgent.evaluate_failure()` grades the diagnosis and counterfactual.
   - If `should_fork=true` and counterfactual grade is AC (Acceptable), a fork task is queued.

   **Execute:**
   - `MoveSequence` steps are executed via `_execute_move_sequence()`.
   - Single actions go through `action_adapter.resolve_object_ids()` → `adapt()` → `env.step()`.
   - Success: step recorded, memory updated, continue loop.
   - Failure: cascade_level incremented, Phase 3+4 triggered.

6. **Termination** — Loop ends on: task complete, dead loop detected, unrecoverable verdict, or 200-step hard limit.
7. **Fork Branches** — `Scheduler._run_fork()` replays shared context, executes the alternative action, rewrites reasoning via `ForkManager`, and runs `BranchRunner.run()` from the divergence point.

## Key Abstractions & Classes

| Class | File | Role |
|-------|------|------|
| `Scheduler` | `src/scheduler.py` | Global orchestrator; manages episode/fork queues, parallel execution |
| `BranchRunner` | `src/branch_runner.py` | Phase 1–4 loop for a single branch |
| `EBAgent` | `src/eb_agent.py` | Embodied agent: planning, diagnosis, action review, scan analysis |
| `OracleAgent` | `src/oracle_agent.py` | Supervisor: injection decisions, failure evaluation, fork decisions |
| `ExecutorAgent` | `src/executor.py` | Low-level action executor (8B model) |
| `EnvController` | `src/env_controller.py` | AI2-THOR environment wrapper |
| `EpisodeManager` | `src/episode_manager.py` | Per-episode JSON persistence with incremental flush |
| `StepRecorder` | `src/step_recorder.py` | Step entry builder + PNG screenshot saver |
| `TrapPlanner` | `src/trap_planner.py` | Initial trap selection from failure type library |
| `ForkManager` | `src/fork_manager.py` | Reasoning rewriting for fork branch self-consistency |
| `EgocentricMemory` | `src/egocentric_memory.py` | Compressed spatial memory of all seen objects |
| `VLMClient` | `src/vlm_client.py` | VLM API client (SiliconFlow, Ollama, OpenRouter) |

## Branch & Fork Model

- **Main branch** — The original execution path for an episode.
- **Fork branch** — Created when Oracle identifies an acceptable counterfactual (grade AC). The fork replays shared history, substitutes an alternative action at the divergence point, and continues execution independently.
- Branch identity: `BranchConfig` carries `branch_id`, `parent_branch_id`, `shared_context_step_ids`, `diverges_at_step_id`, and `fork_config`.
- All steps (from all branches) are stored in the single `{episode_id}.json` file, keyed by `branch_id`.

## Configuration & Models

- Primary model: `Qwen/Qwen3-VL-32B-Instruct` (Planner + Oracle)
- Executor model: `Qwen/Qwen3-VL-8B-Instruct`
- API provider: SiliconFlow (OpenAI-compatible endpoint)
- Simulation engine: AI2-THOR 5.0.0
- Dataset seed: ALFRED `json_2.1.0`
