<!-- refreshed: 2026-07-15 -->
# Architecture

**Analysis Date:** 2026-06-12

## System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Entry Points                                      │
│  scripts/e2e_test.py  scripts/run_pipeline.py  scripts/resume_episode.py   │
│  (argparse)            (Scheduler)              (CLI args: episode.json)   │
└───────────┬───────────────────┬──────────────────────────────────┬─────────┘
            │                   │                                  │
            ▼                   ▼                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     Scheduler (src/scheduler.py)                           │
│  SchedulerConfig + Scheduler                                               │
│  - Phase 1: ThreadPoolExecutor main branches (parallel)                    │
│  - Phase 2: serial fork branches (depends on parent completion)            │
│  - Warmup oracle model before starting                                     │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    BranchRunner (src/branch_runner.py)                     │
│  run_single_branch() — full episode lifecycle                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 1: Planner -> Executor -> Review -> Execute                  │   │
│  │                                                                      │   │
│  │  Planner plan_intent (32B)      --> high-level intent               │   │
│  │     +-- "scan room"    --> 4view --> 32B analyze_scan_room          │   │
│  │     +-- "Done"         --> task_conditions hard check               │   │
│  │     \-- normal intent  --> Executor                                 │   │
│  │                                                                      │   │
│  │  Executor execute_intent (8B)   --> 1-5 concrete actions            │   │
│  │     - "repeat" batching for same-direction moves                     │   │
│  │     - SOLID obstacle labels for large furniture                      │   │
│  │                                                                      │   │
│  │  Planner review_actions (32B)   --> approve or corrected_actions     │   │
│  │     - Default approve (only reject critical errors)                  │   │
│  │     - STUCK detection: 2+ consecutive -> MoveBack is correct         │   │
│  │                                                                      │   │
│  │  MoveSequence execution --> sequential, stops on first failure      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 2: Oracle Injection (cascade_level <= 1)                      │   │
│  │  oracle_agent.decide_injection() --> env_injector.inject()          │   │
│  │  - 6 methods: set_object_property, close_container, occlude_object,  │   │
│  │    hide_object, swap_object, remove_object                           │   │
│  │  - Guard: irreversible props / target-modify on critical objects     │   │
│  │  - Max 3 injections per episode, auto-retry on inject failure        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 3: Planner diagnosis (eb_agent.diagnose_failure)              │   │
│  │  - Gets current_intent context                                        │   │
│  │  - eb_diagnosis / eb_proposed_recovery_action from Planner            │   │
│  │  - eb_counterfactual (optional structured format)                    │   │
│  │                                                                      │   │
│  │  Phase 4: Oracle evaluation (oracle_agent.evaluate_failure)          │   │
│  │  - diagnosis_correct, ground_truth, counterfactual_grade (WA/PA/AC)  │   │
│  │  - recovery_verdict (recovered/recoverable/unrecoverable)            │   │
│  │  - should_fork (only AC grade -> create fork task)                   │   │
│  │                                                                      │   │
│  │  Recovery execution: if verdict=recoverable, execute recovery action │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Termination paths:                                                         │
│    task_complete | step_hard_limit(200) | dead_loop | unrecoverable         │
│    | unrecoverable_hard:... | worker_crash:...                              │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              AI2-THOR 5.0.0 Simulator (src/env_controller.py)              │
│  - WSLg (:0) or Xvfb fallback                                              │
│  - _fix_visible_bounds() for visibleBounds2D bug                           │
│  - ALFRED scene restore via restore_alfred_scene()                         │
│  - ALFRED task state tracking (heated/cooled/cleaned objects)              │
└────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File | Lines |
|-----------|---------------|------|-------|
| BranchRunner | Main Phase 1-4 loop, MoveSequence, recovery, resume | `src/branch_runner.py` | 2390 |
| EBAgent | Planner: plan_intent, review_actions, diagnose_failure, analyze_scan_room | `src/eb_agent.py` | 1760 |
| ExecutorAgent | Executor: intent -> concrete action chunk | `src/executor.py` | 178 |
| OracleAgent | Phase 2 injection, Phase 4 evaluation | `src/oracle_agent.py` | 288 |
| VLMClient | API calls, HTTP-200 5-stage validation, JSON retry, logging | `src/vlm_client.py` | 719 |
| SemanticMemory | Receptacle-grouped memory, freshness, task tracking (DEFAULT) | `src/semantic_memory.py` | 403 |
| GeometricMemory | Flat-list geometric memory (legacy fallback) | `src/geometric_memory.py` | 871 |
| MemoryInterface | Shared memory API base | `src/memory_interface.py` | 93 |
| EgocentricMemory | Backward-compat shim | `src/egocentric_memory.py` | 7 |
| EnvController | AI2-THOR wrapper, Xvfb, visibleBounds2D fix | `src/env_controller.py` | 124 |
| Scheduler | Parallel main branches (per-worker agents), serial forks | `src/scheduler.py` | 325 |
| TaskConditions | 7 task completion checkers, dead_loop, unrecoverable | `src/task_conditions.py` | 322 |
| ActionAdapter | objectType->objectId resolution, PickupObject invisible-object guard | `src/action_adapter.py` | 123 |
| ContextBuilder | EB/Oracle history rendering (field permissions) | `src/context_builder.py` | 125 |
| EpisodeManager | Episode JSON incremental flush | `src/episode_manager.py` | 82 |
| TrapPlanner | Trap selection from failure_type_library.json | `src/trap_planner.py` | 160 |
| ForkManager | Fork reasoning rewriting | `src/fork_manager.py` | 56 |
| StepRecorder | Step entry building, frame saving | `src/step_recorder.py` | 55 |
| AlfredScene | Scene restore, task state, Faucet plumbing | `src/alfred_scene.py` | 192 |
| AlfredParser | traj_data.json loading, api_action parsing | `src/alfred_parser.py` | 148 |

## Pattern Overview

**Overall:** Agentic four-phase loop with Planner(32B)+Executor(8B)+Oracle(32B) VLM delegation.

**Key Characteristics:**
- **Oracle-blinded**: Oracle sees unified EB history, does not know about Planner/Executor split
- **Intent-level memory scoping**: Planner sees intent_history + last 3 raw steps; Executor sees only current-intent steps
- **Special intent interception**: "scan room" and "Done" handled by branch_runner, never reach AI2-THOR
- **visibleBounds2D filtering**: All "objects in view" lists filtered by instance segmentation (not metadata.visible)
- **Spatial memory**: Cumulative across steps, objects persist with direction/distance, aged after 20 steps
- **MoveSequence**: Sequential batched movement, stops on first failure, supports "repeat"
- **gridSize=0.125m**: All movement steps are 0.125m/tick
- **Planner review**: Default-approve, only reject on critical errors

## Layers

**Entry Points:**
- Purpose: Script-level entry points that parse args and orchestrate runs
- Location: `scripts/`
- Contains: `e2e_test.py`, `run_pipeline.py`, `resume_episode.py`
- Depends on: All `src/` modules
- Constraint: `e2e_test.py` uses `ThreadPoolExecutor` directly; `run_pipeline.py` uses `Scheduler`

**Phase 1 (EB Agent: Planner + Executor):**
- Purpose: Propose and execute actions toward task completion
- Location: `src/eb_agent.py`, `src/executor.py`
- Contains: Plan intents, review actions, execute move sequences
- Depends on: `VLMClient`, `EgocentricMemory`, `ContextBuilder`
- Used by: `BranchRunner`

**Phase 2 (Oracle Injection):**
- Purpose: Inject environmental failures at strategic moments
- Location: `src/oracle_agent.py` (decide_injection), `src/env_injector.py` (inject)
- Contains: Oracle injection decisions, 6 injection methods, task-critical guards
- Guard: cascade_level <= 1, remaining_injections > 0, not on passive actions

**Phase 3 (EB Agent: Diagnosis):**
- Purpose: Diagnose failure and propose recovery
- Location: `src/eb_agent.py` (diagnose_failure)
- Contains: Failure diagnosis, recovery reasoning, counterfactual proposal

**Phase 4 (Oracle: Evaluation):**
- Purpose: Evaluate EB diagnosis and decide fork/recovery
- Location: `src/oracle_agent.py` (evaluate_failure)
- Contains: Diagnosis correctness, counterfactual grading (WA/PA/AC), recovery verdict, fork decision

**Environment Layer:**
- Purpose: AI2-THOR simulation management
- Location: `src/env_controller.py`, `src/alfred_scene.py`
- Contains: Controller wrapper, ALFRED scene restore, Xvfb management
- Depends on: AI2-THOR 5.0.0

**Data Layer:**
- Purpose: Episode data persistence
- Location: `src/episode_manager.py`, `src/step_recorder.py`, `src/context_builder.py`
- Contains: JSON serialization, step recording, history rendering with permission filtering
- Key pattern: `EpisodeManager` flushes to disk after every step

## Data Flow

### Primary Request Path (Phase 1)

1. **Initial LookAround** (step 0): agent rotates 4 directions, captures images, `EBAgent.analyze_scan_room()` produces direction + first intent (`src/branch_runner.py:155-173`)
2. **Planner plan_intent**: `EBAgent.plan_intent()` with intent_history + recent 3 raw steps (`src/branch_runner.py:218-231`)
3. **Special intents intercepted**: "scan room" -> 4view + analysis; "Done" -> task_conditions check (`src/branch_runner.py:247-319`)
4. **Executor execute_intent**: `ExecutorAgent.execute_intent()` with scoped current-intent history + SOLID obstacle labels (`src/branch_runner.py:322-335`)
5. **Planner review**: `EBAgent.review_actions()` -- approve or corrected_actions (`src/branch_runner.py:338-359`)
6. **MoveSequence**: `_execute_move_sequence()` sequential, stops on first failure (`src/branch_runner.py:619-768`)
7. **Success**: `EpisodeManager.add_step()`, `EgocentricMemory.update()`

### Failure Path (Phase 3 + 4)

1. MoveSequence or single action fails (`src/branch_runner.py:646-665` or `831-850`)
2. `cascade_level += 1`, `last_error` set, failed objectIds tracked
3. Phase 3: `EBAgent.diagnose_failure()` (`src/branch_runner.py:667-679`)
4. Phase 4: `OracleAgent.evaluate_failure()` (`src/branch_runner.py:682-704`)
5. Hard checks: `check_unrecoverable()`, `detect_dead_loop()` (`src/branch_runner.py:706-714`)
6. Oracle verdict: "unrecoverable" -> terminate; "recoverable" -> execute recovery (`src/branch_runner.py:720-768`)
7. Fork: counterfactual_grade="AC" + enable_fork -> create fork task (`src/branch_runner.py:920-930`)

### Resume Path

1. `BranchRunner.resume()` loads JSON, replays branch steps (`src/branch_runner.py:939-977`)
2. `replay_steps()` expands MoveSequence, replays LookAround rotations (`src/branch_runner.py:1283-1355`)
3. Continues from `start_step_index = last_step_index + 1`

### State Management

- **Per-episode JSON** (`src/episode_manager.py`): Written incrementally after every step
- **Spatial memory** (`src/egocentric_memory.py`): Per-branch in-memory, updated every step, ~1500 char compact text
- **ALFRED task state** (`src/alfred_scene.py`): In `EnvController`, tracks heated/cooled/cleaned, updated per step
- **intent_history** (`src/branch_runner.py`): Completed/failed intents, tracked as intents change
- **failed_object_ids** (`src/branch_runner.py`): Set of failed objectIds, prevents re-proposing

## Key Abstractions

**BranchConfig (`src/branch_runner.py:36-42`):**
- Purpose: Configuration for one branch (main or fork)
- Fields: episode_id, branch_id, parent_branch_id, shared_context_step_ids, diverges_at_step_id, fork_config
- Pattern: Dataclass

**BranchResult (`src/branch_runner.py:46-51`):**
- Purpose: Termination result of a branch
- Fields: branch_id, termination_reason, total_steps, fork_tasks, fork_source_step_ids
- Pattern: Dataclass

**BranchRunner (`src/branch_runner.py:54`):**
- Purpose: Main orchestration class, owns the Phase 1-4 loop
- Key methods: `run()`, `resume()` (classmethod), `_execute_move_sequence()`, `_build_fork_task()`, `_finalize()`

**EBAgent (`src/eb_agent.py:420`):**
- Purpose: Central agent for planning, reviewing, diagnosing (32B model)
- Key methods: `plan_intent()`, `review_actions()`, `propose_action()`, `propose_action_lookaround()`, `analyze_scan_room()`, `diagnose_failure()`
- System prompts: `PHASE1_SYSTEM`, `PLANNER_SYSTEM`, `PLANNER_REVIEW_SYSTEM`, `PHASE3_SYSTEM`, `SCAN_ANALYSIS_SYSTEM`

**ExecutorAgent (`src/executor.py:56`):**
- Purpose: Low-level action chunk producer (8B model)
- Key methods: `execute_intent()`, `_sanitize_actions()`

**OracleAgent (`src/oracle_agent.py:223`):**
- Purpose: Supervisor for injection and evaluation (32B model)
- Key methods: `decide_injection()` (Phase 2), `evaluate_failure()` (Phase 4)
- System prompts: `PHASE2_SYSTEM`, `PHASE2_SYSTEM_FORK`, `PHASE4_SYSTEM`

**EgocentricMemory (`src/egocentric_memory.py:67`):**
- Purpose: Cumulative spatial memory of all seen objects
- Constants: AGING_THRESHOLD=20, MAX_OBJECTS=15, MAX_OBSTACLES=3, BLOCK_THRESHOLD=2
- Methods: `update()`, `render()`

**VLMClient (`src/vlm_client.py:59`):**
- Purpose: VLM API abstraction
- Factory methods: `ollama()`, `openrouter()`, `openai()`, `siliconflow()`
- Methods: `chat_with_image_json()`, `chat_with_images_json()`, `chat_text_json()`
- Retry: JSON parse (2 attempts), connection (3 attempts, exponential), timeout (300->400->500s)

## Entry Points

**e2e_test.py (`scripts/e2e_test.py`):**
- Triggers: CLI `--api-key sk-xxx --task pick_and_place_simple` or `--all`
- Responsibilities: Parse args, create agents per worker, call `run_single_branch()`
- Supports: `--all` (7 tasks parallel), `--task`, `--random`, `--no-traps`, `--parallel N`

**run_pipeline.py (`scripts/run_pipeline.py`):**
- Triggers: CLI `--max 10` etc.
- Responsibilities: Create `SchedulerConfig`, instantiate `Scheduler`, call `run()`

**resume_episode.py (`scripts/resume_episode.py`):**
- Triggers: CLI `<episode.json> <api-key>`
- Responsibilities: Replay steps to restore env, call `BranchRunner.resume()`

**run_single_branch (`src/branch_runner.py:1406-1485`):**
- Triggers: Called from `e2e_test.py` and `Scheduler._run_branch()`
- Responsibilities: Load ALFRED -> init env -> apply traps -> BranchRunner.run -> close env

## Architectural Constraints

- **Threading:** `Scheduler` uses `ThreadPoolExecutor` for main branches (parallel). Fork branches are serial (depend on parent). Each worker creates its own VLMClient (not thread-safe across models).
- **Global state:** `EnvController._xvfb_proc` is class-level singleton. `VLMClient` logging uses module-level logger.
- **Circular imports:** Avoided via lazy imports (executor.py imports from eb_agent inside method body).
- **Information permissions:** Oracle never sees Planner/Executor split. EB never knows fork/injection exist. Fork reasoning rewritten to sound autonomous.

## Anti-Patterns

### Meta-actions leaking into MoveSequence

**What happens:** Executor sometimes outputs "Done" or "LookAround" inside action sequences (`src/branch_runner.py:1046-1064`). These should be handled by branch_runner/Planner only.
**Why it's wrong:** Meta-actions are never sent to AI2-THOR. The sequence truncates before them.
**Do this instead:** The truncation behavior is the correct handling. The validation exists in `_execute_move_sequence`.

### Single-agent fallback still present

**What happens:** `branch_runner.py:361-377` has the old `self.eb_agent.propose_action()` path when `executor_agent is None`.
**Why it's wrong:** Bypasses intent-history and review. Raw actions directly.
**Do this instead:** Always provide `executor_agent`. Fallback kept for backward compatibility.

### Initial LookAround vs. "scan room" code duplication

**What happens:** Step 0 initial scan (`branch_runner.py:116-153`) and "scan room" intent (`branch_runner.py:262-319`) both do identical 4-view capture + VLM analysis.
**Why it's wrong:** Duplicated code paths.
**Do this instead:** Unify into a single helper method.

## Error Handling

**Strategy:** Fail-fast with explicit error types. Environment failures -> Phase 3+4. Model errors -> retry.

**Patterns:**
- **Environment failure**: Logged to `failures_{branch_id}.jsonl`, cascade_level++, Phase 3+4, recovery
- **Model invalid action**: Retry with error message, up to 3 times, then `RuntimeError`
- **Model invalid JSON**: VLMClient retry with shorter-JSON instruction, 2 attempts, then `ValueError`
- **Done rejected**: Failed step with `error_type: done_rejected`
- **Agent STUCK**: 2+ consecutive failures -> MoveBack is correct
- **Dead loop**: `detect_dead_loop()` checks last 10 steps for 5+ same action+params
- **Unrecoverable**: `check_unrecoverable()` (all target instances broken); Oracle can also declare
- **Injection failure**: Retried 3 times, then `RuntimeError`

## Cross-Cutting Concerns

**Logging:**
- `vlm_client.py`: Full request/response to `logs/api_calls.jsonl`; failure dumps to `logs/failure_*.json`
- `branch_runner.py`: Failure events to `failures_{branch_id}.jsonl`

**Validation:**
- `action_adapter.py`: objectType->objectId resolution (visible+pickupable > visible > pickupable > any)
- `env_injector.py`: Guards against target object modification
- `task_conditions.py`: 7 task-specific completion checkers
- `branch_runner.py`: Action validation against `_VALID_ACTIONS` (27 actions), meta-action filtering

**Authentication:**
- `VLMClient` passes `Bearer {api_key}` to `{base_url}/chat/completions`
- API key from CLI `--api-key` or env var

---

*Architecture analysis: 2026-06-12*
