# Coding Conventions

**Analysis Date:** 2026-06-12

## Naming Patterns

**Files:**
- `snake_case.py` throughout — all source files in `src/` follow this pattern
- Scripts in `scripts/` also use `snake_case.py`

**Classes:**
- `PascalCase` — `BranchRunner`, `VLMClient`, `EBAgent`, `OracleAgent`, `EpisodeManager`, `EnvController`, `EgocentricMemory`, `TrapPlanner`, `StepRecorder`, `ForkManager`, `ExecutorAgent`, `Scheduler`
- Dataclasses: `BranchConfig`, `BranchResult`, `SchedulerConfig`, `_ObjectEntry`, `_ObstacleEntry`

**Functions/Methods:**
- `snake_case` — `run_single_branch()`, `build_phase1_prompt()`, `propose_action()`, `plan_intent()`, `execute_intent()`, `diagnose_failure()`, `check_task_complete()`, `decide_injection()`, `evaluate_failure()`
- Private methods/functions: `_leading_underscore` — `_encode_image()`, `_sanitize_actions()`, `_extract_task_target_types()`, `_egocentric_angle()`, `_egocentric_direction()`, `_age_entries()`, `_render_objects()`

**Constants:**
- `UPPER_SNAKE_CASE` at module level — `_VALID_ACTIONS`, `_META_ACTIONS`, `PHASE1_SYSTEM`, `PHASE2_SYSTEM`, `PHASE3_SYSTEM`, `PHASE4_SYSTEM`, `PLANNER_SYSTEM`, `PLANNER_REVIEW_SYSTEM`, `EXECUTOR_SYSTEM`, `_OBJECT_ACTIONS`, `_DEPRECATED_PARAMS`, `_CHECKERS`, `_TASK_ALIASES`, `_CRITERIA_RULES`, `_ENTITY_ALIASES`, `EB_FIELDS`, `ORACLE_FIELDS`
- Module-level mutable constants: `ALL_TASKS` in `scripts/e2e_test.py`

**JSON/Dict Keys:**
- `snake_case` in episode step data — `step_index_in_branch`, `error_message`, `eb_reasoning`, `oracle_injection_decision`, `counterfactual_grade`, `recovery_verdict`
- `step_id` format: `s0`, `s1`, `s2` — same format across all branches; `branch_id` differentiates
- Step entry schema includes `None`-initialized fields for all agent fields (see `step_recorder.py:build_step()`)

## Code Style

**Formatting:**
- No linter or formatter configured (`pyproject.toml` has no such dependencies, no `ruff`/`black`/`isort`)
- No `.eslintrc`, `.prettierrc`, `biome.json`, `ruff.toml`, or `pyproject.toml` tool config
- Style is manually maintained — varies between files (inconsistent whitespace, line lengths)

**Linting:**
- None configured

## Import Organization

**Order:**
1. Standard library (`os`, `json`, `math`, `time`, `threading`, `dataclasses`, `collections`, `atexit`, `subprocess`, `base64`, `io`, `pathlib`, `glob`, `concurrent.futures`)
2. Third-party (`numpy`, `ai2thor`, `PIL`)
3. Local (`src.xxx`)
- Blank line groups between each category

**Path Aliases:**
- No aliases — `from src.xxx import yyy` used throughout
- Some modules use lazy imports inside functions (`from src.eb_agent import _direction, _append_task_context` in `executor.py:79`)
- `executor.py` also lazy-imports `numpy` and `PIL` inside `vlm_client.py:chat_with_images()`

**`from __future__ import annotations`:**
- Used in: `executor.py`, `egocentric_memory.py`, `context_builder.py` (3 of 18 source files)

## Multi-Agent Architecture Pattern

The project implements a **Planner(32B) + Executor(8B) + Oracle(32B)** split across three agent classes. Each agent has a dedicated system prompt and methods.

**EBAgent (`src/eb_agent.py`):**
- `EBAgent.plan_intent()` — Phase 1 Planner: proposes high-level intent from current state (32B model)
- `EBAgent.review_actions()` — Phase 1 Reviewer: approves/rejects Executor's proposed action sequence (32B)
- `EBAgent.propose_action()` — Fallback single-agent Phase 1 path (32B, used when `executor_agent=None`)
- `EBAgent.propose_action_lookaround()` — Special case for LookAround multi-image decision (32B)
- `EBAgent.analyze_scan_room()` — Analyzes 4-directional scan to produce direction + intent (32B)
- `EBAgent.diagnose_failure()` — Phase 3 failure diagnosis + recovery reasoning (32B)
- Private helpers: `_egocentric_angle()`, `_is_in_front()`, `_direction()`, `_append_task_context()`
- System prompts as module-level constants: `PHASE1_SYSTEM`, `PHASE3_SYSTEM`, `PLANNER_SYSTEM`, `PLANNER_REVIEW_SYSTEM`, `SCAN_ANALYSIS_SYSTEM`

**ExecutorAgent (`src/executor.py`):**
- `ExecutorAgent.execute_intent()` — Takes Planner intent + full context, outputs 1-5 concrete actions (8B model)
- `ExecutorAgent._sanitize_actions()` — Caps excessive actions (max 12 total, max 6 consecutive same-direction)
- Shares private helpers from `EBAgent` via lazy import: `from src.eb_agent import _direction, _append_task_context`
- System prompt: `EXECUTOR_SYSTEM` (module-level constant)
- **"repeat" convention:** Executor batches same-direction movement with `"repeat": N` instead of individual per-step entries. `N = ceil(distance / 0.125)`

**OracleAgent (`src/oracle_agent.py`):**
- `OracleAgent.decide_injection()` — Phase 2: decides whether to inject a cascading failure (32B)
- `OracleAgent.evaluate_failure()` — Phase 4: evaluates EB diagnosis, grades counterfactual (WA/PA/AC), decides fork
- System prompts: `PHASE2_SYSTEM`, `PHASE2_SYSTEM_FORK`, `PHASE4_SYSTEM`

**Decision flow when executor is active (`branch_runner.py:210-358`):**
1. Planner outputs `{intent, target, reasoning}`
2. If `intent == "Done"` → hard check via `check_task_complete()`, reject with specific reason if not met
3. If `intent == "scan room"` → 4-view capture → `analyze_scan_room()` → Rotate to face direction → fall through to Executor with new intent
4. Executor outputs `{actions: [...], status, reasoning, status_reason}`
5. Reviewer outputs `{approved, reason, corrected_actions}`
6. If approved: action = `MoveSequence` with Executor's actions
7. If rejected: action = `MoveSequence` with Reviewer's `corrected_actions`

## Intent Tracking Pattern

In `branch_runner.py:106-107`:
```python
intent_history: list[dict] = []  # completed/failed intent summaries
current_intent = {"intent": "", "target": "", "steps": [], "start_step": 0}
```
- On intent change: close previous intent (set `completed` and `end_step`), append to `intent_history`, start new `current_intent`
- Executor receives only `current_intent["steps"]` as action history (not full `eb_history`)
- Planner sees full `intent_history` (last 15 entries) for context

## Meta-Actions Convention

**`_META_ACTIONS`** module-level constant in `branch_runner.py:32`:
```python
_META_ACTIONS = {"Done", "LookAround"}
```
These are handled by the Planner / branch_runner loop and must NEVER reach AI2-THOR. The `_execute_move_sequence()` method guards against Executor leaking meta-actions into MoveSequence:
- If a meta-action appears in the middle of a sequence → truncate before it (keep what was executed)
- If a meta-action appears as the first step → fail with explicit error

## Error Handling

**Patterns:**
- "Let it crash" philosophy — `RuntimeError` for unrecoverable states:
  - Invalid action after 3 retries — `raise RuntimeError(error_msg)` in `branch_runner.py`
  - Injection setup failure after 3 attempts — `raise RuntimeError(...)` in `branch_runner.py`
  - Initial LookAround rotation failure — `raise RuntimeError(f"Init LookAround RotateLeft failed: ...")` in `branch_runner.py`
  - Replay mismatch — `raise RuntimeError(f"Replay failed at ...")` in `branch_runner.py`
  - ALFRED scene restoration failure — `raise RuntimeError` in `alfred_scene.py`
  - VLM JSON parse failure after max retries — `raise ValueError(...)` in `vlm_client.py`
- `try/finally` used in `run_single_branch()` and `_run_fork()` to guarantee `env.close()` cleanup
- No custom exception hierarchy — all built-in types
- Assertion-style: `_step_or_raise()` pattern used in `alfred_scene.py` and `env_injector.py`

**Retry patterns:**
| Failure Type | Max Retries | Backoff | File |
|---|---|---|---|
| VLM API connection error | 3 | 1s, 2s | `vlm_client.py:337-343` |
| VLM API timeout | 3 | +100s each (300→400→500) | `vlm_client.py:345-355` |
| VLM API rate limit (429) | 3 | 5s, 10s, 15s | `vlm_client.py:326-331` |
| JSON parse failure | 2 | Immediate (retry prompt with error feedback) | `vlm_client.py:236-252` |
| Injection execution | 2 (total 3 attempts) | Immediate | `branch_runner.py:447-509` |
| EB non-executed action (invalid/object resolution/repeated LookAround) | 3 | Immediate | `branch_runner.py` scattered |

**Validation before execution:**
- Invalid action names blocked via `_VALID_ACTIONS` check in `branch_runner.py:379`
- Object resolution warnings caught before AI2-THOR call (`branch_runner.py:773-798`)
- Task completion verified by `check_task_complete()` before accepting `Done`
- Fork reasoning rewriting: text must not mention "counterfactual", "alternative", "fork", or "correction"
- MoveSequence execution stops on first failure (`_execute_move_sequence()` at `branch_runner.py:1017`)

## Logging

**Framework:** Python `logging` module

**Setup in `vlm_client.py`:**
- Logger name: `"vlm_client"`
- File handler: `logs/api_calls.log` at DEBUG level
- Stream handler: stderr at INFO level
- Global log directory: `LOG_DIR = Path("logs")` (module-level, created on import)

**Structured JSONL logging:**
- API call details → `logs/api_calls.jsonl` (from `_write_api_exchange()`)
- Per-episode failures → `failures_{branch_id}.jsonl` in episode output dir (from `_write_failure_log()`)
- Failure dump files → `logs/failure_{timestamp}_{model}.json` (from `_dump_failure()`, non-retryable API errors only)
- JSONL lines use `ensure_ascii=False` for Chinese characters

**Console:**
- `scheduler.py` uses `\r` carriage return for in-place progress updates (`_report()`)
- Scripts print summary tables with `=` separators

**Env var controls:**
- `EFD_LOG_FULL_API` (default "1") — set to "0" to suppress full API JSONL logging

## Comments

**When to Comment:**
- Module-level docstrings in most files describing module purpose (some in Chinese, some in English)
- Section separators: `# ----`, `# === ===` in `branch_runner.py`, `oracle_agent.py`, `eb_agent.py`
- Minimal inline comments — code is mostly self-documenting through naming
- No formal JSDoc/Google/NumPy docstring format
- Chinese comments found in: `branch_runner.py`, `eb_agent.py`, `oracle_agent.py`, `action_adapter.py`, `env_controller.py`

## Function Design

**Size:**
- `BranchRunner.run()` — largest function at ~850 lines (`branch_runner.py:78-932`)
- Most other functions are 10-80 lines
- Data helpers are concise (2-15 lines)
- EgocentricMemory class: 397 lines across all methods

**Parameters:**
- Functions with many parameters use named arguments extensively
- Long parameter lists in agent methods: `plan_intent()` has 12 parameters, `execute_intent()` 13, `diagnose_failure()` 11, `decide_injection()` 10
- No builder/factory pattern for complex parameter construction
- Some functions pass `**kwargs` through stub agents in tests

**Return Values:**
- Dict type returns for structured data — `{"success": bool, "error": str}`, `{"action": str, "params": dict, "reasoning": str}`
- `(bool, str)` tuple returns for validation functions like `check_task_complete()`
- `BranchResult` dataclass for runner return value
- `str | None` for error messages / warnings

## Module Design

**Exports:**
- No `__all__` defined in any module
- `src/__init__.py` is empty — no public API surface
- Each major class in its own file
- Utility modules group related functions without a class: `context_builder.py`, `task_conditions.py`, `action_adapter.py`

**Barrel Files:** None. Importers reference `src.xxx` directly.

## Dataclass Pattern

Used for configuration and result objects:
- `BranchConfig` — per-branch runtime config (`episode_id`, `branch_id`, `parent_branch_id`, `shared_context_step_ids`, `diverges_at_step_id`, `fork_config`)
- `BranchResult` — branch execution result (`branch_id`, `termination_reason`, `total_steps`, `fork_tasks`, `fork_source_step_ids`)
- `SchedulerConfig` — global scheduler config
- `_ObjectEntry` / `_ObstacleEntry` — internal EgocentricMemory data (dataclass with `field(default_factory=...)`)
- Dataclasses use `Optional[...]` for nullable fields

## EgocentricMemory Conventions

**Key constants** (`src/egocentric_memory.py:68-71`):
- `AGING_THRESHOLD: int = 20` — forget remembered objects after 20 unseen steps
- `MAX_OBJECTS: int = 15` — max objects rendered in memory text
- `MAX_OBSTACLES: int = 3` — max obstacles shown in memory
- `BLOCK_THRESHOLD: int = 2` — require 2+ blocks before reporting obstacle

**Internal data structures:**
- `_ObjectEntry` dataclass — tracks `object_id`, `object_type`, `is_receptacle`, `is_task_receptacle`, `status` (visible/held/placed/remembered), `egocentric_dir`, `egocentric_dist`, `last_seen_step`, `seen_count`
- `_ObstacleEntry` dataclass — tracks `object_type`, `direction`, `block_count`, `last_block_step`
- Objects stored by `objectId` in `self._objects: dict[str, _ObjectEntry]`
- Obstacles stored as list `self._obstacles: list[_ObstacleEntry]`

**Key behaviors:**
- Only accumulates objects with `visibleBounds2D=True`
- Handles `held` status from both `inventoryObjects` and `visible_objects[].isPickedUp`
- Handles `PutObject` success: marks held objects as "placed"
- `_age_entries()` removes remembered entries older than `AGING_THRESHOLD`
- `_update_area_label()` infers room type from visible object types (bedroom/kitchen/living room/office)

## Prompt Management

**Pattern:**
- System prompts as module-level string constants (`PHASE1_SYSTEM`, `EXECUTOR_SYSTEM`, `PHASE2_SYSTEM`, `PHASE3_SYSTEM`, `PHASE4_SYSTEM`)
- Prompt builder functions: `build_phase1_prompt()`, `build_phase2_prompt()`, `build_phase3_prompt()`, `build_phase4_prompt()`, `build_eb_history_context()`, `build_oracle_history_context()`
- Multi-step conditional assembly: memory text first, then hand status, task criteria, visible objects, recent history, last error
- Multi-image prompts add direction labels: `"VIEW 1: ahead"`, `"VIEW 2: left"`, etc. (in `vlm_client.py`)
- Output enforcement: `required_fields` tuple checked after JSON parsing + retry
- JSON retry injects specific error feedback into the user prompt

**History rendering (`context_builder.py`):**
- Uses compact JSONL format: one JSON object per step, `ensure_ascii=False, sort_keys=True, separators=(",", ":")`
- EB history strips oracle fields and internal objectId params
- Oracle history includes oracle fields with shortened keys (`oracle_` prefix stripped)
- Branch history construction: shared parent steps first, then current branch steps

## Commit Conventions

- Messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- No conventional commits prefix (no `feat:`, `fix:`, etc.)
- Direct to `master` branch

## Type Hints

**Status:** Inconsistent across files

- **Modern syntax (`str | None`, `dict | None`, `list[dict]`):** Used in `eb_agent.py`, `executor.py`, `oracle_agent.py`, `branch_runner.py`, `egocentric_memory.py`, `env_injector.py`, `context_builder.py`, `step_recorder.py`
- **Legacy syntax (`Optional[str]`, `Optional[dict]`):** Still appears in `trap_planner.py` (`from typing import Optional`), `vlm_client.py` (`from typing import Optional`), `alfred_parser.py`
- **No type hints:** Many functions in utility modules are untyped or partially typed
- `from __future__ import annotations` in 3 of 18 source files
- No `mypy`/`pyright`/`pyanalyze` configuration in `pyproject.toml`

## Anti-Patterns Observed

**Large monolithic function:**
- `BranchRunner.run()` is ~850 lines (`branch_runner.py:78-932`), handling Phase 1-4 loop, LookAround, MoveSequence, error handling, fork creation, with deeply nested conditionals

**Hardcoded values:**
- `LOG_DIR = Path("logs")` at module-level in `vlm_client.py`
- API timeout ramp: 300→400→500s hardcoded in `vlm_client.py:309`
- `AGING_THRESHOLD: int = 20`, `MAX_OBJECTS: int = 15`, `BLOCK_THRESHOLD: int = 2` as class variables in `EgocentricMemory`
- `_CLOSE_BLOCKLIST = {"Blinds"}` in `env_injector.py` — AI2-THOR timeout workaround
- `_ENTITY_ALIASES` dict in `egocentric_memory.py` — hardcoded wall/receptacle aliases
- Step limit: `if step_index >= 200: return ...` hardcoded in `branch_runner.py:176`

**Module-level mutable state:**
- `EnvController._xvfb_proc` class variable — Xvfb process singleton; `_xvfb_display = ":99"` — global default display
- `logger` setup at module level in `vlm_client.py`
- `LOG_FULL_API = os.environ.get(...)` evaluated at module import time

**String-based dispatch:**
- `_CHECKERS` dict mapping task type strings to checker functions (`task_conditions.py:304-311`)
- `_INJECTORS` dict mapping method strings to handler functions (`env_injector.py`)
- `_ADAPTERS` dict mapping action strings to adapters (`action_adapter.py:120-123`)

**Lazy imports within functions:**
- `from src.eb_agent import _direction, _append_task_context` inside `ExecutorAgent.execute_intent()`
- `import numpy as np` and `from PIL import Image` inside `vlm_client.chat_with_images()` and `vlm_client.chat_with_image()`

**Action type confusion:**
- `MoveSequence` is both an action proposed by EB and a code execution path. `params` contains `{"steps": [{"action": "MoveAhead", ...}]}` — the inner action format duplicates outer action syntax.

**Hardcoded API key:**
- API key (`sk-...`) visible in `CLAUDE.md` as environment variable documentation and potentially in code history

**Context window risk:**
- No explicit token budget control in prompt building — VLM context grows with task steps, risking truncation

---

*Convention analysis: 2026-06-12*
