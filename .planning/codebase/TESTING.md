# Testing Practices

**Analysis Date:** 2026-06-12

## Test Framework

**Runner:**
- `unittest` — Python standard library
- No third-party test runner (no `pytest`, no `nose2`)
- No test dependency in `pyproject.toml`

**Config:**
- No config file (`pyproject.toml`, `setup.cfg`, etc. have no test configuration)

**Run Commands:**
```bash
uv run python -m unittest discover tests/           # Run all tests
uv run python -m unittest tests/test_xxx.py          # Single test file
uv run python tests/test_xxx.py                      # Single file with __main__
```

## Test File Organization

**Location:**
- All tests in `tests/` directory (separate from source, not co-located)

**Naming:**
- File pattern: `test_{module_name}.py` — `test_task_conditions_alfred.py`, `test_context_builder.py`, `test_vlm_client.py`, `test_env_injector.py`, `test_alfred_scene.py`, `test_error_handling.py`, `test_eb_agent_prompts.py`, `test_training_prompt.py`

**Structure:**
```
tests/
├── test_alfred_scene.py           # 4 tests — scene restoration, state tracking
├── test_context_builder.py        # 5 tests — EB/Oracle history rendering, branch history
├── test_eb_agent_prompts.py       # 2 tests — prompt template inspection
├── test_env_injector.py           # 3 tests — injection logic
├── test_error_handling.py         # 9 tests — BranchRunner error paths
├── test_task_conditions_alfred.py # 5 tests — all 7 task type checkers
├── test_training_prompt.py        # 2 tests — training input generation
├── test_vlm_client.py             # 1 test — multi-image request structure
```

## Test Structure

**Suite Organization:**
```python
import unittest

class ModuleNameTest(unittest.TestCase):
    def test_specific_behavior(self):
        ...

if __name__ == "__main__":
    unittest.main()
```

**Patterns:**
- Setup: Factory helper functions at module level (`make_obj()`, `make_step()`, `_obj()`) — not `setUp` class methods
- Teardown: `tempfile.TemporaryDirectory()` context manager for test output (used in `test_error_handling.py`)
- Assertions: `assertEqual`, `assertTrue`, `assertFalse`, `assertIn`, `assertNotIn`, `assertRaises`, `assertNotIsInstance`
- Test method naming: `test_descriptive_name_with_underscores` — describes the expected behavior under test

## Mocking

**Framework:** No mocking library — all mocks are manual/hand-rolled (no `unittest.mock`, no `pytest-mock`)

**Patterns:**

Fake AI2-THOR Controller:
```python
class FakeController:
    def __init__(self, metadata=None):
        self.calls = []  # Records all step() calls for assertion
        self.metadata = metadata or {"lastActionSuccess": True, "objects": []}

    def step(self, action=None, **params):
        call = dict(params)
        if action is not None:
            call["action"] = action
        self.calls.append(call)
        return FakeEvent(self.metadata)
```

Fake EB Agent for BranchRunner tests:
```python
class InvalidActionAgent:
    def __init__(self):
        self.last_errors = []

    def propose_action(self, **kwargs):
        self.last_errors.append(kwargs.get("last_error"))
        if len(self.last_errors) == 1:
            return {"action": "Fly", "params": {}, "reasoning": "invalid action"}
        return {"action": "MoveAhead", "params": {}, "reasoning": "valid retry"}
```

Fake VLM Client for prompt content testing:
```python
class CapturingClient:
    def __init__(self):
        self.user_text = None

    def chat_with_image_json(self, *, user_text, **kwargs):
        self.user_text = user_text
        return {"diagnosis": "I failed.", ...}

    def chat_with_images_json(self, *, user_text, **kwargs):
        self.user_text = user_text
        return {"action": "MoveAhead", "params": {}, "reasoning": "I move."}
```

Inline monkey-patching for module-level overrides:
```python
# test_error_handling.py — inject fake module-level functions
original_inject = branch_runner_module.inject
branch_runner_module.inject = fake_inject
try:
    ...
finally:
    branch_runner_module.inject = original_inject
```

**What to Mock:**
| Component | Mock Strategy | Tests Using It |
|---|---|---|
| `ai2thor.controller.Controller` | `FakeController` with `FakeEvent` | `test_env_injector`, `test_alfred_scene`, `test_error_handling` |
| `VLMClient` | `CapturingClient` or inline `fake_chat`/`fake_chat_text` | `test_eb_agent_prompts`, `test_vlm_client`, `test_error_handling` |
| `EBAgent` | Hand-rolled stub classes | `test_error_handling` |
| `OracleAgent` | `NoopOracle` stub | `test_error_handling` |
| `EnvController` | `CompletingAfterMoveEnv`, `FailingMoveEnv`, `LookAroundThenMoveEnv`, `RaisingStepEnv` | `test_error_handling` |
| `TrapPlanner` | `FakeTrapPlanner` | `test_error_handling` |
| `alfred_parser` functions | Module-level function replacement via inline monkey-patch | `test_error_handling` |

**What NOT to Mock:**
- Pure logic functions (`check_task_complete`, `build_eb_history_context`, `build_oracle_history_context`, `detect_dead_loop`, etc.) — tested directly with real inputs
- JSON parsing — `parse_json_response()` tested inline via fake API responses
- Injection helper functions (`_occlude_object`, `_swap_object`) — tested via `FakeController` step call assertions

## Fixtures and Factories

**Test Data helpers (module-level functions):**

```python
# test_task_conditions_alfred.py
def make_obj(object_id, object_type, *, pickupable=False, receptacle=False,
             toggleable=False, visible=True, is_toggled=False,
             receptacle_ids=None, parent_ids=None):
    return {"objectId": object_id, "objectType": object_type, ...}

# test_context_builder.py
def make_step(idx, branch="main", success=True, **extra):
    step = {"step_id": f"s{idx}", "branch_id": branch,
            "step_index_in_branch": idx, "action": "MoveAhead", ...}
    step.update(extra)
    return step

# test_error_handling.py
def _obj(object_id, object_type, **extra):
    data = {"objectId": object_id, "objectType": object_type, ...}
    data.update(extra)
    return data
```

**No fixtures directory or JSON fixture files** — all data constructed inline in test functions.

## Coverage

**Requirements:** None enforced. No coverage tool configured. No CI pipeline. No `.coveragerc` or `pyproject.toml` coverage config.

## Test Types

**Unit Tests (8 files, 31 total tests):**
- `test_task_conditions_alfred.py` (5 tests): Each checker function tested with mock metadata — covers all 7 task types via aliases
- `test_context_builder.py` (5 tests): History rendering, oracle field inclusion/exclusion, branch history ordering, pending step inclusion
- `test_vlm_client.py` (1 test): Multi-image message structure verification (text order, image_url type)
- `test_env_injector.py` (3 tests): `_occlude_object`, `_swap_object` execute correct AI2-THOR actions; declared injection methods via `inject()`
- `test_alfred_scene.py` (4 tests): Scene restoration order, pose merging (merge ALFRED poses with current AI2-THOR objects), init action format, ALFRED state tracking side effects
- `test_eb_agent_prompts.py` (2 tests): Prompt template content verification (hand status, task criteria in lookaround and Phase 3)
- `test_training_prompt.py` (2 tests): Training input format (oracle field exclusion, parameter stripping), reasoning prompt structure (no pipe format)
- `test_error_handling.py` (9 tests): BranchRunner error paths — invalid action retry, object resolution retry, Done rejection, LookAround repeat rejection, environment failure recording, env step exception handling, injection retry, initial trap failure

**Scope of each test file:**
- Pure logic tests: `test_task_conditions_alfred`, `test_context_builder`, `test_eb_agent_prompts`, `test_training_prompt`
- Fake AI2-THOR tests: `test_alfred_scene`, `test_env_injector` (use `FakeController` to verify step call order)
- BranchRunner integration tests: `test_error_handling` (verify JSON output, failure logs, retry behavior, step indexing)
- Client test: `test_vlm_client` (verify message formatting only, no real API call)

**Integration Tests (manual, not automated):**
- `scripts/e2e_test.py` — runs the full Phase 1-4 pipeline against real AI2-THOR + VLM API
  - `--task TASK_TYPE` for single task type
  - `--all` for all 7 types
  - `--random` for random episode selection
  - `--parallel N` for concurrent execution (ThreadPoolExecutor)
  - `--no-traps` to skip trap injection
  - `--output DIR` for custom output directory
  - Default output: timestamped `output_e2e_YYYYMMDD_HHMMSS/`

**Pipeline/Batch Testing:**
- `scripts/run_pipeline.py` — batch processing wrapper around `run_single_branch()`
  - `--max N` to limit episodes
  - `--task TYPE` to filter by task type
  - `--parallel N` for concurrency
  - `--no-fork` to disable fork branching

**Other Scripts:**
- `scripts/bench_api.py` — VLM API latency benchmarking
- `scripts/generate_ft_data.py` — generate fine-tuning data from episode outputs
- `scripts/resume_episode.py` — resume failed episodes from saved state

## Common Patterns

**Fake AI2-THOR Controller pattern (`test_alfred_scene.py`, `test_env_injector.py`):**
```python
class FakeController:
    def __init__(self, metadata=None):
        self.calls = []  # Record all step() calls for assertion
        self.metadata = metadata or {"lastActionSuccess": True, "objects": []}

    def step(self, action=None, **params):
        call = dict(params)
        if action is not None:
            call["action"] = action
        self.calls.append(call)
        return FakeEvent(self.metadata)
```

**BranchRunner error handling test pattern (`test_error_handling.py`):**
```python
def test_specific_error_path(self):
    def _episode(self, output_dir):
        return EpisodeManager("ep", output_dir, {...metadata...})

    with tempfile.TemporaryDirectory() as tmp:
        ep = self._episode(tmp)                         # Create EpisodeManager with metadata
        env = FailingMoveEnv()                          # Fake environment
        agent = SomeFakeAgent()                         # Fake EB agent
        runner = BranchRunner(agent, NoopOracle(), tmp, enable_phase2=False, enable_fork=False)

        result = runner.run(BranchConfig("ep", "main", None), env, ep)

        # Assert termination condition
        self.assertEqual("expected_reason", result.termination_reason)
        # Assert JSON output was written correctly
        with open(ep.file_path) as f:
            data = json.load(f)
        self.assertEqual("expected_action", data["steps"][0]["action"])
        # Assert failure log was written
        with open(f"{tmp}/ep/failures_main.jsonl") as f:
            log = f.read()
        self.assertIn("expected_failure_type", log)
```

**Prompt content testing pattern (`test_eb_agent_prompts.py`):**
```python
class CapturingClient:
    def __init__(self):
        self.user_text = None

    def chat_with_image_json(self, *, user_text, **kwargs):
        self.user_text = user_text
        return default_response

class SomePromptTest(unittest.TestCase):
    def test_prompt_contains_required_section(self):
        client = CapturingClient()
        agent = EBAgent(client)
        agent.some_method(...)
        self.assertIn("EXPECTED TEXT", client.user_text)
        self.assertNotIn("FORBIDDEN TEXT", client.user_text)
```

**Inline monkey-patching for module-level overrides (`test_error_handling.py`):**
```python
def test_failed_injection_asks_oracle_again_without_consuming_quota(self):
    def fake_inject(controller, method, **params):
        return {"success": False, "error": "bad injection setup"}

    original_inject = branch_runner_module.inject
    branch_runner_module.inject = fake_inject
    try:
        ...
    finally:
        branch_runner_module.inject = original_inject
```

## Test Coverage Gaps (Ranked by Impact)

1. **`egocentric_memory.py` has zero tests** (397 lines) — spatial memory logic (object accumulation, duplicate-type numbering, aging, area labeling, obstacle tracking, error compression) is the single largest untested pure-logic module. Errors here affect all prompt context across all phases.

2. **`action_adapter.py` has zero tests** (123 lines) — `resolve_object_ids()` is called every action step. `objectType`→`objectId` resolution logic (visible/pickupable prioritization, receptacle-only matching, fallback chains) directly determines whether EB actions succeed or fail.

3. **`trap_planner.py` has zero tests** (160 lines) — trap selection, `_find_object()` placeholder resolution, `exclude_types` filtering, `breakable` filter, Blinds blocklist. Critical for Phase 2 injection quality.

4. **`alfred_parser.py` has zero tests** (148 lines) — `load_traj()`, `extract_metadata()`, `extract_low_actions()` parse real ALFRED JSON. Format changes in ALFRED data would break the pipeline silently.

5. **`episode_manager.py` has no dedicated tests** (82 lines) — JSON serialization/deserialization tested only as side effect of `test_error_handling`.

6. **`step_recorder.py` has no tests** (53 lines) — `build_step()` entry structure, `save_frame()` image writing.

7. **`executor.py` deterministic logic is untested** (149 lines) — `execute_intent()` prompt building is pure data manipulation; only the VLM call prevents unit testing the assembly. `_sanitize_actions()` is a pure function that caps excessive actions.

8. **`branch_runner.py` has incomplete test coverage** (1486 lines) — only tested through `test_error_handling.py` specific error paths. The normal Phase 1-4 loop with ExecutorAgent path (lines 210-358) has no unit test coverage. `replay_steps()` is tested only for raising behavior, not correctness. `MoveSequence` execution (`_execute_move_sequence()`) has no dedicated test.

9. **`oracle_agent.py` prompt building is untested** — `build_phase2_prompt()` and `build_phase4_prompt()` handle conditional assembly (visible objects, injection errors, fork mode) but have no prompt content tests.

10. **`eb_agent.py` prompt building has incomplete tests** — `plan_intent()` prompt (intent history, `scan room` handling) and `review_actions()` prompt (executor sequence display, STUCK detection) have no tests.

11. **Heat/cool/clean tasks insufficiently E2E tested** — the E2E test runs one episode per type, but only `pick_and_place_simple`, `look_at_obj_in_light`, and `pick_and_place_with_movable_recep` have been exercised frequently.

12. **MoveSequence partial execution untested** — `_execute_move_sequence()` logic for partial success reporting, meta-action truncation, and objectId tracking in failure is not tested in isolation.

13. **`_fix_visible_bounds()` in `env_controller.py` has no test** (12 lines) — AI2-THOR 5.0.0 bug workaround. If the bug is fixed upstream, this workaround may become incorrect or harmful.

14. **No mutation/stress testing** — all error path tests use deterministic fake environments. Real failure patterns (concurrent VLM timeouts, partial MoveSequence execution mixed with environment failures) are not tested.

15. **No parallel execution tests** — `scheduler.py` uses `ThreadPoolExecutor` for parallel branches, but `EnvController` has module-level Xvfb singleton state (`_xvfb_proc` class variable) that may not be thread-safe.

## Testable Modules (No VLM or AI2-THOR Required)

| Module | File | Lines | Testable Logic | Test File |
|---|---|---|---|---|
| `task_conditions` | `src/task_conditions.py` | 323 | All 7 `check_*` functions, `detect_dead_loop`, `check_unrecoverable`, `get_completion_criteria_text` | `test_task_conditions_alfred.py` (5 tests) |
| `context_builder` | `src/context_builder.py` | 126 | History rendering, oracle field inclusion, branch history ordering, `_render_step()`, `_public_oracle_key()` | `test_context_builder.py` (5 tests) |
| `action_adapter` | `src/action_adapter.py` | 123 | `adapt()`, `resolve_object_ids()`, parameter cleaning, `clean_history_params()` | None |
| `alfred_scene` | `src/alfred_scene.py` | 192 | Scene restoration, pose merging, state tracking | `test_alfred_scene.py` (4 tests) |
| `env_injector` | `src/env_injector.py` | ~217 | Injection methods, task-critical guards | `test_env_injector.py` (3 tests, partial) |
| `step_recorder` | `src/step_recorder.py` | 56 | Step building, frame saving | None |
| `trap_planner` | `src/trap_planner.py` | ~160 | Trap selection, `_find_object()`, `_resolve()` | None |
| `episode_manager` | `src/episode_manager.py` | 82 | JSON serialization/deserialization | None (tested via `test_error_handling`) |
| `alfred_parser` | `src/alfred_parser.py` | ~148 | JSON trajectory parsing | None |
| `egocentric_memory` | `src/egocentric_memory.py` | 397 | Object accumulation, numbering, aging, area labeling, obstacle tracking, error compression | None |
| `executor` | `src/executor.py` | 214 | `_sanitize_actions()` (pure function), prompt building assembly | None |
| `branch_runner` | `src/branch_runner.py` | 1486 | `replay_steps()`, `_extract_task_target_types()`, `_infer_pddl_params()`, `_format_seq()`, `_make_result()`, `_build_fork_task()` | `test_error_handling.py` (runner integration) |

---

*Testing analysis: 2026-06-12*
