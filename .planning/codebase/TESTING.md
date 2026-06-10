# TESTING.md — Testing Practices

## Current State

**There are no automated tests.** No test framework is configured, no test files exist, and `pyproject.toml` has no test dependencies.

## Test Infrastructure (Absent)

| Item | Status |
|------|--------|
| Test framework | Not configured |
| Test files | None exist |
| CI/CD | None |
| Coverage tooling | None |
| Linter/formatter | None |

## E2E Testing

**File**: `scripts/e2e_test.py`

A manual integration test that runs a single ALFRED trajectory through the full Phase 1-4 pipeline:

```bash
uv run python scripts/e2e_test.py --api-key sk-xxx
uv run python scripts/e2e_test.py --api-key sk-xxx --task pick_and_place_simple
uv run python scripts/e2e_test.py --api-key sk-xxx --no-traps
```

What it does:
1. Loads a trajectory from `data/json_2.1.0/`
2. Creates EB and Oracle agents with SiliconFlow client
3. Sets up TrapPlanner and calls `run_single_branch()`
4. Output goes to `output/` directory
5. Requires manual inspection of output JSON and screenshots to verify correctness

**File**: `scripts/e2e_test_output.py`

Same as `e2e_test.py` but accepts `--output-dir` for deterministic output placement.

## API Benchmarking

**File**: `scripts/bench_api.py`

Tests VLM API latency with configurable image count and size:

```bash
uv run python scripts/bench_api.py --api-key sk-xxx
```

Measures: time-to-first-token, total response time, tokens/sec for both single and multi-image calls.

## Pipeline Testing

**File**: `scripts/run_pipeline.py`

Batch processing for multiple trajectories:

```bash
uv run python scripts/run_pipeline.py --max 10 --api-key sk-xxx
uv run python scripts/run_pipeline.py --max 50 --no-fork --api-key sk-xxx
```

This is the closest thing to a regression test — run a batch and check success rates.

## What Can Be Tested Without VLM

The following modules are deterministic and testable:

| Module | Testable Logic |
|--------|---------------|
| `task_conditions.py` | All 7 `check_*` functions given mock metadata |
| `action_adapter.py` | `adapt()` and `resolve_object_ids()` given mock objects |
| `context_builder.py` | History rendering given mock steps |
| `alfred_parser.py` | JSON parsing given fixture files |
| `env_injector.py` | Injection logic (requires AI2-THOR, which has no mock) |
| `episode_manager.py` | JSON serialization/deserialization round-trip |
| `trap_planner.py` | Trap selection given mock scene objects |

## Testing Gaps (Ranked by Impact)

1. **E2E regression suite**: No automated way to verify pipeline correctness after changes
2. **Task completion logic**: `task_conditions.py` has no tests despite being the ground truth for Done
3. **JSON parse resilience**: `parse_json_response()` handles malformed VLM output — untested against real failure patterns
4. **Fork mechanism**: Code exists but `enable_fork=False` by default, has never been E2E tested
5. **Resume/replay**: `BranchRunner.resume()` is untested
6. **Injection guards**: `_guard_task_critical()` logic untested
7. **All prompt templates**: No validation that prompts contain correct action lists, required fields, or task criteria
