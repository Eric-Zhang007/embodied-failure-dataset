# Concerns & Technical Debt

**Analysis Date:** 2026-06-16  
**Analyzed:** 18 source modules, 10 scripts, 8 test files, 1 data file

---

## HIGH SEVERITY

### 1. Bug: Unreachable double-return in `EBAgent.analyze_scan_room()`
**File:** `src/eb_agent.py`, lines 725-726  
```python
        return result
        return result  # UNREACHABLE — dead code, second return never executes
```
The second `return result` on line 726 is unreachable dead code. While harmless at runtime, it indicates a copy-paste or merge error. Remove line 726.

### 2. Ollama `_chat_ollama` retry loop never raises on exhaustion
**File:** `src/vlm_client.py`, lines 449-458  
The retry loop captures status codes 200 and non-200 but calls `resp.raise_for_status()` at line 458 *outside* the loop, after three failed attempts. If all three attempts return a non-200 status, `raise_for_status()` fires. However, this is implicit behavior — a `RuntimeError` or explicit message would be clearer. Also, there is no elapsed time tracking, no logging, and no exponential backoff (uses `time.sleep(2 ** attempt)` but no progressive timeout increase).

### 3. API keys exposed via CLI arguments — visible in process lists
**Files:** Multiple scripts: `scripts/run_pipeline.py:26`, `scripts/e2e_test.py:65`, `scripts/e2e_test_output.py:34`, `scripts/generate_ft_data.py:109`, `scripts/bench_api.py:100`  
All scripts require `--api-key` as a CLI argument, making the key visible in `/proc/*/cmdline` and `ps aux` output on Linux. While `run_pipeline.py` has a default of `""` (empty string), others use `required=True`. Consider using environment variables (`SILICONFLOW_API_KEY`) or `.env` files instead. `SchedulerConfig` already stores the key as an attribute, but it originates from a CLI arg.

### 4. Undeclared runtime dependencies — will fail on fresh install
**File:** `pyproject.toml`  
Only `ai2thor==5.0.0` is declared. The following are imported but NOT declared:
- `numpy` (used in 11 source files)
- `Pillow` / `PIL` (used in `src/vlm_client.py`, `src/step_recorder.py`, scripts)
- `requests` (used in `src/vlm_client.py`)
- `openai` SDK may be needed for OpenAI-compatible backend
- `concurrent.futures` (used in `src/scheduler.py`, scripts — stdlib, OK)
A `pip install` from `pyproject.toml` alone will crash at first import. These packages likely work because they're installed globally or come as transitive deps of ai2thor, but they are NOT guaranteed.

### 5. Missing test coverage for ~10/18 source modules
Only 8 test files exist for 18 source modules. Untested modules include:
- `action_adapter.py` — resolution and adaptation logic (no tests)
- `alfred_parser.py` — trajectory parsing, action string parsing (no tests)
- `egocentric_memory.py` — 410 lines of spatial memory logic (no tests)
- `env_controller.py` — Xvfb management, scene reset (no tests)
- `episode_manager.py` — JSON serialization/deserialization (no tests)
- `executor.py` — Executor agent (no tests)
- `fork_manager.py` — fork reasoning rewriting (no tests)
- `scheduler.py` — thread pool, fork queue (no tests)
- `step_recorder.py` — screenshot capture (no tests)
- `trap_planner.py` — trap selection and application (no tests)

### 6. `branch_runner.py` is 1503 lines (75KB) — monolithic and fragile
**File:** `src/branch_runner.py`  
This file contains the main loop, MoveSequence execution, replay, fork building, finalization, and shared utility functions (`run_single_branch`, `replay_steps`, `_require_alfred_scene`, `_infer_pddl_params`). The `run()` method alone spans ~900 lines with deeply nested conditionals for:
- Planner→Executor→Review cycle
- Single-agent fallback path
- LookAround handling (two different code paths: initial scan and mid-run scan)
- Phase 2 injection with retry
- Phase 3+4 failure handling
- Recovery execution
- Done detection (multiple paths)

This single file is a maintainability risk. Consider splitting into:
- `src/branch_runner.py` — core `BranchRunner` class with the main loop
- `src/sequence_executor.py` — `_execute_move_sequence()` logic
- `src/replay.py` — `replay_steps()`, `run_single_branch()`, `_require_alfred_scene()`
- `src/pddl_inference.py` — `_infer_pddl_params()`

---

## MEDIUM SEVERITY

### 7. Empty `except` blocks followed by `pass` — silent error suppression
**File:** `src/vlm_client.py`, lines 520, 532, 542, 554; `src/alfred_parser.py`, line 185; `src/branch_runner.py`, line 210; `src/env_controller.py`, line 108; `tests/test_error_handling.py`, line 317

In `parse_json_response()` (vlm_client.py:519-520, 531-532, 541-542, 553-554), the multi-stage JSON parsing intentionally catches `json.JSONDecodeError` with `pass` to fall through to the next repair stage. This is a known, acceptable pattern for the JSON repair pipeline. However:
- `src/alfred_parser.py:185` — `except ValueError: pass` in `_parse_action_string` silently drops unconvertible numeric values
- `src/branch_runner.py:210` — `pass` in the `if step_index == 1 and proposed_action:` branch (empty block; should have a comment explaining intent)
- `src/env_controller.py:108` — `except (CalledProcessError, FileNotFoundError): pass` when probing Xvfb displays; silent failure is acceptable here but should log a debug message

### 8. Hardcoded data paths that assume a specific directory layout
- `src/scheduler.py:29` — `data_dir: str = "data/json_2.1.0"` (hardcoded ALFRED version directory)
- `src/trap_planner.py:19-22` — constructs path to `data/failure_type_library.json` relative to `__file__`
- `src/vlm_client.py:22` — `LOG_DIR = Path("logs")` (hardcoded relative path; created on import)
- `scripts/e2e_test.py:71` — `--data-dir` defaults to `"data/json_2.1.0"`
- `scripts/run_pipeline.py:19` — `--data-dir` defaults to `"data/json_2.1.0"`

These paths are relative to CWD and will break if the script is run from a different directory.

### 9. Hardcoded API URLs — no configuration mechanism
**File:** `src/vlm_client.py`  
- `openrouter()` → `https://openrouter.ai/api/v1` (line 93)
- `siliconflow()` → `https://api.siliconflow.cn/v1` (line 104)
- `openai()` → `https://api.openai.com/v1` (line 98, only if `base_url` not provided)
These endpoints are baked into the codebase and cannot be changed without modifying source.

### 10. Magic numbers scattered throughout the codebase
- **Step limit:** `step_index >= 200` (branch_runner.py:177) — hardcoded max steps per branch
- **Max tokens:** Varies widely: `1024`, `2048`, `16384`, `256`, `512`, `50`, `150`, `200`, `5` — no consistent policy
- **Timeouts:** `300` seconds for OpenAI API (vlm_client.py:325), `40` seconds for Ollama (vlm_client.py:450), with progressive increase `timeout += 100` on retry (vlm_client.py:367)
- **Retries:** 3 attempts everywhere (API calls, injection attempts, nonexecuted retries)
- **Cascade levels:** `cascade_level <= 1`, `cascade_level >= 2` (branch_runner.py)
- **Phase 2 injection limit:** `phase2_max_injections = 3` (branch_runner.py:110)
- **Dead loop detection:** `window=10, threshold=5` (task_conditions.py:58), plus a weak-signal check of 3 failures in last 8 steps
- **Repetition caps:** `repeat = min(repeat, 200)` (branch_runner.py:1117), actions capped at 20 (executor.py:168)
- **Aging threshold:** `AGING_THRESHOLD: int = 20` (egocentric_memory.py:69)
- **Path history:** `len(self._agent_path) > 20` (egocentric_memory.py:103)
- **Memory render cap:** `len(result) > 1500` (egocentric_memory.py:224)
- **Camera horizon:** `max +30° up / -60° down` (eb_agent.py:41-42)

Consider extracting these to a central `src/config.py` or `constants.py` module.

### 11. No `.gitignore` for sensitive or generated files
The following are likely generated at runtime but no `.gitignore` was found:
- `logs/` directory (API call logs, failure dumps)
- `output*/` directories (pipeline output with episode JSONs and screenshots)
- `__pycache__/` directories
- `*.png` screenshots within output directories

### 12. `_SOLID_TYPES` redefined locally inside Executor method
**File:** `src/executor.py`, lines 118-120  
The set `_SOLID_TYPES` is defined as a local variable inside `execute_intent()` on every call. It should be a module-level constant. It's also a duplicate of similar logic that may exist in `eb_agent.py` — the same object classification should be centralized.

### 13. Fork manager sends a dummy black image to the VLM
**File:** `src/fork_manager.py`, line 52  
```python
image=np.zeros((100, 100, 3), dtype=np.uint8)
```
`_rewrite_reasoning()` calls `chat_with_image()` with a 100×100 black image because the API requires an image but the task is text-only rewriting. This wastes tokens and may confuse vision models. Consider adding a `chat_text_json` call or refactoring.

### 14. `_last_scan_step` initialized to `-100` — magic sentinel
**File:** `src/branch_runner.py`, line 73  
```python
self._last_scan_step = -100
```
Used to allow the first scan at step 0 (since `step_index - (-100) > 3` is always true). A cleaner approach would be `None` or a constant like `SCAN_COOLDOWN = 3`.

### 15. `trap_planner is not False` in `run_single_branch`
**File:** `src/branch_runner.py`, line 1493  
```python
enable_phase2=(trap_planner is not False and trap_planner is not None),
```
This double-check suggests a design confusion — `trap_planner` should be either `None` (disabled) or a `TrapPlanner` instance (enabled). The `is not False` check implies some caller may pass `False`, which is inconsistent API design.

---

## LOW SEVERITY

### 16. Missing docstrings in multiple functions
Functions lacking docstrings (non-exhaustive):
- `eb_agent.py`: `_egocentric_angle()`, `_is_in_front()`, `_direction()`
- `context_builder.py`: `_build_history_context()`, `_render_step()`, `_public_oracle_key()`
- `action_adapter.py`: `_adapt_teleport_full()`, `_adapt_put_object()`
- `alfred_scene.py`: `object_by_id()`, `find_closest_object_of_type()`, `_object_name()`, `_apply_object_toggles()`, `_apply_dirty_and_empty()`, `_merge_object_poses_for_current_scene()`
- `env_injector.py`: all internal `_*` functions lack docstrings
- `task_conditions.py`: `_state_then_place()`, `_targets()`, `_objects_with_name_and_prop()`, etc.
- `branch_runner.py`: `_execute_move_sequence()`, `_finalize()`, `_build_fork_task()`, `_format_seq()`, `_invalid_action_message()`

### 17. Mixed language in comments (Chinese and English)
Most docstrings and comments are in Chinese (e.g., `src/vlm_client.py`, `src/eb_agent.py`, `src/branch_runner.py`, `src/scheduler.py`), while function names, variable names, and error messages are in English. Some files like `context_builder.py` and `action_adapter.py` are exclusively English. This is acceptable for a Chinese-speaking team but limits community contributions.

### 18. `alfred_parser.py:185` — `except ValueError: pass` in numeric parsing
**File:** `src/alfred_parser.py`, lines 180-185  
```python
try:
    val_int = int(val)
    val = val_int
except ValueError:
    try:
        val_float = float(val)
        val = val_float
    except ValueError:
        pass  # keeps the original string value
```
The intention (keep string if not numeric) is fine, but the nested try/except with bare `pass` is hard to read. Refactor to a helper `_coerce_number()`.

### 19. `EpisodeManager.load()` uses `__new__` bypassing `__init__`
**File:** `src/episode_manager.py`, lines 73-82  
```python
mgr = EpisodeManager.__new__(EpisodeManager)
mgr.episode_id = data["episode_id"]
mgr.file_path = file_path
mgr.data = data
return mgr
```
Using `__new__` to bypass `__init__` is fragile — if `__init__` ever gains important side effects, loaded instances will be inconsistent. A `@classmethod` factory that constructs normally then overwrites would be more robust.

### 20. `generate_ft_data.py` swallows exceptions in worker threads
**File:** `scripts/generate_ft_data.py`, line 148  
The `process_episode` function can raise `FileNotFoundError` (line 64) or `ValueError` (line 97), but the worker pool in `main()` calls `.result()` without try/except. If an episode fails midway, the entire script crashes rather than recording the error and continuing.

### 21. `scripts/e2e_test_output.py` references a local variable `LOCAL_EB_URL` that may not exist
**File:** `scripts/e2e_test_output.py`, line 75  
Uses `LOCAL_EB_URL` which is likely a module-level constant not shown in the truncated read. If undefined at runtime, this causes a `NameError`.

### 22. No type stubs or mypy configuration
No `mypy.ini`, `pyrightconfig.json`, or type-checking configuration exists. While many functions have type hints, the codebase would benefit from CI-enforced static analysis.

### 23. `np.ndarray | str` type hint uses Python 3.10+ union syntax
**File:** Multiple files (e.g., `src/vlm_client.py:193`, `src/eb_agent.py`)  
The `X | Y` union syntax requires Python 3.10+. `pyproject.toml` declares `requires-python = ">=3.10"` which is correct, but if anyone tries to run on 3.9, these will break with `TypeError`.

### 24. No logging in `env_injector.py` for blocked injections
When `_guard_task_critical()` blocks an injection (lines 27-46), the function silently returns the block reason. There's no logging to indicate that an Oracle-proposed injection was rejected, which could make debugging injection behaviors difficult.

---

## SECURITY CONCERNS

### S1. API keys in CLI arguments
As noted above (HIGH #3), API keys passed via `--api-key` are visible to any user on the system via `ps aux` or `/proc`.

### S2. API keys written to log files
**File:** `src/vlm_client.py`, lines 396-411  
`_write_api_exchange()` logs the full request body (including `Authorization: Bearer <key>` header) to `api_calls.jsonl`. While `_sanitize_for_log()` filters image data, it does NOT strip the API key from request bodies or logged headers. Any access to the `logs/` directory exposes API credentials.

### S3. Subprocess calls with user-controlled input
**File:** `scripts/download_alfred.py`, lines 31, 37  
Calls `subprocess.check_call(["wget", URL, ...])` and `subprocess.check_call(["7z", "x", ARCHIVE, ...])` with hardcoded URLs (safe), but if these were ever parameterized from user input, they would be injection vectors.

### S4. Xvfb subprocess managed via class-level state
**File:** `src/env_controller.py`, lines 87-124  
`_start_xvfb()` spawns a `subprocess.Popen` for Xvfb and registers `atexit` cleanup. If multiple `EnvController` instances are created in different threads, the class-level state (`_xvfb_proc`, `_xvfb_display`) could race. The `atexit` handler references the class variable, which may have been overwritten by another instance.

---

## IMPROVEMENT OPPORTUNITIES

### IO1. Centralized configuration
Create `src/config.py` with dataclass-based configuration replacing scattered magic numbers, hardcoded paths, and defaults. This would also allow YAML/TOML config files.

### IO2. Add a `--env-file` or `.env` support
Use `python-dotenv` or read environment variables for API keys instead of CLI args. The `VLMClient` already supports `os.environ` for `OPENROUTER_API_KEY` and `OPENAI_API_KEY`, but `siliconflow()` has no env-var fallback.

### IO3. Add structured error types
Replace string-based error messages with a hierarchy of exception classes (e.g., `InjectionError`, `EpisodeError`, `VLMError`, `ResolutionError`) for better programmatic error handling.

### IO4. Add integration tests that mock VLM responses
Current tests use `unittest.TestCase` with mock agents and fake environments. Adding `pytest` with fixtures and `responses`/`httpx`-based HTTP mocking for the VLM client would catch regressions in the JSON retry logic and API error handling.

### IO5. Add CI pipeline
No CI configuration (`.github/workflows/`, etc.) exists. Adding linting (ruff/flake8), type checking (mypy), and test running would prevent regressions.

### IO6. Add a `README.md`
No README was found in the project root. The `docs/codebase/` directory has STRUCTURE.md, ARCHITECTURE.md, STACK.md, CONVENTIONS.md, INTEGRATIONS.md, and TESTING.md — but no top-level README for quickstart instructions.

---

## SUMMARY COUNTS

| Category | Count |
|----------|-------|
| High severity issues | 6 |
| Medium severity issues | 9 |
| Low severity issues | 9 |
| Security concerns | 4 |
| Improvement opportunities | 6 |
| Empty except blocks | 8 |
| Missing test modules | 10 |
| Hardcoded paths | 5 |
| Magic numbers | 15+ |
