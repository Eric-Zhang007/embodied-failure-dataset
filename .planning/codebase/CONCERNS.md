# Codebase Concerns

**Analysis Date:** 2026-07-15

## Tech Debt

### Zero Automated Tests — RESOLVED (2026-07)

- **Resolution:** 58 unit tests across 8 test files runnable via `unittest discover`. Core modules tested: vlm_client (14), scheduler (9), error_handling (13), task_conditions (5), eb_agent_prompts (2), context_builder (5), alfred_scene (5), env_injector (3), training_prompt (2).
- Still missing: No CI pipeline. No `pytest` integration (uses unittest).

### API Key Hardcoded in CLAUDE.md

- **Issue:** A valid Siliconflow API key (`sk-umtqhyponkhgyhlbwyykumcqebzqxpkcheexbhxgcybsgypa`) is embedded in `CLAUDE.md` (line 18), which is checked into git. The `.gitignore` does not exclude `CLAUDE.md`.
- **Files:** `CLAUDE.md:18`
- **Impact:** Secret permanently exposed in version control. Anyone with repo access can use the key, incurring cost.
- **Fix approach:** Rotate the key. Remove it from `CLAUDE.md`. Use environment variable (`SILICONFLOW_API_KEY`) read at runtime. Reference the env var name in `CLAUDE.md` instead.

### No Type Checking (mypy/pyright)

- **Issue:** No `mypy.ini`, `pyproject.toml` mypy config, or `pyrightconfig.json`. Type annotations exist but are unchecked. Many functions use `list[dict]` instead of typed dicts, `dict` return types without schema.
- **Files:** All `src/*.py` — pervasive `list[dict]` and bare `dict` return types
- **Impact:** Type errors that could be caught statically at dev time instead surface at runtime during expensive VLM API calls.
- **Fix approach:** Add `mypy` to dependencies, create `pyproject.toml` mypy config, add type stubs for ai2thor.

### BranchRunner.run() is Monolithic (~850 Lines)

- **Issue:** The `BranchRunner.run()` method (`branch_runner.py:78-932`) is ~850 lines containing all Phase 1-4 orchestration, error handling, recovery, fork creation, and edge case handling in one deeply nested while loop.
- **Files:** `src/branch_runner.py:78-932`
- **Impact:** Hard to test, hard to reason about, easy to introduce subtle bugs. No separation of concerns — VLM calls, env interaction, disk I/O, and state management are mixed.
- **Fix approach:** Extract Phase 1 (Planner-Executor-Review cycle), Phase 2 (Oracle injection), Phase 3+4 (diagnosis + evaluation), and recovery execution into separate methods or classes.

### Serial Fork Processing Despite max_parallel

- **Issue:** `scheduler.py:147` processes fork branches serially (`while self.fork_queue: task = self.fork_queue.popleft()`), even though `ThreadPoolExecutor` is used for main branches. `max_parallel` only applies to main branches.
- **Files:** `src/scheduler.py:147-153`
- **Impact:** Fork branches are a bottleneck — a pipeline with many forks runs linearly regardless of parallelism budget.
- **Fix approach:** Use a thread pool for forks too, or document the constraint.

### Fork Mechanism Never End-to-End Tested

- **Issue:** Fork code exists across `scheduler.py`, `fork_manager.py`, `branch_runner.py:920-930`, but `enable_fork` defaults to `False` in `run_single_branch()`. The `_rewrite_reasoning` method in `fork_manager.py:52` sends a dummy blank image (`np.zeros((100, 100, 3), dtype=np.uint8)`) to the VLM, which could produce unreliable rewritten reasoning.
- **Files:** `src/branch_runner.py:920-930`, `src/scheduler.py:199-295`, `src/fork_manager.py:19-56`
- **Impact:** Fork code may have bugs that only surface when enabled in production. Rewritten reasoning may be incoherent without visual context.
- **Fix approach:** Replace dummy blank image with actual fork point frame. E2E test with forks enabled.

### heat/cool/clean Task Types Never E2E Tested

- **Issue:** Confirmed in `CLAUDE.md`. The 3 state-based task types (`pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`, `pick_clean_then_place_in_recep`) rely on AI2-THOR's `task_state` mechanism via `update_alfred_task_state()` in `alfred_scene.py`, which has additional complexity.
- **Files:** `src/task_conditions.py:219-251`, `src/env_controller.py:28,37,51`, `src/alfred_scene.py`
- **Impact:** These task types may have hidden bugs in state tracking or completion checking.
- **Fix approach:** E2E tests for each of heat/cool/clean task types.

### Stage 0 Not Implemented

- **Issue:** `failure_type_library.json` with 8 failure types is hand-crafted, not programmatically generated from AI2-THOR API documentation as originally planned.
- **Impact:** Manual curation limits scalability and consistency.
- **Fix approach:** Implement automated failure type classification from collected trajectories.

### No CI/CD Pipeline

- **Issue:** No `.github/workflows/` directory. No GitHub Actions, no automated checks on push/PR.
- **Files:** (missing) `.github/workflows/*.yml`
- **Impact:** Code quality depends entirely on manual review. Regression risk is high.
- **Fix approach:** Set up GitHub Actions with pytest, mypy, and ruff.

### executor.py Inline import logging

- **Issue:** `executor.py` has `import logging` inline inside `_sanitize_actions()` (lines 180, 196) instead of at module level — an unusual lazy-import pattern.
- **Files:** `src/executor.py:180,196`
- **Fix approach:** Move `import logging` to the top of the file.

### Unused _is_in_front Function

- **Issue:** `eb_agent.py:93-95` defines `_is_in_front(angle)` which is never called anywhere.
- **Files:** `src/eb_agent.py:93-95`
- **Fix approach:** Remove dead code.

### planner_feedback Parameter Never Passed

- **Issue:** `executor.py:74` `execute_intent()` accepts `planner_feedback` parameter, but `branch_runner.py` never passes it (always `None`).
- **Files:** `src/executor.py:74`, `src/branch_runner.py:322-335`
- **Fix approach:** Either remove the parameter or pass feedback from the review step.

## Known Bugs

### Phase 3 Planner Hallucinates Task Completion

- **Symptoms:** When a navigation action fails (e.g., agent walks into a cabinet), Phase 3 Planner sometimes declares the task already done — e.g., "knife already in sink" — when the knife is still on the counter.
- **Files:** `src/branch_runner.py:668-679`, `src/eb_agent.py:728-760`
- **Trigger:** Phase 3 diagnosis after an environment failure against a visible obstacle. The Planner confuses "I can see the target" with "the target is in place."
- **Workaround:** None. The model tends to rationalize failures as non-failures.
- **Fix approach:** Forcibly suppress task-completion logic in the Phase 3 prompt. Add an explicit instruction: "The task is NOT complete — do NOT claim it is. Diagnose only why the action failed."

### 8B Executor Under-counts Repeat Values

- **Symptoms:** For a 1.3m distance (10.4 steps at 0.125m/gridSize), the Executor outputs `"repeat": 8` instead of `"repeat": 11`. The prompt explicitly says "distance / 0.125, round up" but the 8B model cannot reliably perform this arithmetic.
- **Files:** `src/executor.py:17` (prompt: "0.6m = repeat 5. 1.0m = repeat 8. 1.8m = repeat 15."), `src/executor.py:150` (reminder: "distance / 0.125 = repeat count")
- **Trigger:** Any movement toward a distant target.
- **Fix approach:** Post-process Executor output to calculate repeat counts server-side from target position in spatial memory. Or switch Executor to 32B model.

### 8B Executor Ignores SOLID Obstacle Labels

- **Symptoms:** The Executor prompt explicitly marks large furniture as "SOLID - do NOT walk through" (`executor.py:125-127`), but the 8B model frequently outputs `MoveAhead` directly into cabinets/tables/desks.
- **Files:** `src/executor.py:116-118` (SOLID_TYPES list), `src/executor.py:125-127` (label appending)
- **Trigger:** Shortest-path reasoning overrides obstacle avoidance at the 8B level.
- **Fix approach:** Change the label to "IMPASSABLE — CANNOT MOVE THROUGH". Add pre-execution validation in `_execute_move_sequence` that checks for known obstacles in the path. Switch to 32B.

### Executor Sometimes Generates Empty Actions

- **Symptoms:** The 8B Executor sometimes returns `"actions": []` (empty list) when confused, causing `MoveSequence` to fail immediately: "MoveSequence has no steps."
- **Files:** `src/executor.py:166-169` (`_sanitize_actions` returns as-is without handling empty case), `src/branch_runner.py:1027-1031`
- **Trigger:** Ambiguous target, conflicting spatial cues, or model confusion.
- **Fix approach:** In `_sanitize_actions`, detect empty actions and return a fallback action (e.g., LookAround) with `status: "failed"`. Or handle empty actions at branch_runner level with a retry.

### scan-to-scan Loop at Startup

- **Symptoms:** The initial LookAround captures 4 views, `analyze_scan_room` runs, decides a direction, rotates the agent. But the very first Planner cycle on step 1 immediately outputs `intent: "scan room"` again, triggering another full scan. The agent spends 2-3 steps just scanning.
- **Files:** `src/branch_runner.py:155-173` (initial scan), `src/branch_runner.py:262-318` (scan room intent handling)
- **Trigger:** The Planner doesn't "remember" that a scan just happened.
- **Fix approach:** Inject a "you just scanned the room — do NOT scan again" message into the first Planner call's intent history.

### AI2-THOR TeleportFull Can Place Agent in Unnavigable Positions

- **Symptoms:** ALFRED's TeleportFull init may place the agent against furniture edges or in corners where all movement directions are blocked by Floor.
- **Files:** `src/env_controller.py:57-68`, `src/alfred_scene.py`, `CLAUDE.md` line 123
- **Workaround:** None — episodes can be immediately unrecoverable.
- **Fix approach:** Add an initial mobility check after TeleportFull. If blocked in all directions, attempt random rotations or small teleport shifts to find navigable space.

### MoveSequence repeat=200 Hard Cap

- **Issue:** `branch_runner.py:1106` caps repeat at 200 (`min(repeat, 200)`). A 200-step movement at 0.125m/gridSize is 25m, which is within range for large scenes. The cap truncates valid long moves silently.
- **Files:** `src/branch_runner.py:1106`
- **Trigger:** Long-distance travel across large rooms.
- **Fix approach:** Increase cap or make configurable. Log a warning when the cap is hit.

## Execution & Performance Concerns

### Three-VLM Round-Trip Per Step (~3-4 API Calls Per Cycle)

- **Issue:** Each successful planning cycle requires: (1) Planner `plan_intent` (32B), (2) Executor `execute_intent` (8B), (3) Planner `review_actions` (32B). Combined with Phase 2 Oracle (32B) and Phase 3+4 on failure (2 more calls), the total is ~3-4 VLM calls per step and ~6 on failure.
- **Files:** `src/branch_runner.py:218-350`
- **Impact:** High API cost and latency. Each 32B call takes 10-30s. A 20-step episode costs ~$0.50-1.00 in API fees.
- **Fix approach:** Remove the Planner Review step (rejection rate ~30% per CLAUDE.md but many rejections are for minor inefficiency that doesn't affect success). Or merge into single 32B model.

### 8B Executor Spatial Reasoning is a Bottleneck

- **Issue:** The documented design rationale is "8B for cost savings." But the 8B Executor fails at basic arithmetic (repeat counts), ignores obstacle labels, outputs empty action lists. The 32B model handles all these tasks reliably.
- **Files:** `src/executor.py` (entire file), `CLAUDE.md` lines 20 (model config)
- **Impact:** The 8B Executor causes ~50% of observed navigation failures. "Cost savings" may be illusory if failed episodes must be re-run.
- **Fix approach:** Benchmark 32B vs 8B on task completion rate and total cost per successful episode. The 32B may be cheaper end-to-end.

### No Systematic Exploration Strategy

- **Issue:** When the target location is unknown, both Planner and Executor have no structured exploration strategy. They rotate in place, move in one direction until blocked, then rotate — essentially a random walk.
- **Files:** `src/eb_agent.py:351-394` (Planner SYSTEM prompt), `src/executor.py:13-53` (Executor SYSTEM prompt — no exploration guidelines)
- **Impact:** Agent wastes steps on inefficient search, especially in large rooms.
- **Fix approach:** Add a BFS-style exploration pattern: pick a cardinal direction, move to the farthest navigable point, rotate 90 degrees, repeat. Implement canonical scan paths.

### Phase 2 Oracle Guard (cascade_level <= 1) Limits Cascading Failures

- **Issue:** The guard `cascade_level <= 1` at `branch_runner.py:447` means Phase 2 injections only happen at cascade_level 0 (no prior failure) or 1 (first failure). Once cascade_level >= 2, no more injections happen.
- **Files:** `src/branch_runner.py:447`
- **Impact:** The "cascade" in "cascading failure" is limited to 2 failures deep. Real cascading failures would have 3+ layers.
- **Fix approach:** Relax to `cascade_level <= 2` or make configurable.

### Phase 2 Oracle Injection Has No State Persistence

- **Issue:** The Oracle makes injection decisions independently each step. It receives `cascade_level` as context but has no memory of what it already injected. It may inject duplicate or incompatible methods on the same objects.
- **Files:** `src/oracle_agent.py:44-55` (guidelines say "each trap should be NOVEL" but no code enforcement)
- **Impact:** Duplicate or incompatible injection methods cause agent confusion that's hard to debug.
- **Fix approach:** Pass the list of previous injections to the Oracle so it can avoid duplicates.

### EgocentricMemory Lacks Semantic Spatial Relationships — RESOLVED (2026-07)

- **Resolution:** SemanticMemory module (`src/semantic_memory.py`) stores receptacle groups, freshness tracking, task progress. Objects grouped by parent receptacle. GeometricMemory preserved as `--memory geometric` fallback.
- Remaining concern: Planner-level integration not yet complete.

### Agent Path Buffer (20 entries) is Small

- **Issue:** `egocentric_memory.py:104` caps agent path at 20 entries. At one step per entry, this represents only ~20 steps of recent movement. Longer episodes lose path context.
- **Files:** `src/egocentric_memory.py:104`
- **Impact:** Limited path context for spatial reasoning.
- **Fix approach:** Increase buffer or make configurable.

### heap-growing eb_history in Prompts

- **Issue:** Full `eb_history` list grows unbounded across steps and is periodically included in VLM prompts (e.g., `branch_runner.py:222-223` passes `eb_history[-3:]` but the history itself accumulates all steps).
- **Files:** `src/branch_runner.py` (eb_history appends every step)
- **Impact:** Context window pressure for long episodes. May cause VLM to lose early context.
- **Fix approach:** When eb_history exceeds some limit, summarize older entries into compact format.

## Fragile Areas

### _fix_visible_bounds Workaround

- **Files:** `src/env_controller.py:57-68`
- **Why fragile:** Patches AI2-THOR 5.0.0 initialization order bug where `process_visible_bounds2D` runs before `instance_detections2D` is populated. Runs on EVERY `env.step()` and `get_state_snapshot()`. If AI2-THOR fixes this upstream, the workaround may conflict. If the bug manifests differently, the fix may not apply.
- **Safe modification:** Add AI2-THOR version detection and only apply for versions < 5.1.0.
- **Test coverage:** Zero — depends on visual inspection of object visibility behavior.

### Pluggable Executor / Single-Agent Fallback Path

- **Files:** `src/branch_runner.py:210,360-377`
- **Why fragile:** `BranchRunner` supports both `executor_agent` (Planner+Executor+Review) and fallback single-agent (`propose_action`) paths in the same method. Two different code paths (lines 210-359 vs 360-377) share the same state variables. A bug could silently mix the two.
- **Safe modification:** Remove the single-agent fallback path once the Executor approach is validated.
- **Test coverage:** None.

### Hardcoded Limits and Magic Numbers

- **Files:** Various
- **Fragile values (all hardcoded):** 200 step limit (`branch_runner.py:176`), 3 max injection attempts (`branch_runner.py:448`), 3 max Phase 2 injections (`branch_runner.py:109`), 3 nonexecuted retries (`branch_runner.py:111`), 20-step path buffer (`egocentric_memory.py:104`), 15 max remembered objects (`egocentric_memory.py:69`), 300/400/500 timeout escalation (`vlm_client.py:309,351`), 2 JSON retries (`vlm_client.py:242`), MAX_TOTAL=12 actions (`executor.py:170`), MAX_CONSECUTIVE=6 (`executor.py:171`), AGING_THRESHOLD=20 steps (`egocentric_memory.py:68`)
- **Safe modification:** Centralize tunable constants in a config class or dataclass. Pass through constructors.
- **Test coverage:** None.

### executor._sanitize_actions Cap (MAX_TOTAL=12, MAX_CONSECUTIVE=6)

- **Files:** `src/executor.py:170-171`
- **Why fragile:** MAX_TOTAL caps action sequence at 12, MAX_CONSECUTIVE at 6. Long movement sequences (traversing a long corridor requires 15-20 `MoveAhead`) are silently truncated. The cap changes `status` from "done" to "partial" which can affect Planner's next intent.
- **Safe modification:** Make caps configurable. Log when truncation occurs.
- **Test coverage:** None.

### Duplicated Blinds Blocklist

- **Files:** `src/env_injector.py:159` (`_CLOSE_BLOCKLIST`), `src/trap_planner.py` (inline blocklist)
- **Why fragile:** Blocklist exists in two files with duplicated logic. If updated in one but not the other, Blinds can slip through and cause hangs.
- **Fix approach:** Centralize in a shared constant.

## Security Considerations

### API Key in Version Control

- **Risk:** Siliconflow API key (`sk-umtqhyponkhgyhlbwyykumcqebzqxpkcheexbhxgcybsgypa`) in `CLAUDE.md:18`, tracked in git.
- **Files:** `CLAUDE.md:18`
- **Current mitigation:** None.
- **Recommendations:** Rotate the key NOW. Remove from git history (`git filter-branch` / BFG). Add `.env` to `.gitignore` and read key from environment variable.

### Full API Request/Response Logging

- **Risk:** `vlm_client.py:380-394` logs full request bodies and responses to `logs/api_calls.jsonl`. While `_sanitize_for_log` strips base64 images, response content is logged in full. `_dump_failure` (lines 362-378) writes the complete request body (including system prompts) to a file on any API error.
- **Files:** `src/vlm_client.py:25` (`LOG_FULL_API` default "1"), lines 362-394
- **Current mitigation:** `_sanitize_for_log` strips base64 image data. Authorization header not in body dict.
- **Recommendations:** Default `LOG_FULL_API` to `"0"`. Add response content truncation. Add request body truncation in `_dump_failure`.

## Dependencies at Risk

### AI2-THOR 5.0.0

- **Risk:** No Windows build (WSL-only). `_fix_visible_bounds` workaround suggests incomplete QA. `Blinds` timeout bug. No community support for 5.0.0-specific bugs.
- **Files:** `src/env_controller.py:57-68`, `src/env_injector.py:159`
- **Impact:** The entire project depends on AI2-THOR for simulation. An upstream update could break the workaround.
- **Migration plan:** Pin version to `==5.0.0` in pyproject.toml. Monitor AI2-THOR releases. Consider ABL as contingency.

## Test Coverage Gaps

### Core Logic Modules With Zero Tests

- **What's not tested:** `action_adapter.py` (124 lines), `episode_manager.py` (82 lines), `trap_planner.py` (160 lines), `alfred_parser.py` (148 lines), `context_builder.py` (125 lines), `step_recorder.py` (55 lines)
- **Files:** All files listed above
- **Risk:** These modules handle object-Id resolution, episode I/O, trap configuration, ALFRED JSON parsing, and context formatting — all critical path for every episode. A bug in any one corrupts output silently.
- **Priority:** High

### BranchRunner.run() Has No Tests

- **What's not tested:** The entire main loop (~850 lines of orchestration logic)
- **Files:** `src/branch_runner.py:78-932`
- **Risk:** The most complex and critical method. Every bug fix risks regression here.
- **Priority:** High

### EgocentricMemory Has No Tests

- **What's not tested:** All spatial memory logic (397 lines) — rendering, aging, obstacle tracking, deduplication numbering, area label inference.
- **Files:** `src/egocentric_memory.py` (397 lines)
- **Risk:** Broken memory degrades all subsequent VLM decisions.
- **Priority:** High

### Task Completion Checkers Have Limited Coverage

- **What's not tested:** 6 of 7 task types (only `pick_and_place_simple` E2E tested). Heat/cool/clean completion checkers untested.
- **Files:** `src/task_conditions.py:144-292`
- **Risk:** Heat/cool/clean tasks depend on `alfred_task_state` which may have edge-case bugs.
- **Priority:** Medium

### Fork Code Has Zero Coverage

- **What's not tested:** `scheduler._run_fork()`, `fork_manager._rewrite_reasoning()`, `branch_runner._build_fork_task()`, `replay_steps()` fork path
- **Files:** `src/scheduler.py:199-295`, `src/fork_manager.py`, `src/branch_runner.py:1188-1245,1283-1355`
- **Risk:** Fork branches may produce garbage data silently.
- **Priority:** Medium

### VLM Client JSON Parser Untested

- **What's not tested:** `parse_json_response()`, `_json_response_error()`, `_chat_json_with_retry()` — critical for Thinking model CoT extraction
- **Files:** `src/vlm_client.py:454-497`
- **Risk:** All VLM interactions fail if JSON parsing breaks. CoT extraction logic is heuristic (find `{`...`}` brace matching) — fragile against markdown fences or escaped JSON.
- **Priority:** High

---

*Concerns audit: 2026-06-12*
