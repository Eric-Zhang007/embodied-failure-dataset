---
status: awaiting_human_verify
trigger: "Executor returns actions=[] with status=done, causing an empty MoveSequence Phase 3 recovery loop"
created: 2026-07-15
updated: 2026-07-15T00:10:00Z
---

# Debug Session: Executor Empty Action Loop

## Symptoms

### Expected Behavior
When the Executor reports `status: "done"` with no actions because the current intent target has already been reached, the runner should mark the current intent complete and return control to the Planner for the next intent. It should not enter Phase 3 failure diagnosis.

### Actual Behavior
Fresh checkpoint evidence reports that an empty MoveSequence is routed through Phase 3 diagnosis and recovery, after which the same intent can produce another empty sequence and loop. The debugger must confirm the exact trace in existing logs and code.

### Error Messages
Not yet confirmed. Inspect existing E2E output and API logs for validation, simulator, or synthetic empty-sequence errors.

### Timeline
Reported in the 2026-07-14 project checkpoint after navigation spikes, bug fixes, tree history, ablation infrastructure, and StuckTracker integration. Whether it existed earlier is unknown.

### Reproduction
Inspect existing E2E/API logs for `actions: []` with `status: "done"`, trace the branch-runner handling, add a focused regression test, and run the smallest relevant test suite. Do not start a paid API-backed E2E run unless focused verification is insufficient or explicitly approved.

### Working Tree Safety
The current tree contains pre-existing modified and untracked files that contradict the older clean-tree checkpoint. Inspect diffs before editing and preserve unrelated user work. Do not revert or overwrite unrelated changes.

## Current Focus

reasoning_checkpoint:
  hypothesis: "`BranchRunner.run()` discarded an Executor `status: done` response and unconditionally dispatched its empty `actions` list as `MoveSequence`; the sequence helper manufactured a failure, causing incorrect Phase 3/4 recovery."
  confirming_evidence:
    - "The pre-fix integration invoked Planner review and constructed `MoveSequence` exclusively from Executor actions; it did not inspect Executor status."
    - "`_execute_move_sequence()` returns failure for `steps=[]`, and the caller routes non-success sequences through Phase 3/4."
    - "The current implementation now checks `status == 'done' and not actions` immediately after `execute_intent()`, completes the intent in history, and bypasses review/MoveSequence."
  falsification_test: "The deterministic runner test would call `review_actions`, construct an empty MoveSequence, or invoke Phase 3/4 when its first Executor response is `{actions: [], status: 'done'}`."
  fix_rationale: "The runner boundary now recognizes the Executor contract for an already-satisfied intent, records an empty completed intent, and returns control to the Planner without inventing a simulation failure."
  blind_spots: "Full discovery can reveal unrelated regressions. The regression test also needs to preserve execution for nonempty `status: done` output, while empty `partial`/`failed` responses retain existing failure feedback."

hypothesis: "The guarded branch completes an empty `status: done` intent without Planner review, MoveSequence dispatch, Phase 3, or Phase 4, while the next nonempty done response still executes."
test: "Run full unittest discovery under WSL after examining the exact source/test diff."
expecting: "All discovered tests pass; especially the empty-done regression confirms one Planner review for the second, nonempty response and task completion."
next_action: "Request human verification against the real Executor/API workflow; paid API-backed E2E was intentionally not launched without approval."

## Evidence

- timestamp: 2026-07-15
  checked: `src/branch_runner.py` Planner/Executor integration (lines 561-602)
  found: `execute_intent()` output is sent to `review_actions()`, then both approved and rejected paths unconditionally set `proposed_action = "MoveSequence"` using the Executor actions or Planner corrections. Neither branch reads `exec_result["status"]`.
  implication: An Executor reply of `{ "actions": [], "status": "done" }` is indistinguishable from an invalid empty sequence at the runner boundary.

- timestamp: 2026-07-15
  checked: `BranchRunner._execute_move_sequence()` and MoveSequence dispatch in `src/branch_runner.py` (lines 863-956, 1301-1315)
  found: The sequence helper returns `success=False` for an empty `steps` list; the dispatch classifies every non-`all_succeeded` sequence as an environment failure and invokes Planner Phase 3 followed by Oracle Phase 4 and recovery.
  implication: The reported empty-action Phase 3 recovery loop is directly reachable without any simulator failure.

- timestamp: 2026-07-15
  checked: `src/executor.py` Executor response contract (lines 42-53, 165-183)
  found: The contract allows `status: "done"` and the implementation returns the model response without validating or interpreting the status/action combination (except the overlong-action safety cap).
  implication: Empty `actions` with status `done` can reach the runner and must be handled at the runner integration boundary.

- timestamp: 2026-07-15
  checked: working tree status and relevant tracked diffs
  found: The user has pre-existing modifications in unrelated scripts, package configuration, and source files; `src/branch_runner.py`, `src/executor.py`, and tests have no current diff.
  implication: A targeted runner change and focused test can preserve unrelated user work.

- timestamp: 2026-07-15T00:10:00Z
  checked: `src/branch_runner.py:561-616` and `tests/test_error_handling.py:160-231,497-522`
  found: The source has the targeted `status == "done" and not executor_actions` guard before Planner review. The regression test uses an empty first Executor response and a nonempty second `status: done` response, asserts task completion, exactly one review call, and a completed zero-step intent in the next Planner history; Phase 3/4 doubles raise if called.
  implication: The implementation directly covers the reported path and distinguishes empty terminal output from a nonempty terminal action sequence.

- timestamp: 2026-07-15T00:10:00Z
  checked: WSL command `uv run python -m unittest discover -s tests -v`
  found: All 26 discovered tests passed, including `test_executor_empty_done_completes_intent_without_phase3`.
  implication: The focused regression and adjacent unit tests pass without starting a paid API-backed E2E run.

## Eliminated

- hypothesis: The AI2-THOR simulator causes the reported Phase 3 diagnosis.
  evidence: The helper manufactures `success=False` for an empty Python list before it calls `env.step`, and the caller routes that result directly to Phase 3/4.
  timestamp: 2026-07-15

- hypothesis: Executor `status: "done"` already prevents MoveSequence dispatch.
  evidence: No `exec_result.get("status")` or equivalent status check exists between `execute_intent()` and the unconditional MoveSequence construction.
  timestamp: 2026-07-15

## Resolution

- root_cause: `BranchRunner.run()` discards the Executor response status. It always wraps `exec_result["actions"]` in `MoveSequence`; `_execute_move_sequence()` turns an empty list into a synthetic failure, which the runner incorrectly sends through Phase 3/4 recovery.
- fix: Added a runner-boundary guard immediately after `ExecutorAgent.execute_intent()`: when `status == "done"` and the action list is empty, `BranchRunner` marks and records the current intent as completed with zero steps, clears retry/failure state, and starts the next Planner cycle without review, MoveSequence, Phase 3, or Phase 4. Added a deterministic regression test exercising this flow and confirming a later nonempty `status: done` sequence still executes.
- verification: Source-level trace confirmed the guard precedes review and MoveSequence dispatch. WSL `uv run python -m unittest discover -s tests -v` passed all 26 tests, including `test_executor_empty_done_completes_intent_without_phase3`. A real API-backed E2E test remains for human verification because it would incur API usage and must observe the original runtime workflow.
- files_changed:
  - src/branch_runner.py
  - tests/test_error_handling.py
