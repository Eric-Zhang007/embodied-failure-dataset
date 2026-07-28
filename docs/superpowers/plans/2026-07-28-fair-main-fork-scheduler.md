# Fair Main/Fork Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace depth-first fork insertion with a shared 1:2 fair scheduler and reject counterfactual branches deeper than three.

**Architecture:** Add a small `FairBranchQueue` with one main FIFO, three fork-depth FIFOs, and a `main, fork, fork` cursor. Both global and task-lane scheduler modes use it. Persist explicit fork depth for new jobs, derive it from parent links for legacy jobs, and terminally reject malformed or depth-4 lineages without invoking a worker.

**Tech Stack:** Python 3.10, `collections.deque`, `dataclasses`, `unittest`/pytest, existing atomic `EpisodeManager` JSON persistence.

---

## File Map

- Create `src/fair_branch_queue.py`: queue ownership, weighted selection, depth-priority selection, and queue counts.
- Create `tests/test_fair_branch_queue.py`: direct deterministic queue contract tests and 100-dispatch canary.
- Modify `src/branch_runner.py`: add `fork_depth` to `BranchConfig` and generated fork tasks.
- Create `tests/test_fork_depth.py`: branch-generation, legacy-depth resolution, rejection, and restart ordering tests.
- Modify `src/episode_manager.py`: persist derived depth and reject pending fork entries atomically.
- Modify `src/scheduler.py`: resolve/admit fork depth, use `FairBranchQueue` in both scheduler modes, and report queue counts.
- Preserve existing modified `tests/test_scheduler.py`; add only narrowly required integration tests there if existing helpers make a separate test impractical.

### Task 1: Deterministic Fair Queue

**Files:**
- Create: `src/fair_branch_queue.py`
- Create: `tests/test_fair_branch_queue.py`

- [ ] **Step 1: Write failing queue-order tests**

Add tests that construct tasks with `BranchConfig(fork_depth=...)` and assert:

```python
queue = FairBranchQueue()
for task in (main_1, main_2, fork_1a, fork_1b, fork_2a):
    queue.append(task)

assert [queue.popleft()["name"] for _ in range(5)] == [
    "main-1", "fork-1a", "fork-1b", "main-2", "fork-2a",
]
```

Split additional behaviors into separate tests: one-class draining, shallowest fork depth first, FIFO within one depth, newly appended child staying behind existing work, main retry at the main head while still weight-gated, and independent queue cursors.

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/test_fair_branch_queue.py -q`

Expected: collection failure because `src.fair_branch_queue` does not exist.

- [ ] **Step 3: Implement the minimal queue**

Implement this public surface:

```python
class FairBranchQueue:
    MAX_FORK_DEPTH = 3
    _SCHEDULE = ("main", "fork", "fork")

    def append(self, task: dict) -> None: ...
    def append_main_retry(self, task: dict) -> None: ...
    def popleft(self) -> dict: ...
    def counts(self) -> dict[str, int]: ...
    def __bool__(self) -> bool: ...
    def __len__(self) -> int: ...
```

`append()` routes main to its FIFO and forks to the exact depth FIFO, rejecting depths outside 1..3 with `ValueError`. `popleft()` advances the schedule cursor after every successful selection, including fallback selection when the scheduled class is empty. Fork fallback always selects the lowest non-empty depth.

- [ ] **Step 4: Run queue tests and verify GREEN**

Run: `uv run pytest tests/test_fair_branch_queue.py -q`

Expected: all queue tests pass.

- [ ] **Step 5: Commit the queue unit**

```bash
git add src/fair_branch_queue.py tests/test_fair_branch_queue.py
git commit -m "feat: add fair main fork work queue"
```

### Task 2: Explicit Fork Depth At Creation

**Files:**
- Modify: `src/branch_runner.py` (`BranchConfig`, `_build_fork_task`)
- Modify: `tests/test_fork_depth.py`

- [ ] **Step 1: Write failing branch-depth tests**

Exercise `_build_fork_task()` with parent configs at depths 0, 1, and 2. Assert generated task dictionaries contain depths 1, 2, and 3 respectively and that `BranchConfig(**fork_task)` retains the value. Use a fake episode object that captures `add_pending_fork()` without starting AI2-THOR.

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/test_fork_depth.py -q`

Expected: assertions fail because `BranchConfig` and fork task dictionaries do not contain `fork_depth`.

- [ ] **Step 3: Implement explicit depth propagation**

Change the dataclass and fork task construction to:

```python
@dataclass
class BranchConfig:
    episode_id: str
    branch_id: str
    parent_branch_id: Optional[str]
    shared_context_step_ids: list[str] = field(default_factory=list)
    diverges_at_step_id: Optional[str] = None
    fork_config: Optional[dict] = None
    fork_depth: int = 0

# in _build_fork_task
"fork_depth": config.fork_depth + 1,
```

Keep the new field last so positional main-branch constructors remain compatible.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run pytest tests/test_fork_depth.py -q`

Expected: explicit depth tests pass.

- [ ] **Step 5: Commit depth propagation**

```bash
git add src/branch_runner.py tests/test_fork_depth.py
git commit -m "feat: persist counterfactual branch depth"
```

### Task 3: Legacy Lineage Resolution And Durable Rejection

**Files:**
- Modify: `src/episode_manager.py`
- Modify: `src/scheduler.py`
- Modify: `tests/test_fork_depth.py`

- [ ] **Step 1: Write failing legacy-lineage tests**

Build episode dictionaries whose pending tasks intentionally use opaque names unrelated to depth. Assert the scheduler resolver returns 1, 2, and 3 by following `parent_branch_id`. Add separate tests for a missing parent, a cycle, and depth 4.

Add persistence assertions:

```python
manager.set_pending_fork_depth("opaque-child", 2)
manager.reject_pending_fork(
    "opaque-too-deep",
    termination_reason="fork_depth_limit",
    fork_depth=4,
    diagnostic="maximum fork depth is 3",
)
```

Reload the JSON and verify the task depth was backfilled and the rejected entry is retained with `state="rejected"` but excluded by `get_pending_fork_tasks()`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/test_fork_depth.py -q`

Expected: missing resolver and `EpisodeManager` mutation methods.

- [ ] **Step 3: Implement atomic persistence methods**

Add `EpisodeManager.set_pending_fork_depth(branch_id, fork_depth)` and `EpisodeManager.reject_pending_fork(...)`. Both find the registry entry by branch ID inside `_mutate`; the first updates both the registry entry and nested task, while the second sets terminal state/reason/depth/diagnostic without modifying `final_outcome.main_branch`.

- [ ] **Step 4: Implement the scheduler resolver**

Add `Scheduler._resolve_fork_depth(episode, fork_task)`. It accepts an explicit integer depth only when consistent with the parent chain if the parent has explicit metadata. For legacy tasks, index all persisted pending tasks by branch ID, treat `main` as depth 0, follow parent links with a visited set, and raise a typed scheduler-only `InvalidForkLineage` for missing parents, cycles, non-main roots, or invalid depth values.

Add `_prepare_fork_task()` to backfill valid legacy depth, reject invalid lineage as `invalid_fork_lineage`, and reject resolved depth above 3 as `fork_depth_limit`. It returns `None` for rejected tasks so no worker can receive them.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `uv run pytest tests/test_fork_depth.py tests/test_pending_forks.py -q`

Expected: all depth and existing pending-fork persistence tests pass.

- [ ] **Step 6: Commit lineage validation**

```bash
git add src/episode_manager.py src/scheduler.py tests/test_fork_depth.py
git commit -m "fix: bound persisted fork lineage depth"
```

### Task 4: Integrate Fair Selection In Both Scheduler Modes

**Files:**
- Modify: `src/scheduler.py`
- Modify: `tests/test_fork_depth.py`
- Modify only if necessary: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing scheduler-boundary tests**

Create no-network scheduler fixtures with `_run_task_worker` replaced by a deterministic function that records dispatch order. Cover:

- global mode emits `main, fork, fork` while both classes remain populated;
- task lanes use one `FairBranchQueue` per task type and independent cursors;
- released child forks go to queue tails;
- a main retry uses `append_main_retry` but waits for a main weighted slot; and
- a depth-3 result that proposes a child records `fork_depth_limit` and never invokes the worker for that child.

- [ ] **Step 2: Run the integration tests and verify RED**

Run: `uv run pytest tests/test_fork_depth.py tests/test_scheduler.py -q`

Expected: ordering assertions fail against the existing `deque`/`appendleft` implementation.

- [ ] **Step 3: Integrate global mode**

Replace the global `pending: deque` with `FairBranchQueue`. Route initial jobs through the queue, main retries through `append_main_retry`, and valid fork descendants through `append`. Keep future completion and worker-count behavior unchanged.

- [ ] **Step 4: Integrate task-lane mode**

Replace each lane deque with an independent `FairBranchQueue`. Remove reversal and `appendleft` for new forks. Preserve one active worker per ALFRED task type and the current empty-lane validation.

- [ ] **Step 5: Make startup use depth preparation**

During `load_tasks()`, run every persisted pending fork through `_prepare_fork_task()` before constructing `BranchConfig`. New child admission uses the same validation path, ensuring live and restart behavior cannot differ.

- [ ] **Step 6: Run scheduler tests and verify GREEN**

Run: `uv run pytest tests/test_fair_branch_queue.py tests/test_fork_depth.py tests/test_pending_forks.py tests/test_scheduler.py tests/test_scheduler_no_traps.py -q`

Expected: all scheduler-related tests pass.

- [ ] **Step 7: Commit scheduler integration**

```bash
git add src/scheduler.py tests/test_fork_depth.py tests/test_scheduler.py
git commit -m "fix: schedule main and fork work fairly"
```

### Task 5: Queue Observability And Verification

**Files:**
- Modify: `src/scheduler.py`
- Modify: `tests/test_fair_branch_queue.py`

- [ ] **Step 1: Write a failing queue-count/report test**

Assert `counts()` returns exactly:

```python
{"main": 2, "fork_depth_1": 3, "fork_depth_2": 1, "fork_depth_3": 0}
```

Assert scheduler progress formatting includes `queued main=... fork[d1=...,d2=...,d3=...]` without parsing branch names.

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_fair_branch_queue.py -q`

Expected: report assertion fails until queue counts are connected to scheduler output.

- [ ] **Step 3: Implement queue reporting**

Track the active queue-count provider in each mode and extend `_report()` with main and per-depth fork counts. Log rejected tasks with episode, branch, parent, resolved depth, and reason. Do not modify the PowerShell collector dashboard in this phase.

- [ ] **Step 4: Run all relevant tests**

Run:

```bash
uv run pytest \
  tests/test_fair_branch_queue.py \
  tests/test_fork_depth.py \
  tests/test_pending_forks.py \
  tests/test_scheduler.py \
  tests/test_scheduler_no_traps.py \
  tests/test_e2e_status.py -q
```

Expected: zero failures.

- [ ] **Step 5: Run the 100-dispatch synthetic canary**

Run: `uv run pytest tests/test_fair_branch_queue.py -q -k hundred_dispatch`

The canary continuously replenishes both classes, asserts exactly 34 main and 66 fork selections across 100 dispatches, and asserts no selected fork has depth above 3.

- [ ] **Step 6: Run the broader unit suite**

Run: `uv run pytest tests -q`

Expected: zero failures, excluding tests that explicitly require an unavailable Unity display or live API only when those tests are marked/skipped by the repository. Any unmarked environment failure is reported rather than hidden.

- [ ] **Step 7: Inspect the final diff and commit observability**

```bash
git diff --check
git diff --stat
git add src/scheduler.py tests/test_fair_branch_queue.py
git commit -m "test: verify bounded fair branch scheduling"
```

### Task 6: Post-Fix Data And Exploration Audit

**Files:**
- Read only: latest `output_collector_*` episode JSON files and failure logs
- No production-code changes

- [ ] **Step 1: Establish dataset denominators**

Count episode files, main outcomes, fork registry states, unique executed branches, branch depths, replay failures, worker crashes, path-length crashes, and task-complete outcomes. Separate registered branches from executed and terminal branches.

- [ ] **Step 2: Score data usability**

Classify records into:

- usable main trajectory;
- usable counterfactual pair;
- diagnostic-only; and
- invalid/incomplete.

The rules must use persisted evidence: terminal outcome, replay status, exact resolved actions, screenshots, and branch lineage. Do not infer validity from file count.

- [ ] **Step 3: Measure exploration efficiency**

For main branches separately from forks, compute action success rate, navigation share, interaction share, repeated-intent rate, repeated successful pickup/put cycles, unique receptacles visited/searched per step, collision rate, steps to first task-object interaction, step-limit/dead-loop/permanent-stuck rates, and task completion rate by ALFRED task type.

- [ ] **Step 4: Compare cohorts**

Compare the latest `replancreditfix` batch with the nearest pre-snowball batches where fields are compatible. Explicitly distinguish agent-policy changes from scheduler amplification.

- [ ] **Step 5: Report findings and residual uncertainty**

Lead with validity and exploration conclusions, show denominators, identify which metrics are trustworthy, and recommend the smallest next canary needed before restarting collection.
