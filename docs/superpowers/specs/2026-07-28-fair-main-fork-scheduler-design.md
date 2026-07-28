# Fair Main/Fork Scheduler Design

## Purpose

Prevent recursively generated counterfactual branches from starving main episodes or driving one task lane into depth-first branch expansion. Preserve nested counterfactual experiments up to depth 3 without imposing a per-episode fork-count budget.

This design changes scheduling and the minimum branch metadata needed by scheduling. It does not decide which failures deserve a fork, redesign replay, reclassify runtime exceptions, change task-progress tracking, or migrate branch identifiers.

## Observed Failure Mode

The current task-lane scheduler inserts every newly released fork at the front of its lane. A fork can run with `enable_fork=True`, so it can create additional forks that are also inserted at the front. Once pending-fork persistence and nested-lineage replay became reliable, this formed a depth-first feedback loop.

The July 28 collection provides the concrete evidence:

- 80 episode files registered 1,411 forks.
- 704 forks remained in `pending` or `running` state at inspection time.
- Branch depth reached 30.
- A parent branch produced 3.75 children on average and as many as 22.
- Earlier collections already reached depth 4, showing that recursion existed before the large run; later durability and recovery fixes allowed it to persist long enough to expand.

The scheduler must therefore provide both fairness between main and fork work and a structural depth boundary. A scheduling weight alone cannot bound recursive lineage, and a depth limit alone cannot prevent a large fork backlog from starving main work.

## Scheduling Contract

### Work classes

Each raw ALFRED task-type lane owns four FIFO queues:

1. `main`: main episode work, including interrupted-main resume jobs and main worker retries.
2. `fork_depth_1`: direct children of `main`.
3. `fork_depth_2`: children of depth-1 forks.
4. `fork_depth_3`: children of depth-2 forks.

The non-lane scheduler uses the same logical queue set globally. Both scheduler modes must call the same queue-selection component so their ordering rules cannot drift.

### Weighted fairness

When both main and fork work are available, selection follows the repeating schedule:

```text
main, fork, fork
```

This is a 1:2 main-to-fork dispatch ratio. The cursor advances only when a task is selected. If the selected class is empty, the other non-empty class fills the slot immediately; workers are never left idle to preserve the ratio. When the previously empty class receives work, it becomes eligible at the next matching weighted slot.

The ratio controls dispatch count, not wall-clock time. A long-running main or fork does not reserve later slots, and completed futures are processed independently of submission order.

### Fork ordering

The fork class uses breadth-first priority:

```text
depth 1 before depth 2 before depth 3
```

Within one depth, tasks are FIFO. Newly generated descendants are appended to their depth queue tail. They are never inserted at the front. This ensures that a newly completed fork cannot immediately monopolize a lane with its descendants.

Breadth-first priority is strict only among fork queues. It does not override the next `main` slot in the 1:2 weighted schedule.

### Lane isolation

With `task_lanes=True`, each ALFRED task type retains one active worker and its own fairness cursor and queues. A fork-heavy task type cannot consume another task type's lane. With `task_lanes=False`, the scheduler uses one global queue set and one fairness cursor across the configured worker pool.

## Fork Depth Contract

`main` has `fork_depth=0`. A fork has its parent's depth plus one. The maximum admitted depth is 3. A depth-3 branch may run normally, but any child it produces is rejected before it enters an executable queue.

Every newly created fork task persists an integer `fork_depth` alongside `branch_id` and `parent_branch_id`. `BranchConfig` carries the field at runtime. Depth is never inferred from the branch name for new records.

Legacy pending forks may lack `fork_depth`. On load, the scheduler derives their depth by following `parent_branch_id` through the episode's persisted branch tasks and outcomes until it reaches `main`. It must reject malformed legacy lineage when:

- a parent is missing;
- a parent cycle is detected;
- the chain does not terminate at `main`; or
- the derived depth exceeds 3.

The derived depth is used for that run. The scheduler may durably backfill it through `EpisodeManager`, but it must not rewrite unrelated episode fields. New records must always contain the explicit field.

## Admission And Terminal Recording

Scheduler admission remains identity-deduplicated by `(episode_id, branch_id)`. Depth validation happens before the identity is added to the admitted set and before the task is placed in a queue.

When a child exceeds the maximum depth, the scheduler must not execute it. Its pending-fork registry entry becomes terminal with:

```json
{
  "state": "rejected",
  "termination_reason": "fork_depth_limit",
  "fork_depth": 4
}
```

The rejection is included in scheduler statistics and logs, but it does not mark the episode or parent branch as failed. No descendants of the rejected branch can be released.

Malformed legacy lineage is rejected similarly with `termination_reason="invalid_fork_lineage"` and a diagnostic reason. These are scheduler terminal outcomes, not worker crashes.

There is no per-episode fork-count limit in this design. The depth boundary and fair dispatch are the only scheduler-level growth controls. Failure-value admission will be designed separately.

## Retry Contract

Only main worker crashes retain the existing automatic retry behavior. A retry is placed at the head of the main FIFO because it is continuation of already selected work, but it does not execute immediately unless the weighted cursor selects a main slot. It cannot bypass two eligible fork slots by using a separate submission path.

Fork worker crashes remain terminal and are not retried by this change. Replay-unavailable and incomplete-before-restart outcomes retain their existing terminal treatment.

## Startup And Resume

`load_tasks()` classifies every loaded job before execution:

- unfinished main episode to `main`;
- valid pending/running fork to its explicit or derived depth queue;
- finished or cancelled registry entry to no executable queue;
- depth-exceeded or malformed lineage entry to a persisted scheduler rejection.

Queue reconstruction must be deterministic. For jobs loaded from disk, FIFO order follows their order in `pending_forks`; main order follows the sorted trajectory order already used by the scheduler. Restarting cannot move descendants ahead of older forks at the same depth.

## Components And File Boundaries

### `src/scheduler.py`

Introduce a focused queue-selection unit, tentatively `FairBranchQueue`, responsible for:

- separate main and depth-indexed fork FIFOs;
- the `main, fork, fork` cursor;
- shallowest-first fork selection;
- enqueue-front support only for main retries; and
- queue length and emptiness reporting.

Both `Scheduler.run()` and `Scheduler._run_task_lanes()` use this unit. They remain responsible for futures, worker lifecycle, statistics, and translating completed results into newly admitted jobs.

### `src/branch_runner.py`

Extend `BranchConfig` with `fork_depth: int = 0`. `_build_fork_task()` writes `fork_depth=config.fork_depth + 1`. No branch execution, replay, failure diagnosis, or fork-value policy changes are included.

### `src/episode_manager.py`

Add narrow mutation methods to record a derived depth and to terminally reject one pending fork. Existing `finished` and `cancelled` states remain unchanged. The pending-fork registry continues to retain terminal records as an audit trail.

### Tests

Scheduler tests cover queue behavior directly and then exercise both scheduler modes at their integration boundary. Episode-manager tests cover depth persistence and rejection without invoking AI2-THOR or a VLM.

## Data Flow

```text
load main/pending jobs
        |
        v
resolve and validate fork_depth
        |
        +-- invalid or > 3 --> persist scheduler rejection
        |
        v
enqueue main or fork-depth FIFO
        |
        v
weighted selector: main, fork, fork
        |
        v
worker completes
        |
        +-- main retry --> head of main FIFO, still weight-gated
        |
        +-- valid child forks --> depth FIFO tail
        |
        +-- depth-4 child --> persist fork_depth_limit
```

## Required Tests

The implementation is accepted only if automated tests demonstrate all of the following:

1. With both classes continuously populated, dispatch order repeats `main, fork, fork`.
2. An empty class never idles a worker; the non-empty class drains in FIFO order.
3. When main work arrives after a fork-only period, it is selected at the next main slot.
4. Fork selection is depth 1, then 2, then 3, with FIFO stability within a depth.
5. A child fork is appended behind existing same-depth work rather than running immediately.
6. Main retry returns to the main head but remains governed by the fairness cursor.
7. Task lanes maintain independent cursors and cannot steal work from one another.
8. The non-lane scheduler uses the same selection semantics.
9. New fork tasks persist exact depths 1, 2, and 3.
10. A proposed depth-4 fork is not submitted and is persisted as `fork_depth_limit`.
11. Legacy lineage derives depths from parent links, never from `branch_id` text.
12. Missing-parent and cyclic legacy lineages become `invalid_fork_lineage`, not worker crashes.
13. Restart preserves pending FIFO order at each fork depth.
14. Existing duplicate suppression, invalid-parent cancellation, and main retry tests remain green.

No test in this phase requires Unity, network access, or model credentials.

## Operational Observability

Progress output must expose enough queue state to identify another imbalance without parsing branch names. At minimum, reports include queued main count and queued fork counts by depth. Logs for rejected branches include episode ID, branch ID, parent ID, resolved depth, and terminal reason.

The collector monitor may continue displaying the currently active branch. Dashboard redesign and historical-output audit are outside this phase.

## Rollout Gate

After unit and scheduler integration tests pass, run a no-network synthetic scheduler canary with generated jobs to prove bounded depth and weighted ordering over at least 100 dispatches. Then run a small real collection canary rather than resuming the large batch immediately. The real canary must show:

- no admitted branch deeper than 3;
- no `appendleft` path for newly generated forks;
- main jobs continue to start while fork jobs remain queued; and
- persisted queue counts reconcile with started and terminal branch counts.

Failure-value admission, fixed-length branch IDs, replay-event normalization, typed termination, and progress-cycle detection remain required follow-up work before treating a large collection as benchmark-ready.
