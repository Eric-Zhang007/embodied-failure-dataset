---
spike: 001
name: searched-markers
type: standard
validates: "Given a searched-and-empty receptacle, when Planner proposes next intent, then it no longer chooses to approach that receptacle"
verdict: PENDING
related: []
tags: [navigation, spatial-memory, planner, exploration]
---

# Spike 001 -- Searched Markers

## What this validates (Given/When/Then)

**Given** an EB agent has approached a receptacle (e.g. Fridge), opened it, visually confirmed the task target is NOT inside, closed it, and moved on.

**When** the Planner proposes the next intent.

**Then** the Planner no longer chooses to approach that same receptacle, because spatial memory now shows it as `[SEARCHED]`.

## Research notes

### Problem observed

The Planner repeatedly chooses `approach Fridge` as the next intent even after:
1. Successfully approaching the Fridge
2. Opening the Fridge
3. Finding that the target object is NOT inside
4. Closing the Fridge and leaving

The Planner's intent history and spatial memory had no mechanism to mark a receptacle as "already checked / empty." The Planner would return to the same receptacle multiple times because nothing in the prompt told it not to.

### Design decisions

**1. Where to store the flag: `_ObjectEntry.searched` (spatial memory)**

The `EgocentricMemory` class already accumulates all objects by `objectId` and tracks their status (visible/held/placed/remembered), direction, distance, and receptacle properties. Adding a `searched` boolean to `_ObjectEntry` is the natural home -- it persists across frames and renders in the spatial memory section that both Planner and Executor see.

**Alternatives considered:**
- Dedup in Planner prompt only: fragile, relies on VLM to remember across turns
- Separate "searched" set in branch_runner: works but not visible to VLM
- Mark in intent history: only visible to Planner, not Executor

**2. When to trigger: intent transition**

The `_check_and_mark_searched()` function is called when `current_intent` changes target. This is the natural place because:
- It captures all paths: MoveSequence success/failure, scan room, single actions
- It avoids premature marking (agent is still interacting with the same receptacle)
- The `completed` flag prevents marking on failures

**3. Dual detection: explicit OpenObject + approach heuristic**

Two trigger patterns are needed:

| Pattern | Detection | Example |
|---------|-----------|---------|
| Explicit open | `OpenObject` or `CloseObject` found in step history | Agent approaches, opens, checks, closes Fridge |
| Already open | Intent keywords (`approach`, `open`, `check`, `search`, `look`) + completed=True | Cabinet was already open; agent looked in, found nothing, moved on |

The already-open pattern (requirement 5) is handled by the `is_receptacle_intent` check -- if the intent uses approach/check language about a receptacle AND the last step succeeded (confirmed the agent was there), we mark it searched.

### Files changed

| File | Change | Lines |
|------|--------|-------|
| `src/egocentric_memory.py` | `searched: bool = False` field on `_ObjectEntry` | +1 |
| `src/egocentric_memory.py` | `mark_searched()` method on `EgocentricMemory` | +21 |
| `src/egocentric_memory.py` | `[SEARCHED]` tag in `_render_object_line()` | +2 |
| `src/branch_runner.py` | `_check_and_mark_searched()` call at intent transition | +2 |
| `src/branch_runner.py` | `_check_and_mark_searched()` function definition | +59 |

### Boundaries / known limitations

1. **Conservative completion guard**: Only marks when `completed=True` (last step succeeded). If the agent fails to reach the receptacle, it is NOT marked searched. This is intentional -- the agent should retry.

2. **Type-level marking**: `mark_searched(object_type=...)` marks ALL instances of that type. If the scene has two Fridges, both are marked. This is conservative but safe -- avoids revisiting both. If per-instance precision is needed later, use `mark_searched(object_id=...)` instead.

3. **SEARCHED is permanent**: Once marked, never cleared. Objects age out via `_age_entries()` (20-step threshold, remembered-only), so searched entries persist until natural aging.

4. **No distinction between "searched-empty" vs "searched-found"**: The tag just says `SEARCHED`. The Planner interprets it in context. This is intentional for the spike -- "don't go back here" is the right message either way.

5. **Single-agent path not covered**: The old `propose_action` codepath (no executor) does not track intent history, so searched markers won't trigger there. That path is deprecated.

## How to run / test

### Quick smoke test (Python import check)

```bash
cd ~/embodied-failure-dataset
uv run python -c "
from src.egocentric_memory import EgocentricMemory
m = EgocentricMemory()
# No objects yet, mark_searched should return 0
assert m.mark_searched(object_type='Fridge') == 0
print('OK: mark_searched on empty memory returns 0')
"
```

### Unit-level verification

```bash
uv run python -c "
from src.egocentric_memory import EgocentricMemory, _ObjectEntry

# Simulate an agent seeing Fridge
m = EgocentricMemory()
entry = _ObjectEntry(
    object_id='Fridge|+01.00|+00.00|+01.50',
    object_type='Fridge',
    is_receptacle=True,
    is_task_receptacle=False,
    status='visible',
    egocentric_dir='ahead',
    egocentric_dist=1.5,
    last_seen_step=1,
    seen_count=1,
)
m._objects[entry.object_id] = entry
m._step_counter = 1

# Before marking
assert '[SEARCHED]' not in m.render()
print('Before mark: no SEARCHED tag')

# Mark as searched
count = m.mark_searched(object_type='Fridge')
assert count == 1
print(f'Marked {count} object(s) as searched')

# After marking
rendered = m.render()
assert 'SEARCHED' in rendered
print('After mark: SEARCHED tag present')
print(rendered)
"
```

### Integration test (run an actual episode)

Run a single episode with `pick_heat_then_place_in_recep` task type to verify the searched markers appear in the pipeline output:

```bash
uv run python scripts/e2e_test.py \
  --api-key sk-f26... \
  --task pick_heat_then_place_in_recep \
  --random \
  --no-traps \
  --output spike001_test
```

Then check the episode JSON for spatial memory content in `eb_reasoning` fields and verify the agent does not loop on the same receptacle.

### Metrics to compare (for quantifying effect)

| Metric | Description | How to measure |
|--------|-------------|----------------|
| Receptacle revisit count | Number of times `approach <same_receptacle>` intent repeats | Count duplicate `(intent, target)` pairs in episode JSON steps |
| Total steps to find target | Steps before target object is picked up | Find first `PickupObject <target>` step index |
| Distinct receptacles explored | How many different receptacles the agent visits before finding target | Count unique `target` values in intent history |
| Success rate | % episodes where task completes | tasks completed / total attempted |

### Expected improvement

The searched markers should reduce receptacle revisit count (fewer repeated `approach Fridge` intents) and potentially reduce total steps to find target (agent tries new receptacles instead of re-checking old ones). Success rate should be unchanged or improved.

## Investigation trail

1. Read `egocentric_memory.py` -- understood `_ObjectEntry`, `EgocentricMemory.update()`, `render()`, `_render_objects()`, `_render_object_line()`, and the existing `get_unvisited_receptacles()` helper.
2. Read `branch_runner.py` -- traced the main loop, intent tracking (`current_intent` / `intent_history`), the Planner->Executor->Review cycle, and MoveSequence execution.
3. Read `eb_agent.py` -- understood Planner intent format: `{"intent": "approach Fridge", "target": "Fridge", "reasoning": "..."}`. The `target` field maps to the AI2-THOR `objectType`.
4. Read `executor.py` -- confirmed Executor outputs action sequences with `OpenObject`/`CloseObject` as concrete actions.
5. Identified the intent transition point (line ~250) as the optimal hook for marking receptacles as searched.
6. Designed dual-trigger detection: explicit OpenObject/CloseObject in step history + intent keyword heuristic for already-open receptacles.
7. Implemented changes in 2 files with 5 modifications totaling ~85 lines.

## Results and verdict

**Verdict: PENDING** -- code implementation complete, awaiting runtime validation via episode run.

To complete the verdict, run the integration test above and compare the metrics against a baseline episode. If receptacle revisit count decreases and the agent no longer loops on `approach Fridge`, change verdict to `PASSED`. If no measurable difference, change to `FAILED` with notes.
