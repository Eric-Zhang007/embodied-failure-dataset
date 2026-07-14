---
spike: 002a
name: intent-dedup
type: comparison
validates: "Given N repeated failed intents, when Planner proposes the same intent again, then the system blocks it and forces exploration"
verdict: PENDING
related: [001]
tags: [navigation, planner, dedup, exploration]
---

# Spike 002a: Intent Deduplication

## What This Validates

The Planner (32B model) frequently repeats the same high-level intent (e.g., "approach Fridge") even after multiple failed attempts. The intent history is available in the prompt, but the model ignores it or doubles down. This spike implements a **hard mechanism** (not prompt-based) to detect and block repeated intents.

### Hypothesis

> A hard deduplication check that counts repeated (intent, target) pairs in recent history and blocks them after 3+ INCOMPLETE occurrences will reduce dead-loop episodes and improve task completion rates.

## Approach Comparison

### Before (No Dedup)

```
Planner proposes "approach Fridge" -> Executor tries -> fails (blocked)
Planner proposes "approach Fridge" -> Executor tries -> fails (blocked)
Planner proposes "approach Fridge" -> Executor tries -> fails (blocked)
Planner proposes "approach Fridge" -> Executor tries -> fails (blocked)
... dead loop, episode fails or times out
```

The Planner sees the intent history in its prompt but ignores the failure pattern.

### After (With Dedup)

```
Planner proposes "approach Fridge" -> Executor tries -> fails (blocked)
  -> 1st failure, no dedup action

Planner proposes "approach Fridge" -> Executor tries -> fails (blocked)
  -> 2nd failure, WARNING: hard constraint injected into NEXT prompt
  -> "You have already tried 'approach Fridge' 2 times without finding the target. Choose a DIFFERENT approach."

Planner sees HARD CONSTRAINT, proposes "approach CounterTop" instead
  -> explores alternative area, may find target there

OR (if Planner ignores warning):

Planner proposes "approach Fridge" AGAIN
  -> 3rd failure pattern detected -> FORCE OVERRIDE
  -> System replaces intent with fallback: "approach <nearest unseen receptacle>" or "scan room"
```

### Mechanism Tiers

| Tier | Condition | Action |
|------|-----------|--------|
| Normal | 0-1 prior INCOMPLETE occurrences | No action |
| Warning | 2 prior INCOMPLETE occurrences | Inject hard constraint into next prompt |
| Force | 3+ prior INCOMPLETE occurrences | Override intent with fallback exploration |

### Fallback Strategy

When forcing rejection (3+ occurrences), the system generates a fallback intent using spatial memory:

1. **Unvisited receptacles**: Receptacle types in memory that haven't been approached yet (prioritized by recency)
2. **Visible receptacles**: Currently visible receptacles (prioritized by proximity)
3. **Ultimate fallback**: "scan room" to get a fresh overview

## Implementation

### Files Modified

1. **`src/eb_agent.py`** -- Core dedup logic
   - `EBAgent.__init__`: Added `_pending_dedup_constraint` and `dedup_stats` tracking
   - `EBAgent.reset_dedup_stats()`: Reset per episode
   - `EBAgent.plan_intent()`: Added `memory` parameter; injects pending constraints before VLM call; checks proposed intent against history after VLM returns
   - `EBAgent._check_intent_dedup()`: Counts (intent, target) occurrences in last 5 history entries, returns (blocked, warning, fallback)
   - `EBAgent._generate_fallback_intent()`: Uses spatial memory to find alternative exploration targets

2. **`src/egocentric_memory.py`** -- Exploration target helpers
   - `EgocentricMemory.get_remembered_receptacles()`: Return remembered (not visible) receptacle types
   - `EgocentricMemory.get_visible_receptacles()`: Return visible receptacle types sorted by distance
   - `EgocentricMemory.get_unvisited_receptacles(approached_types)`: Return receptacle types not yet approached, excluding given set

3. **`src/branch_runner.py`** -- Integration
   - `BranchRunner.run()`: Passes `memory=memory` to `plan_intent()`; logs dedup decisions to failure log
   - `run_single_branch()`: Calls `eb_agent.reset_dedup_stats()` at episode start

### Key Design Decisions

1. **Check AFTER VLM proposes, not before**: We don't preempt the Planner -- it might adapt on its own. Only block if it repeats despite all available context.
2. **Memory-based fallback, not hardcoded**: The fallback uses actual spatial memory (what the agent has seen) rather than generic actions like "rotate right". This ensures fallbacks are context-aware.
3. **Constraint injection for NEXT call**: When 2 occurrences are detected, the constraint is stored and injected into the next prompt, giving the Planner one more chance to self-correct before forced override.
4. **Only checks INCOMPLETE intents**: If any prior occurrence of the same (intent, target) completed successfully, dedup is not triggered -- the pair is valid.

### Statistics Tracked

Per episode (`eb_agent.dedup_stats`):
- `warnings`: Number of 2-occurrence warnings issued
- `forced`: Number of 3+ occurrence forced overrides
- `fallbacks`: List of `{original_intent, original_target, fallback_intent, prior_attempts}`
- `blocked_intents`: List of `{intent, target, prior_attempts}` pairs that were blocked

Logged to `failures_{branch_id}.jsonl`:
- `failure_type: "dedup_warning"`: Warning events
- `failure_type: "dedup_blocked"`: Force-override events

## Research Notes

### Why Prompt-Based Solutions Fail

The Planner is a 32B model with `reasoning_effort=xhigh`. Despite having:
- Intent history in the prompt showing repeated failures
- Spatial memory showing obstacles
- Last error messages

...it still repeats the same intent. This is consistent with known LLM failure modes:
- **Recency bias**: The model weights its own current reasoning over historical data in the prompt
- **Anchoring**: Once it decides "I need to approach X", it sticks to that plan
- **Prompt saturation**: Dense prompts with spatial memory + objects + history cause the model to miss patterns

### Why Hard Mechanisms Are Necessary

Hard mechanisms provide guarantees that prompt engineering cannot:
1. **Deterministic**: Always triggers at the same threshold, not dependent on model attention
2. **Escalating**: Tiered response (warn -> force) gives the model a chance to self-correct
3. **Measurable**: Statistics are exact, not estimated from prompt interpretation

### Related Work

- Spike 001 (searched-markers): Marks receptacles as "already searched" to prevent revisiting
- CriticGuard (002b): Validates Planner intents before Executor execution

## How to Run

### Run a Single Episode

```bash
cd ~/embodied-failure-dataset
export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/e2e_test.py \
  --api-key sk-f26f32d16f62c017d8779940e3917f1c211a33ad80dc2ec7d2d5718e5ef42139 \
  --task pick_and_place_simple \
  --random \
  --no-traps
```

### Check Dedup Stats

After an episode runs, check the dedup statistics:

```python
# In Python after running:
from src.eb_agent import EBAgent
# eb_agent.dedup_stats contains:
# {
#   "warnings": int,
#   "forced": int,
#   "fallbacks": [...],
#   "blocked_intents": [...]
# }
```

### Check Dedup Logs

Dedup events are logged to `output_e2e_*/{episode_id}/failures_main.jsonl`:

```bash
grep '"dedup_' output_e2e_*/{episode_id}/failures_main.jsonl | python -m json.tool
```

### Run Comparison (Dedup vs No Dedup)

To run a controlled comparison, modify the `plan_intent` call to conditionally skip dedup:

```python
# In branch_runner.py, change:
memory=memory,  # enables dedup
# to:
memory=None,   # disables dedup
```

Or add a `--no-dedup` flag via `BranchRunner.__init__` parameter.

## Investigation Trail

### Metrics to Track

For each episode, record:
1. **Dedup warnings**: count of 2-occurrence warnings
2. **Dedup forced overrides**: count of 3+ occurrence blocks
3. **Fallback success rate**: did the fallback intent lead to finding the target within 3 subsequent steps?
4. **Episode outcomes**: task_complete vs dead_loop vs step_hard_limit vs unrecoverable

### Comparison Framework

| Metric | Without Dedup | With Dedup | Delta |
|--------|--------------|------------|-------|
| Episodes with dead_loop termination | TBD | TBD | TBD |
| Average steps per completed episode | TBD | TBD | TBD |
| Unique intents per episode (diversity) | TBD | TBD | TBD |
| Intent repetition rate (same intent/target 3+ times) | TBD | TBD | TBD |
| Fallback generated count | N/A | TBD | N/A |
| Fallback success rate | N/A | TBD | N/A |

### Log Format for Analysis

Each dedup event in the failure log follows this schema:

```json
// Warning event
{
  "step_index": 12,
  "branch_id": "main",
  "failure_type": "dedup_warning",
  "warning": "You have already tried 'approach Fridge' 2 times...",
  "proposed_intent": "approach Fridge",
  "proposed_target": "Fridge"
}

// Force-override event
{
  "step_index": 15,
  "branch_id": "main",
  "failure_type": "dedup_blocked",
  "original_intent": "approach Fridge",
  "original_target": "Fridge",
  "override_intent": "approach CounterTop",
  "eb_reasoning": "[DEDUP OVERRIDE] Original intent 'approach Fridge' blocked..."
}
```

## Results and Verdict

### Status: PENDING

Results will be populated after running comparison experiments.

- [ ] Run N episodes without dedup, collect baseline metrics
- [ ] Run N episodes with dedup, collect comparison metrics
- [ ] Analyze fallback success rate
- [ ] Determine if hard constraint injection (2-occurrence warning) actually changes Planner behavior
- [ ] Determine if spatial-memory-based fallbacks lead to task completion more often than dead loops

### Verdict Options

- **PASS**: Dedup measurably reduces dead loops or increases task completion rate
- **FAIL**: Dedup has no measurable impact or degrades performance
- **CONDITIONAL**: Dedup helps in specific scenarios but needs refinement (e.g., threshold tuning)
