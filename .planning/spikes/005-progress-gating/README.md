---
spike: 005
name: progress-gating
type: standard
validates: "Given recovery phase after collision, when Planner proposes intent, then prompt includes phase-specific anti-fixation rules"
verdict: PENDING
related: [001, 002a, 003]
tags: [prompt, phase-detection, guardrails, exploration]
---

# Spike 005: Progress Gating

## What This Validates

The Planner's current prompt is monolithic -- the same system prompt applies whether exploring, approaching, interacting, or recovering from failure. This spike injects **phase-specific guardrails** into the prompt to give the Planner contextual rules tuned to the current situation.

### Hypothesis

> Adding phase-aware prompt injection that detects exploration, navigation, approach, interaction, and recovery phases from heuristic state signals will reduce fixation loops by giving the Planner targeted anti-pattern rules during recovery, without adding runtime cost or API calls.

## Approach

### Phase Detection (Heuristic, Zero-Cost)

Phase is determined from the current state BEFORE the VLM call:

| Phase | Detection Rule | Key Guardrails |
|-------|---------------|----------------|
| **exploration** | Primary target NEVER seen in spatial memory | Visit each location ONCE, do not revisit |
| **navigation** | Target seen but not currently visible | Move toward known location, check memory direction/distance |
| **approach** | Target visible, surface distance > 0.5m | Close distance with MoveSequence, no interaction yet |
| **interaction** | Target visible, surface distance <= 0.5m | Pick up / open / manipulate; re-assess after |
| **recovery** | Last action failed (last_error or incomplete recent intent) | Anti-fixation: do NOT repeat same action, rotate 90deg, mark receptacles SEARCHED |

### Priority Order

1. **recovery** (highest priority -- if last step failed, recovery guidance takes precedence)
2. **exploration** (target never seen)
3. **interaction** (target visible and within 0.5m)
4. **approach** (target visible but > 0.5m)
5. **navigation** (target seen but not currently visible)

### Phase-Specific Anti-Fixation Rules (Recovery Phase)

The recovery phase rules directly attack the fixation problem:

```
PHASE: RECOVERY
The last action FAILED. You must diagnose and try an alternative.
- Do NOT repeat the same action or approach the same target.
- If a receptacle was opened and found empty, mark it as SEARCHED.
- Rotate 90 and look for alternative paths or unexplored receptacles.
- If stuck for 3+ steps, propose "scan room" to re-assess.
- Identify the root cause: was it a collision, a reach error, or a missing object?
```

## Implementation

### Files Modified

1. **`src/eb_agent.py`** -- Core phase detection and injection
   - `PHASE_RULES`: Module-level dict mapping phase names to guardrail text blocks (5 phases, 3-5 rules each)
   - `EBAgent.__init__`: Added `_current_phase` and `_phase_transitions` tracking
   - `EBAgent.reset_dedup_stats()`: Resets phase tracking per episode
   - `EBAgent._extract_targets(task_goal)`: Static method extracting CamelCase object type names following "the " in task goal strings
   - `EBAgent._detect_phase(task_goal, visible_objects, memory, last_error, intent_history, agent_pos, agent_rot_y)`: Heuristic phase classifier using spatial memory queries and visible object data. Gracefully handles `memory=None` fallback.
   - `EBAgent._get_phase_block(phase)`: Returns the phase-specific guardrail text
   - `EBAgent.plan_intent()`: Phase detection call wrapped in try/except at the top of prompt building, right after task goal line. Phase block injected before critic feedback and other context.
   - Result dict includes `phase` field for downstream logging
   - Phase transitions logged to `_phase_transitions` list: `[{from, to, step_index}]`

2. **`src/egocentric_memory.py`** -- Phase detection query helpers
   - `EgocentricMemory.has_type(object_type)`: Check if any instance of this type has ever been seen
   - `EgocentricMemory.is_type_visible(object_type)`: Check if an instance is currently in view
   - `EgocentricMemory.get_type_distance(object_type)`: Get AABB surface distance to closest visible instance
   - `EgocentricMemory.get_remembered_type_distance(object_type)`: Get last known distance from memory
   - `EgocentricMemory.get_searched_receptacle_types()`: Get set of receptacle types marked as searched

3. **`src/branch_runner.py`** -- Integration
   - Primary `plan_intent()` call now passes `memory=memory` to enable spatial memory queries during phase detection

### Key Design Decisions

1. **Try/Except wrapping**: Phase detection is wrapped in try/except -- if it fails for any reason, the main execution flow continues unaffected. This is a pure prompt enhancement that can never cause a crash.

2. **Zero API cost**: Phase detection is pure heuristic logic (string extraction from task_goal, memory dict lookups) -- no additional LLM calls.

3. **Memory-aware fallback**: When `memory=None`, phase detection falls back to `visible_objects` only. When memory is available, it uses AABB surface distances from spatial memory for accurate distance checks.

4. **Prompt position**: Phase block is injected immediately after the task goal line and BEFORE any other context (critic feedback, memory text, curiosity table). This ensures the phase-specific rules are the most prominent instruction the Planner reads after the task goal.

5. **Recovery priority**: Recovery phase takes highest priority because a recent failure changes the Planner's needs more than any other state signal.

6. **Phase transitions logged**: When phase changes, a transition record is appended to `_phase_transitions` with `{from, to, step_index}`, enabling post-hoc analysis of phase flow.

### Phase Transition Tracking

Each phase transition is recorded:
```python
{
    "from": "exploration",   # or None for first detection
    "to": "navigation",
    "step_index": 5
}
```

Available on `eb_agent._phase_transitions` after the episode.

## How to Run

```bash
cd ~/embodied-failure-dataset
export PATH="$HOME/.local/bin:$PATH"

# Single task test
uv run python scripts/e2e_test.py \
  --api-key sk-f26f32d16f62c017d8779940e3917f1c211a33ad80dc2ec7d2d5718e5ef42139 \
  --task pick_and_place_simple \
  --random \
  --no-traps
```

### Check Phase Transitions

After an episode runs, inspect the phase transitions:

```python
# In Python after running:
# eb_agent._phase_transitions contains:
# [
#   {"from": None, "to": "exploration", "step_index": 0},
#   {"from": "exploration", "to": "approach", "step_index": 3},
#   {"from": "approach", "to": "interaction", "step_index": 5},
#   {"from": "interaction", "to": "recovery", "step_index": 7},
#   ...
# ]
```

### Verify Phase Block in Prompt

Phase is also included in the `plan_intent` result dict:
```python
planner_intent["phase"]  # "exploration", "navigation", "approach", "interaction", or "recovery"
```

## Results and Verdict

### Status: PENDING

- [ ] Run episodes and verify phase blocks appear in Planner prompts
- [ ] Verify phase transitions track correctly through a complete episode
- [ ] Measure if recovery-phase anti-fixation rules reduce repeated intents
- [ ] Compare task completion rates with vs without phase gating
- [ ] Check if exploration-phase rules reduce wandering loops
- [ ] Analyze if phase detection heuristics are accurate (are we detecting the right phase?)

### Metrics to Track

| Metric | Without Gating | With Gating | Delta |
|--------|---------------|-------------|-------|
| Average steps per completed episode | TBD | TBD | TBD |
| Intent repetition rate (same intent 3+ times) | TBD | TBD | TBD |
| Recovery-phase steps (time spent in recovery) | TBD | TBD | TBD |
| Exploration-phase steps before first target sighting | TBD | TBD | TBD |
| Phase transition count (more = more adaptive) | TBD | TBD | TBD |
| Episode success rate | TBD | TBD | TBD |

### Verdict Options

- **PASS**: Phase gating measurably reduces fixation, recovery loops, or improves completion rate
- **FAIL**: Phase gating has no measurable impact
- **CONDITIONAL**: Helps in specific phases (e.g., recovery) but not others -- refine rules per phase
