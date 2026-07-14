---
spike: 006
name: search-trail-cost
type: standard
validates: "Given agent has visited Fridge area 3+ times, when Planner scores candidate locations, then revisit count penalizes repeated areas"
verdict: PENDING
related: [001, 003]
tags: [memory, trail, revisit-penalty, exploration]
---

# Spike 006 -- Search Trail Cost

## What this validates (Given/When/Then)

**Given** an EB agent has physically visited the grid cell near Fridge 3 or more times during a single episode.

**When** the Planner proposes the next intent targeting a receptacle area (e.g. "approach Fridge") and the trail shows that area has been visited 3+ times.

**Then** a TRAIL NOTE warning is injected into the NEXT Planner prompt: "You have already visited the Fridge area 3 times. Consider whether this is productive."

Additionally, every Planner and Executor prompt shows a compact TRAIL summary listing all known receptacle areas with their visit counts, encouraging prioritization of unvisited areas.

## Research notes

### Problem observed

The Planner repeatedly sends the agent to the same location (e.g., Fridge) even after the agent has physically been there multiple times. While the spatial memory (`EgocentricMemory`) tracks WHAT objects have been seen, it does not track WHERE the agent has physically been. This leads to wasteful revisits.

The `SearchTrail` tracks a 2D grid of visited positions and provides a revisit penalty signal.

### Design decisions

**1. Grid-based position tracking (not object-based)**

Rather than tracking which objects were visited (what spatial memory already does), the trail tracks the agent's physical position in a 2D grid. Grid resolution is 0.5m -- coarse enough to group nearby steps into "areas" but fine enough to distinguish distinct receptacle locations.

This is complementary to spatial memory: spatial memory says "you've seen Fridge, it's a receptacle", trail says "you've physically been near the Fridge's grid cell 3 times."

**2. Passive rendering (no forced blocking)**

The trail renders informational text in the Planner/Executor prompts, showing visit counts per receptacle area. This provides a soft signal rather than a hard block.

The soft-scoring warning only activates when the Planner independently proposes approaching a heavily-visited area (3+ visits) and an approach/check/search intent. It does NOT force the intent to change -- it injects a warning into the next prompt.

This is deliberate: hard blocking could prevent legitimate re-approaches (e.g., coming back to place an object on a previously-searched CounterTop).

**3. Trail recording at coarse points, not every sub-step**

Trail positions are recorded after each successful action at the branch_runner level (not inside `_execute_move_sequence` per sub-step). This is sufficient for revisit detection because:
- Sub-steps within a MoveSequence are small (0.125m each), often staying in the same 0.5m grid cell.
- The final position after a MoveSequence is where the agent "arrives" -- the location that matters for revisit scoring.

**4. render_summary uses visible receptacles from current frame**

The TRAIL summary shows visit counts for all visible receptacles in the current frame. This means remembered (not currently visible) receptacles are not shown on the trail -- the Planner already sees those in the spatial memory. The trail focuses on what the agent can act on right now.

### Alternatives considered

- **Force-block revisited areas**: Too aggressive -- the agent may need to return to a receptacle to place an object on it.
- **Track per-object rather than per-grid-cell**: Too fine-grained -- same area might have multiple objects of different types, and the agent shouldn't be penalized for visiting different objects in the same area.
- **Use AI2-THOR room segmentation**: Rooms are too coarse -- a kitchen has many distinct search areas (Fridge, CounterTop, Cabinet) that should be tracked separately.

## Implementation details

### Files modified

| File | Change |
|------|--------|
| `src/egocentric_memory.py` | Added `SearchTrail` class (grid-based position tracker) |
| `src/branch_runner.py` | Created `SearchTrail` instance, record after every successful step, render trail text, soft-scoring check after Planner intent |
| `src/eb_agent.py` | Added `trail_text` param to `plan_intent()`, render in prompt, added `_pending_trail_warning` field |
| `src/executor.py` | Added `trail_text` param to `execute_intent()`, render in prompt |

### Data flow

```
Agent moves → trail.record(x, z) after each successful step
                    ↓
    Before Planner call: trail.render_summary(receptacles)
                    ↓
        Planner prompt gets "TRAIL (visited areas): Fridge area x3, CounterTop x1"
                    ↓
        Planner proposes "approach Fridge" → soft-scoring check → 
            if Fridge area score >= 3 → _pending_trail_warning set
                    ↓
        Next Planner cycle: warning injected into prompt
```

### Soft scoring trigger

The soft-scoring check fires when ALL of:
1. The Planner's proposed intent targets a specific receptacle (`intent_target != ""`)
2. The intent contains approach/check/search keywords (`approach`, `open`, `close`, `check`, `search`, `look`)
3. The trail shows that target area has been visited 3+ times (`trail_score >= 3`)

This is intentionally narrower than the dedup trigger -- it only warns about approach-like intents, not about actions performed on a target (e.g., `pickup X on CounterTop` where CounterTop is the container, not the approach target).

## Remaining work

- [ ] End-to-end test: run an episode where the Planner loops "approach Fridge" and verify the TRAIL NOTE appears
- [ ] Tune visit threshold (3 may be too high/low depending on scene complexity)
- [ ] Consider extending `render_summary` to also include remembered receptacles from spatial memory
- [ ] Consider adding trail recording inside `_execute_move_sequence` for long sequences crossing multiple grid cells
