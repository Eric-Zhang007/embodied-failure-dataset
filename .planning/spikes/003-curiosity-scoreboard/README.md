---
spike: 003
name: curiosity-scoreboard
type: standard
validates: "Given a candidate exploration target, when the Planner proposes an intent, then it is guided by a multi-dimensional curiosity score that makes unexplored locations more attractive and visited locations progressively less attractive"
verdict: PENDING
related: [001, 002a, 002b]
tags: [navigation, curiosity, intrinsic-motivation, exploration, semantic-prior, habituation]
---

# Spike 003: Curiosity Scoreboard

## What This Validates

The Planner fixates on revisiting already-searched locations because all prompt-based context (intent history, spatial memory, last errors) fails to create a strong enough "don't go there" signal. Existing spikes 001 (searched-markers) and 002a (intent-dedup) are REACTIVE — they wait for the Planner to propose a bad intent, then block/interrupt. But blocking is not the same as guiding.

This spike introduces a PROACTIVE approach: compute a continuous, multi-dimensional "curiosity score" for every known receptacle and present it as a structured table BEFORE the Planner forms its intent. The table makes unexplored locations actively ATTRACTIVE (curiosity bonus) and visited locations progressively LESS attractive (habituation penalty). The Planner is not blocked from choosing a low-scoring location — it is shown WHY it should prefer a higher-scoring one.

### Hypothesis

> A multi-dimensional curiosity scoreboard presented proactively in the Planner's prompt (before intent formation) will reduce fixation loops by making alternative locations quantitatively more attractive, improving exploration diversity and reducing dead-loop episodes.

## Why This Is Innovative / Different from 001/002

| Dimension | 001 (searched-markers) | 002a (intent-dedup) | 002b (critic-guard) | **003 (curiosity-scoreboard)** |
|---|---|---|---|---|
| **Trigger** | Reactive: marks after interaction | Reactive: blocks after 2-3 repeats | Reactive: LLM critic checks before execution | **Proactive: scored table shown BEFORE intent formation** |
| **Signal type** | Binary (searched / not-searched) | Binary (blocked / not-blocked) | Binary (approved / rejected) | **Continuous [0.0, 1.0] multi-dimensional score** |
| **Mechanism** | Flag on memory entries | Hard threshold counter | Extra LLM call | **Deterministic scoring, no extra LLM calls** |
| **Direction** | Negative: removes options | Negative: blocks bad options | Negative: criticizes bad options | **Positive: makes good options MORE attractive** |
| **Knowledge** | Pure spatial (visited/not) | Pure history (repeat/not) | LLM reasoning about history | **Semantic prior + novelty + discovery reinforcement** |

### Three Scoring Dimensions

1. **Semantic Prior** (0.4 weight): "How likely is the target object to be found here?" Uses a static commonsense table mapping (object_category, receptacle_category) -> prior score. E.g., Apple in Fridge = 0.95, Apple in Sink = 0.15.

2. **Novelty Score** (0.4 weight): "How fresh is this location?" Exponential decay on visitation count. `novelty = exp(-visit_count)`. Visit 0 = 1.00, visit 1 = 0.37, visit 3 = 0.05. Inspired by habituation in neuroscience and the FICM paper (Yang et al., 2019).

3. **Discovery Score** (0.2 weight): "Did interacting with this receptacle reveal new objects?" Positive reinforcement. `min(1.0, objects_found / max(1, times_opened))`. Rewards receptacles that delivered discoveries.

### Combined Score Example

```
CURIOSITY SCOREBOARD — ranked by exploration priority
Target: Apple | Higher score = explore first
======================================================================
#  Location       Where                   Visit   Sem   Nov  Disc  Score Signal
--------------------------------------------------------------------------------
1  CounterTop     visible ahead 1.2m          0  0.70  1.00  0.50  0.88 ★★★ EXPLORE
2  Cabinet        visible right 1.8m          0  0.50  1.00  0.50  0.70 ★★ EXPLORE
3  Shelf          remembered (~8 steps ago)   0  0.30  1.00  0.50  0.62 ★★ EXPLORE
4  DiningTable    visible left 2.1m           0  0.60  1.00  0.50  0.74 ★★★ EXPLORE
5  Fridge         visible ahead 0.5m          3  0.95  0.05  0.50  0.50 ★ MAYBE
6  Sink           visible right 3.0m          1  0.15  0.37  0.50  0.31 ★ MAYBE
7  Microwave      visible ahead 1.0m          2  0.50  0.14  0.50  0.38 ★ MAYBE
--------------------------------------------------------------------------------
Sem=Semantic prior (how likely is Apple here?)
Nov=Novelty (1.0=never visited, decays with each visit)
Disc=Discovery (objects found here / times opened)
```

### Guard Rails (Post-Hoc)

After the VLM returns its proposed intent, two guards check the proposed target:

- **Soft guard** (score < 0.30 AND alternative >= 0.50 exists): Warns in the next prompt but allows the current intent. "Consider exploring X instead."
- **Hard guard** (score < 0.10): Force overrides the intent with the highest-scoring alternative. Similar to 002a's level-3 but based on SCORE not repetition count.

### Design Inspiration

- **SENSEI (ICML 2025)**: VLM-derived "interestingness" reward for model-based RL agents, distilled into intrinsic motivation signal
- **Episodic Curiosity / Reachability (Savinov et al., ICLR 2019)**: Novelty determined by "effort to reach" current state from past states — avoids fixation by tracking reachability, not visual novelty
- **FICM (Yang et al., 2019)**: Catastrophic forgetting in ICM causes agents to revisit states they've forgotten. Solution: optical flow-based novelty that doesn't forget
- **Semantic Frontier Exploration (Noda & Tanaka, 2025)**: LLM-guided frontier prioritization in AI2-THOR, 85% SR vs ~60% for geometric-only
- **iLLM (Bougie & Watanabe, ACML 2025)**: LLM as intrinsic reward source — maps state-action pairs to token embeddings for novelty scoring

## Implementation

### Files Modified

1. **NEW: `src/curiosity_scorer.py`** — Core scoring logic
   - `SEMANTIC_PRIORS`: Static table mapping (target_category, receptacle_category) -> prior score
   - `build_curiosity_table_text()`: Formats the CURIOSITY SCOREBOARD for prompt injection
   - `check_proposed_target()`: Post-hoc guard — checks proposed intent against scoreboard
   - `CuriosityStats`: Per-episode tracking (warnings, blocks, scores)

2. **MODIFIED: `src/egocentric_memory.py`** — Visitation tracking
   - `EgocentricMemory.__init__`: Added `receptacle_visit_counts`, `receptacle_open_counts`, `objects_found_by_receptacle` dicts
   - `record_receptacle_visit(otype)`: Increment visit count
   - `record_receptacle_open(otype)`: Increment open count (also counts as visit)
   - `record_object_discovered_in(otype)`: Increment discovery count
   - `get_receptacle_entries_for_curiosity()`: Returns receptacle data for the scoreboard
   - `_auto_record_discovery()`: Automatically credits parent receptacle when new object enters memory

3. **MODIFIED: `src/eb_agent.py`** — Scoreboard injection + post-hoc check
   - `EBAgent.__init__`: Added `curiosity_stats` and `_pending_curiosity_warning`
   - `EBAgent.reset_dedup_stats()`: Also resets curiosity state
   - `plan_intent()`: Injects curiosity scoreboard after memory text; injects pending curiosity warning; runs post-hoc check after VLM returns
   - `_build_curiosity_for_prompt()`: Bridge function — extracts target from task_goal, gathers memory data, calls scorer
   - `_check_curiosity_posthoc()`: Bridge function — checks proposed target against scoreboard

4. **MODIFIED: `src/branch_runner.py`** — Visitation recording
   - `_check_and_mark_searched()`: Added `memory.record_receptacle_visit()` and `memory.record_receptacle_open()` calls alongside existing `mark_searched()`

### Key Design Decisions

1. **Proactive, not reactive**: The scoreboard is computed and injected BEFORE the VLM call. The Planner sees the scores while forming its intent. This is fundamentally different from 001/002 which only act AFTER the bad intent is proposed.

2. **Continuous scores, not binary flags**: Instead of "SEARCHED" (001) or "BLOCKED" (002a), locations get a score between 0.0 and 1.0. This creates a gradient — the Planner can see that Fridge (0.50) is only "maybe" worth it while CounterTop (0.88) is a clear "explore now."

3. **Semantic commonsense baked in**: The static SEMANTIC_PRIORS table encodes commonsense knowledge (Apple -> Fridge, CounterTop; Pillow -> Bed, Sofa) without requiring an extra LLM call. This is a one-time curation cost, not a per-step cost.

4. **Discovery reinforcement**: The discovery score creates a self-reinforcing loop — receptacles that yielded new objects get scored higher, encouraging the agent to check novel receptacles first.

5. **Exponential habituation**: Each visit cuts novelty to ~37% of its previous value. After 3 visits, novelty approaches zero. This matches the FICM finding that agents forget visited states and treat them as novel again — by using an explicit counter, we prevent forgetting.

6. **Extracts target from task_goal**: A simple heuristic extracts the target object type (e.g., "Apple" from "Put a clean apple in the fridge.") from task_goal using a known-type lookup table. No additional parameters needed.

## How to Run/Test

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

### Check Curiosity Stats

After an episode runs, the `eb_agent.curiosity_stats` object tracks:

```python
from src.curiosity_scorer import CuriosityStats
# eb_agent.curiosity_stats.to_dict() returns:
# {
#   "scoreboard_shown": 12,       # times scoreboard appeared in prompts
#   "soft_warnings": 3,           # soft guard triggered
#   "hard_blocks": 1,             # hard guard triggered (force fallback)
#   "total_locations_scored": 6,  # unique receptacles scored
#   "avg_score_of_proposed_targets": 0.62,  # average curiosity score of chosen targets
#   "blocked_intents": [...]      # details of hard-blocked intents
# }
```

### Check Curiosity Logs

Curiosity events appear in `output_e2e_*/{episode_id}/steps.json` as part of step entries:

```bash
grep -o '"curiosity_[^"]*"[^}]*' output_e2e_*/{episode_id}/steps.json
```

Key fields: `curiosity_blocked`, `curiosity_warning`, `curiosity_original_target`.

### Comparison Test (Curiosity vs No Curiosity)

To disable curiosity for comparison, modify the `plan_intent` call in `branch_runner.py` to NOT pass memory for curiosity (the scoreboard won't be built if `memory` is None):

Or add a `--no-curiosity` CLI flag by modifying `BranchRunner.__init__` to accept `enable_curiosity: bool = True`.

### Unit Tests (Manual)

The core scoring logic can be tested without AI2-THOR:

```python
from src.curiosity_scorer import (
    compute_combined_score, compute_novelty_score,
    get_semantic_prior, build_curiosity_table_text,
)

# Test novelty decay
assert compute_novelty_score(0) == 1.0
assert round(compute_novelty_score(1), 2) == 0.37
assert compute_novelty_score(3) < 0.06

# Test semantic prior
assert get_semantic_prior("Apple", "Fridge") >= 0.90
assert get_semantic_prior("Pillow", "Fridge") <= 0.10
assert get_semantic_prior("Apple", "CounterTop") >= 0.60

# Test combined score
score = compute_combined_score("Apple", "Fridge", visit_count=3, objects_found=0, times_opened=3)
# Semantic=0.95*0.4=0.38, Novelty=0.05*0.4=0.02, Discovery=0.0*0.2=0.0 → 0.40
assert 0.35 <= score <= 0.45
```

## Expected Behavior

### Before (No Curiosity Scoreboard)

```
Planner: approach Fridge → open → nothing → (stuck, no guidance)
Planner: approach Fridge → open → nothing → (still stuck)
Planner: approach Fridge → (finally blocked by dedup)
Fallback: approach CounterTop → finds target
```

### After (With Curiosity Scoreboard)

```
Planner sees:
  CounterTop  ★★★ EXPLORE (score 0.88)
  Fridge      ✗ AVOID visited 3x (score 0.50)
  Cabinet     ★★ EXPLORE (score 0.70)

Planner: approach CounterTop (highest score!) → open → finds target
Or if not there: approach Cabinet → open → finds target
```

### Key Metrics to Track

| Metric | Expected Change |
|---|---|
| Unique receptacles visited per episode | Increase (more diverse exploration) |
| Steps per completed episode | Decrease (less wasted on repeats) |
| Dead-loop episodes | Decrease |
| Average curiosity score of chosen targets | Increase (Planner prefers high-score locations) |
| Soft guard warnings per episode | Initially ~2-3, should decrease as Planner adapts |
| Hard guard blocks per episode | <1 (Planner should self-correct on warnings) |

## What Could Go Wrong (Honest Assessment)

### 1. Semantic Prior Table Is Incomplete
**Risk:** The static SEMANTIC_PRIORS table only covers 9 target categories x 11 receptacle categories = 99 entries. Missing mappings default to 0.25 (conservative). If an object type isn't matched to ANY category, the _classify_target function defaults to "food_cold", which may be wrong.

**Mitigation:** The fuzzy-matching fallback in _classify_target covers many edge cases. The DEFAULT_SEMANTIC_PRIOR of 0.25 is intentionally low — it pushes the agent toward novelty rather than anchoring on a wrong semantic prior.

### 2. Exponential Decay Is Too Aggressive
**Risk:** HABITUATION_DECAY = 1.0 means novelty drops from 1.0 to 0.05 in just 3 visits. A receptacle might need to be visited multiple times for legitimate reasons (e.g., Fridge has multiple shelves).

**Mitigation:** The soft guard only warns when score < 0.30 AND alternatives >= 0.50 exist. The hard guard only forces at score < 0.10. So even a 3x-visited Fridge with semantic prior 0.95 still scores 0.50 (semantic keeps it afloat). Tuning HABITUATION_DECAY to 0.5 would make novelty last longer (visit 3 = 0.22).

### 3. Target Extraction Is Heuristic
**Risk:** `_extract_target_object()` uses a simple keyword match against known AI2-THOR types. Task goals like "Put a clean sponge in the sink" → "Sponge" is not in our known types list (it's "DishSponge" or "ScrubBrush"). The target may not be extracted, and the scoreboard won't be built.

**Mitigation:** If no target is extracted, `_build_curiosity_for_prompt` returns empty string — the scoreboard is silently skipped. This is graceful degradation. The known types list can be expanded.

### 4. Scoreboard Adds Prompt Bloat
**Risk:** The curiosity table adds ~15 lines to an already-dense prompt (~1-2KB of text). This could contribute to prompt saturation and cause the Planner to miss other important signals.

**Mitigation:** The table is capped at 8 rows and only shown when memory has receptacle data. Early steps (before any receptacles are known) won't show it.

### 5. Weight Orthogonality May Not Hold
**Risk:** Semantic prior and novelty are conceptually independent but may correlate in practice — unvisited receptacles tend to be semantically unlikely (e.g., Toilet for Apple). The combined score might not create useful differentiation.

**Mitigation:** The weights (0.4/0.4/0.2) balance the dimensions. If one dominates, they can be tuned. The soft/hard guard thresholds (0.30/0.10) are also tunable.

### 6. Discovery Tracking Is Approximate
**Risk:** `_auto_record_discovery` credits the parent receptacle when a new object enters memory. But the object might have been revealed by RotateLeft (not by opening a receptacle), causing false attribution.

**Mitigation:** The discovery score only has 0.2 weight, so false attribution is a minor issue. A more precise approach would track the action that revealed the object, but this adds complexity.

### 7. Interaction With Existing Spikes
**Risk:** The curiosity guard (soft/hard) may conflict with the dedup guard (warn/force). If both trigger on the same intent, the dedup takes precedence (checked first), then curiosity checks the (possibly already-overridden) result.

**Mitigation:** The curiosity post-hoc check explicitly skips if `result.get("dedup_blocked")` is True. The curiosity warning and dedup warning coexist — both are injected into the next prompt.

### 8. No Direct Control Over Planner Reasoning
**Risk:** The Planner might see the scoreboard but still propose a low-scoring target because of other constraints (obstacle avoidance, hand status, task progress). The scoreboard is guidance, not control.

**Mitigation:** This is by design. The hard guard at score < 0.10 provides a safety net, but the primary mechanism is attraction (making good targets more appealing), not coercion. This respects the Planner's reasoning while steering it.

## Results and Verdict

### Status: PENDING

Results will be populated after running comparison experiments.

- [ ] Run N episodes with curiosity scoreboard, collect metrics
- [ ] Run N episodes without curiosity, compare exploration diversity
- [ ] Analyze avg_score_of_proposed_targets trend (should increase)
- [ ] Tune HABITUATION_DECAY if novelty decays too fast/slow
- [ ] Tune SCORE_SOFT_GUARD / SCORE_HARD_GUARD thresholds
- [ ] Verify semantic prior table coverage across all ALFRED task types
- [ ] Test interaction with dedup (002a) and critic (002b)

### Verdict Options

- **PASS**: Curiosity scoreboard measurably increases exploration diversity and/or reduces dead-loop episodes
- **FAIL**: No measurable improvement over baseline; novelty decay too aggressive OR semantic prior table not useful enough
- **CONDITIONAL**: Helps in some task types (object search) but not others (heat/cool/clean). Scoreboard may need per-task-type specialization.
