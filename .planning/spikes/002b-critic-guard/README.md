---
spike: 002b
name: critic-guard
type: comparison
validates: "Given a proposed intent targeting an already-searched location, when Critic checks it, then the intent is rejected before execution"
verdict: PENDING
related: [001, 002a]
tags: [navigation, critic, validation, exploration]
---

# Spike 002b: Critic Guard

## What This Validates

The Planner repeatedly chooses "approach Fridge" even after opening it and finding nothing. This spike adds a lightweight pre-execution Critic that validates the Planner's proposed intent BEFORE the Executor wastes steps acting on it.

**Core hypothesis**: A lightweight validation layer (rule-based + optional LLM) can catch wasteful intents earlier and with more contextual awareness than hard count-based deduplication alone (002a).

## Comparison Table vs 002a (Intent Dedup)

| Dimension | 002a: Intent Dedup | 002b: Critic Guard |
|---|---|---|
| **Mechanism** | Hard count-based blocking after 3+ repeated (intent, target) pairs | Context-aware validation using rules + optional LLM |
| **When it triggers** | After 2 warnings, at 3+ repeated attempts | Every time before Executor, when enabled |
| **Context used** | Intent history counts only | Intent history + spatial memory + visible objects + searched receptacles |
| **Rejection granularity** | Coarse: "tried X 3+ times, block" | Fine: "X was already opened and found empty" / "X is known empty" / "X tried 3+ times" |
| **API calls** | Zero (purely rule-based) | Zero for rule mode; 1 text-only call for LLM mode |
| **Recovery** | Forces fallback intent from spatial memory | Feeds reason+suggestion back to Planner for 1 re-generation |
| **False positive risk** | High — blocks even if context changed (e.g., agent moved, new angle) | Low for rule checks; moderate for LLM checks |
| **False negative risk** | Medium — only catches exact (intent, target) matches | Lower — catches semantically similar wasteful behavior |
| **Latency overhead** | ~0ms | ~0ms (rule mode); ~200-500ms (LLM mode) |

## Design

### Architecture

```
Planner.propose_intent()
       |
       v
[002a: Intent Dedup] -- blocks repeated intents
       |
       v
[002b: CriticGuard.check()] -- validates with richer context
       |
       +--> approved? --> Executor.execute_intent()
       |
       +--> rejected? --> Planner.regen_intent(critic_feedback)
                              |
                              v
                         [CriticGuard.check() again]
                              |
                              +--> approved? --> Executor
                              +--> rejected? --> Executor (let through, track)
```

### Rule-Based Checks (always run)

| Rule | Condition | Rejection Reason |
|---|---|---|
| `repeated_intent` | Same (intent, target) tried 3+ times in intent_history | Pattern repetition |
| `searched_receptacle` | Intent is "approach X" where X was already opened | Already searched |
| `known_empty` | Target object was previously opened and found empty | Known to be empty |

### LLM-Based Critic (optional, text-only)

When `critic_llm=True`, the Critic gets:
- Proposed intent + target
- Intent history (what has been tried and outcomes)
- Currently visible objects list
- Already-searched receptacles list
- Spatial memory text

The LLM uses a compact system prompt (`CRITIC_SYSTEM`) and outputs:
```json
{"approved": bool, "reason": "1-2 sentences", "suggestion": "concrete alternative"}
```

### Re-Generation Flow

When the Critic rejects:
1. Build feedback text from `reason` + `suggestion`
2. Call `Planner.plan_intent(critic_feedback=feedback)` with the feedback as a hard constraint
3. The Planner sees "CRITIC REJECTED YOUR PREVIOUS INTENT" block at the top of its prompt
4. Re-check the re-generated intent with Critic
5. If still rejected: let it through but track in stats

Maximum 1 re-generation attempt.

### Statistics Tracked

```python
@dataclass
class CriticStats:
    total_checks: int       # total critic invocations
    approved: int           # passed validation
    rejected: int           # blocked by critic
    rule_rejected: int      # blocked by rule checks
    llm_rejected: int       # blocked by LLM check
    regen_attempted: int    # re-generation calls to Planner
    regen_approved: int     # re-generated intent passed critic
    regen_rejected: int     # re-generated intent also rejected
    rejection_reasons: dict # distribution of rejection reasons

    # Accuracy tracking (to be filled post-hoc)
    correct_rejections: int
    incorrect_rejections: int
    false_approvals: int
```

### Searched Receptacle Tracking

The CriticGuard tracks which receptacles have been opened/searched:
- When `OpenObject` succeeds on a receptacle (in MoveSequence or single-action path)
- Tracked via `critic_guard.mark_receptacle_searched(objectType)`
- Future "approach X" intents for already-searched receptacles are rejected

## Files Modified

| File | Changes |
|---|---|
| `src/critic_guard.py` | **NEW** — CriticGuard class with rule checks, LLM check, stats tracking, re-generation feedback builder |
| `src/branch_runner.py` | Added import, `enable_critic`/`critic_llm` params to `BranchRunner.__init__` and `run_single_branch()`, critic check in main loop before Executor call, `_track_searched_receptacles()` helper, receptacle tracking on OpenObject success |
| `src/eb_agent.py` | Added `critic_feedback: str | None` parameter to `plan_intent()` — shown as prominent "CRITIC REJECTED" block at top of prompt |

## How to Run

```bash
# Enable rule-based CriticGuard (no API calls)
uv run python scripts/e2e_test.py --api-key sk-f26... --task pick_and_place_simple --random --no-traps --enable-critic

# Enable LLM-based CriticGuard (text-only calls)
uv run python scripts/e2e_test.py --api-key sk-f26... --all --random --no-traps --enable-critic --critic-llm

# Via run_single_branch directly
from src.branch_runner import run_single_branch
result = run_single_branch(
    traj_path, eb_agent, oracle_agent, output_dir,
    enable_critic=True,
    critic_llm=False,  # True for LLM critic
)
```

**Note**: `--enable-critic` and `--critic-llm` flags need to be added to `scripts/e2e_test.py` argument parser (not done in this spike — prototype only).

## Investigation Trail with Metrics

### Research Notes

1. **SPIRAL paper (AAAI 2026)** uses a Critic agent alongside the Planner for dense feedback. Our implementation differs: SPIRAL's critic evaluates after execution (outcome-based), while our critic evaluates BEFORE execution (pre-validation). Both share the core insight that a second lightweight evaluation pass catches errors the primary agent misses.

2. **Why not just 002a?** The count-based dedup is fast but coarse. It only catches exact (intent, target) repeats after 3+ attempts. It cannot detect:
   - "approach Fridge" after "open Fridge" found nothing (different intents, same wasteful target)
   - "approach Counter1" then "approach Counter2" when both were already opened
   - Semantic equivalents ("go to Fridge" vs "approach Fridge")

3. **Why text-only for LLM critic?** The image is already processed by the Planner (expensive). The Critic only needs to reason about structured text (intent history, spatial memory, visible objects). Skipping the image saves ~80% of the API cost and latency.

4. **Complementarity with 002a**: The two mechanisms are designed to work together:
   - 002a runs first (inside `plan_intent`, post-response) — coarse hard blocking
   - 002b runs second (in `branch_runner`, pre-Executor) — fine context-aware validation
   - Both can be enabled simultaneously without conflict

### Expected Metrics (to be collected during evaluation)

| Metric | Expected Range | Measurement Method |
|---|---|---|
| Critic approval rate | 70-90% | `stats.approved / stats.total_checks` |
| Rule-triggered rejections | 5-20% of checks | `stats.rule_rejected / stats.total_checks` |
| LLM-triggered rejections | 3-10% of checks | `stats.llm_rejected / stats.total_checks` |
| Re-generation success rate | 30-60% | `stats.regen_approved / stats.regen_attempted` |
| False positive rate (blocking good intents) | <5% | Manual review of rejection reasons |
| Wasted step prevention | TBD | Compare step counts with/without critic |
| Latency overhead (rule mode) | ~0ms | N/A — pure Python checks |
| Latency overhead (LLM mode) | 200-500ms | Text-only API call timing |

### Accuracy Benchmarks (to be filled post-evaluation)

| Metric | 002a (Dedup) | 002b (Critic Guard) |
|---|---|---|
| False positive rate | TBD | TBD |
| Wasted steps prevented | TBD | TBD |
| Correct rejection rate | TBD | TBD |
| End-to-end success impact | TBD | TBD |

## Results and Verdict

**Status: PENDING** — Implementation complete, awaiting evaluation runs.

The implementation is functional and integrated into the main loop. To complete evaluation:

1. Run a set of episodes with `enable_critic=True` and compare against baseline (no critic)
2. Track `critic_guard.stats.to_dict()` for quantitative metrics
3. Manually review a sample of rejections for false positive rate
4. Compare with 002a-only runs to measure incremental benefit
5. Fill in the accuracy benchmarks table above

### Open Questions

1. Does the text-only LLM critic provide enough value over rule-only to justify the latency?
2. Should "searched receptacle" tracking also consider "approach X completed but target not found" (not just OpenObject)?
3. Is the 1 re-generation limit optimal, or should we allow 2?
4. Should the Critic also validate Executor output (action sequence) before execution?
