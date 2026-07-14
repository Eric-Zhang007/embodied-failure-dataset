# Spike 004: Contrastive Planner

**Status:** Implemented
**Type:** standard
**Validates:** Given repeated failed intents targeting the same location, when the Contrastive Planner is activated, then an explore-biased alternative intent is proposed and selected over the fixated one.

## Problem

The Planner fixates on revisiting already-searched locations. When a target object is not found in an expected location (e.g., Fridge), the Planner defaults back to the same location, creating a wasteful loop of: approach Fridge -> open -> find nothing -> wander -> approach Fridge again.

Existing defenses:
- **001 (searched-markers)**: Marks receptacles as SEARCHED in spatial memory, helping the Planner see that a location has been checked.
- **002a (intent-dedup)**: Blocks the same (intent, target) pair after 2-3 repeated incomplete attempts.
- **002b (critic-guard)**: Validates proposed intents pre-execution and rejects those targeting already-searched receptacles.

These are all **reactive**: they catch the problem AFTER the Planner has already fixated. Spike 004 takes a **proactive** approach: it generates a competing explore-biased intent BEFORE the fixation propagates.

## Mechanism

Instead of ONE Planner proposing intents, run TWO parallel Planner calls with DIFFERENT biases, then pick the better one:

1. **Planner A (Exploit)**: Standard Planner - proposes the most obvious next intent using `PLANNER_SYSTEM`.
2. **Planner B (Explore)**: Same context BUT with an added exploration bias (`EXPLORE_BIAS_SYSTEM`) - "You are in EXPLORATION MODE. Prioritize locations you have NOT yet visited. Avoid repeating recently targeted locations."
3. **Selector** (`EBAgent.select_intent`): Compares the two intents using prioritized heuristics:
   - If A matches a previously-tried-and-failed intent AND B does not -> pick B
   - If A's (intent, target) pair appears 2+ times in recent history -> pick B
   - If B's intent is identical to A's -> pick A (B added no diversity, save the compute)
   - If B targets a SEARCHED receptacle -> pick A (B is wasting time)
   - Default: prefer A (exploit is more efficient when not stuck)

### Gating Condition (Cost Optimization)

To avoid doubling API calls on every step, the contrastive planner only activates when:

```
Last 3 intent history entries ALL target the same location type
AND at least one of them is INCOMPLETE
```

This catches the fixation loop directly. Based on **Dynamic Self-Consistency (RASC, 2024)**: extra samples are only needed when the model is uncertain/stuck.

## Literature Grounding

### DiscussNav (Long et al., ICRA 2024)
*"Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions"*

Uses multiple LLM "experts" to discuss before making navigation decisions. Four expert types (instruction analysis, vision perception, completion estimation, decision testing) produce competing proposals, and a "Decision Testing Expert" selects the best one. Our approach simplifies this to two experts (exploit vs. explore) with a heuristic selector.

Key insight: **Diverse perspectives produce better navigation decisions.** The explore-vs-exploit tension forces diversity.

### Self-Consistency (Wang et al., 2022; extended 2024)
*"Self-Consistency Improves Chain of Thought Reasoning in Language Models"*

Samples multiple reasoning paths and selects the most consistent answer via majority voting. Extended in 2024 by:
- **RASC (Dynamic Self-Consistency)**: Early-stopping - only sample more when uncertainty is high. Our gating condition follows this principle.
- **Semantic Self-Consistency**: Uses embedding-based weighting rather than simple voting. Our heuristic selector is a lightweight analog.
- **Nash CoT**: Game-theoretic equilibrium between generation strategies. Our exploit-vs-explore is a simplified adversarial collaboration.

Key insight: **Diversity of reasoning paths matters more than quantity.** Two maximally-different biases (exploit vs. explore) cover more decision space than 5 similar samples.

### RoCo (Mandi et al., ICRA 2024)
*"RoCo: Dialectic Multi-Robot Collaboration with Large Language Models"*

Robots with LLMs discuss and collectively reason about task strategies before execution. The key finding: **debate-based planning outperforms single-agent planning** for complex spatial tasks.

## Implementation

### Files Modified

1. **`src/eb_agent.py`**:
   - Added `EXPLORE_BIAS_SYSTEM` (line ~456): System prompt for exploration mode
   - Added `contrastive_stats` dict to `EBAgent.__init__` (line ~515)
   - Added `EBAgent._should_activate_contrastive()` static method (line ~801): Gating condition
   - Added `EBAgent.plan_intent_explore()` method (line ~825): Explore-biased intent proposal
   - Added `EBAgent.select_intent()` static method (line ~960): Heuristic-based intent selection

2. **`src/branch_runner.py`**:
   - Added contrastive planner call site (line ~243): After standard `plan_intent()`, checks gating condition, calls `plan_intent_explore()`, runs `select_intent()`, logs the decision.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Only 2 parallel calls (not 3+) | Two maximally-different biases > three similar ones. Based on self-consistency literature showing diversity matters more than quantity. |
| Gating on "last 3 same target" | Catches fixation directly. RASC (2024) shows extra samples only needed when stuck. |
| Heuristic selector, not LLM judge | No additional API call. Based on DiscussNav's finding that simple rule-based fusion works well when the proposals are intentionally diverse. |
| Explore bias reuses same context | Saves token budget. Only the system prompt and a brief "avoid these" section differ from the standard call. |
| Still apply dedup to explore intents | Safety net: if even the explore-biased Planner fixates, the hard dedup mechanism kicks in. |
| Log to failure_log as `contrastive_selection` | Enables post-hoc analysis of how often B is chosen and why. |

## Metrics to Track

Stats collected per episode in `eb_agent.contrastive_stats`:

```python
{
    "activations": 0,    # how many times dual-planner was triggered
    "a_selected": 0,     # exploit (standard) chosen
    "b_selected": 0,     # explore chosen
    "selections": [      # per-selection details
        {
            "step": int,
            "chosen": "A" | "B",
            "a_intent": "approach Fridge (Fridge)",
            "b_intent": "approach CounterTop (CounterTop)",
            "reason": "A_intent_matches_known_failure"
        }
    ]
}
```

Key research questions:
1. **How often is B chosen over A?** (b_selected / activations ratio)
2. **Does B lead to finding the target faster?** (compare steps-to-completion for episodes with B selections vs. without)
3. **Which selection reason fires most?** (distribution of "reason" values)
4. **What's the API cost overhead?** (activations / total_steps ratio)

Also logged as JSONL events (`failure_type: "contrastive_selection"`) in the per-episode failures file for downstream analysis.

## Relationship to Other Spikes

| Spike | Relationship |
|-------|-------------|
| 001 (searched-markers) | Complementary: B uses SEARCHED markers from spatial memory to avoid known-empty receptacles. |
| 002a (intent-dedup) | Complementary: Dedup is the hard safety net; Contrastive is the soft prevention. |
| 002b (critic-guard) | Orthogonal: Critic validates AFTER proposal; Contrastive generates an alternative DURING proposal. |

## Known Limitations

1. **Still doubles API calls when gating fires.** For episodes where the Planner is frequently stuck, this could add significant latency. Consider using a cached/lighter model for one planner in future.
2. **Gating condition is heuristic.** May miss cases where 2 (not 3) repeated intents indicate fixation, or where the targets differ in name but are the same location type.
3. **Selector is rule-based, not learned.** A learned selector trained on episode data could outperform the heuristic rules.
4. **Explore bias may produce "scan room" too often.** If B always proposes scan room, the selector will pick A by default, wasting the extra API call.

## References

- Long, Y., Li, X., Cai, W., & Dong, H. (2024). Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions. *ICRA 2024*. arXiv:2309.11382
- Wang, X., Wei, J., Schuurmans, D., et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*. arXiv:2203.11171
- Wan, X., et al. (2024). Dynamic Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling. arXiv:2408.17017
- Mandi, Z., et al. (2024). RoCo: Dialectic Multi-Robot Collaboration with Large Language Models. *ICRA 2024*.
