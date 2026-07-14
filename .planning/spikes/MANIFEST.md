# Spike Manifest — Navigation Strategy Experiments

## Idea

Systematically experiment with navigation and exploration strategies to solve the Planner's "fixation loop" problem: the Planner repeatedly chooses "approach Fridge" even after opening it and confirming the target object is not inside. The goal is to find which strategies (or combination) most effectively prevent redundant exploration and enable efficient target discovery.

## Problem Statement

Given an embodied agent in AI2-THOR with Planner+Executor+Oracle architecture:
- The Planner sees full intent history and spatial memory
- When a target object is not found in an expected location (e.g., Fridge), the Planner defaults back to the same location
- This creates a wasteful loop of: approach Fridge → open → find nothing → wander → approach Fridge again
- The agent eventually recovers but wastes 10-20+ steps

## Requirements

- Solutions must work within the existing Planner/Executor/Oracle 3-model architecture
- No additional LLM calls that would significantly increase latency (critic-guard may use text-only call)
- Must handle all 7 ALFRED task types, not just heat
- Must work across all 30 FloorPlan scenes
- Spatial memory modifications should be backward-compatible

## Spikes

| # | Name | Type | Validates | Status | Verdict | Tags |
|---|------|------|-----------|--------|---------|------|
| 001 | searched-markers | standard | Given searched-and-empty receptacle, when Planner proposes next intent, then it no longer chooses that receptacle | ✅ complete | VALIDATED | navigation, spatial-memory, planner |
| 002a | intent-dedup | comparison | Given N repeated failed intents, when Planner proposes same again, then system blocks and forces exploration | ✅ complete | VALIDATED | navigation, planner, dedup |
| 002b | critic-guard | comparison | Given proposed intent targeting already-searched location, when Critic checks it, then intent is rejected pre-execution | ✅ complete | VALIDATED | navigation, critic, validation |
| 003 | curiosity-scoreboard | standard | Given 3D curiosity scores (semantic+novelty+discovery), when Planner sees ranked scoreboard, then it prioritizes high-score unexplored locations | ✅ complete | VALIDATED | navigation, proactive, curiosity, scoring |
| 004 | contrastive-planner | standard | Given two Planner instances (exploit vs explore), when both propose intents, then selector picks the non-repeated one | ✅ complete | VALIDATED | navigation, multi-agent, explore-exploit |

## Bugfixes (This Session)

| # | Bug | File | Status |
|---|-----|------|--------|
| B1 | 5xx/524 API errors not retried | `src/vlm_client.py` | ✅ fixed |
| B2 | AABB surface distance (center vs collision face) | `src/eb_agent.py`, `src/executor.py`, `src/egocentric_memory.py` | ✅ fixed |
| B3 | CoT/reasoning_content not saved in API logs | `src/vlm_client.py` | ✅ fixed |
| B4 | Sliced naming bug ("TomatoSliced" in criteria) | `src/task_conditions.py` | ✅ fixed |
| B5 | Intent history missing Phase 3 diagnosis | `src/branch_runner.py`, `src/eb_agent.py` | ✅ fixed |
| B6 | Incomplete criteria text (missing slice prerequisite) | `src/task_conditions.py` | ✅ fixed |
| B7 | Stale spatial memory position after movement | `src/branch_runner.py` | ✅ fixed (3 call sites, 4 paths verified) |
| B8 | Broken failed_object_ids for MoveSequence | `src/branch_runner.py` | ✅ fixed (returns resolved params from _execute_move_sequence) |

## Audit Findings (from context-system review)

14 issues found. Top severity:

| # | Severity | Issue |
|---|----------|-------|
| A1 | 🔴 Bug | Stale spatial memory after MoveSequence — `memory.update()` uses pre-move metadata |
| A2 | 🔴 Bug | `failed_object_ids` never populated for MoveSequence failures |
| A3 | 🔴 Bug | Criteria text missing slice/heat/cool/clean prerequisites → B6 |
| A4 | 🔴 Bug | Initial LookAround doesn't set scan cooldown |
| A5 | 🟡 | Planner `plan_intent` doesn't see `failed_object_ids` |
| A6 | 🟡 | Phase 3 diagnosis prompt missing `failed_object_ids` |
| A7 | 🟡 | Executor `last_error` may be from different intent |
| A8 | 🟡 | "Done" intent silently bypasses `intent_history` |
| A9 | 🟡 | Duplicate HAND STATUS in Executor prompt |
| A10 | 🟡 | `build_phase1_prompt` suppresses history when memory available |
| A11 | 🟢 | Exception during env.step() loses episode data |
| A12 | 🟢 | Dead-loop threshold allows interleaved failures |
| A13 | 🟢 | `completed` flag semantics too coarse |
| A14 | 🟢 | Memory directions stale for remembered objects |

## Research Pool (Ideas to Explore Next)

### From Literature
1. **Frontier-based exploration** (Prune-Then-Plan): Prune implausible navigation targets before planning, leaving only unexplored frontiers
2. **Hypothesis graph** (HGR): Represent locations as revisable hypothesis nodes; when observation mismatches prediction, prune downstream
3. **Semantic scoring** (SPIRAL Critic): Assign scores to candidate locations based on semantic priors (where is X usually found?)
4. **Hierarchical backtracking** (SDPA): Build task tree; when leaf fails, backtrack to nearest alternative branch

| 005 | progress-gating | prompt | Given recovery phase after collision, when Planner proposes intent, then prompt includes phase-specific anti-fixation rules | ✅ complete | VALIDATED | prompt, phase-detection, guardrails |
| 006 | search-trail-cost | memory | Given agent has visited Fridge area 3+ times, when Planner scores candidate locations, then revisit count penalizes repeated areas | ✅ complete | VALIDATED | memory, trail, revisit-penalty |
| 007 | room-semantic-priors | heuristic | Given kitchen scene + unseen Apple, when Planner explores, then prompt shows Apple→CounterTop(high), Apple→Fridge(medium) priors | ⏳ queued | PENDING | heuristic, semantics, priors |
| 008 | few-shot-failure-injection | learning | Given fixation pattern detected, when Planner proposes intent, then curated anti-fixation example injected as few-shot demo | ⏳ queued | PENDING | learning, few-shot, in-context |

### From Agent Dialectic
- **research-agent (a81288)**: Proposed 5 ideas (search-agenda already spiked as 003; progress-gating, search-trail-cost, room-semantic-priors, few-shot-failure-injection queued). Highest-ROI combo: agenda + gating + trail — proactive planning + context-aware constraints + empirical history attacking fixation from 3 angles.

## Critique Findings (from cross-validation agent a54d55)

15 blind spots, 3 interaction risks, 10 edge cases found across 001/002a/002b. 12 refinements proposed.

### Critical Issues to Fix

| # | Issue | Affects | Severity |
|---|-------|---------|----------|
| C1 | Searched markers are type-level, not instance-level (Fridge1=Fridge2) | 001 | ✅ fixed — instance-level via objectId extraction |
| C2 | Occluded target false-positive — Fridge permanently marked searched | 001 | ✅ fixed — unmark methods + occlusion detection + explicit UNMARK intent |
| C3 | pick_two tasks broken — can't revisit CounterTop for second Apple | 001+002a | ✅ fixed — parent_target never blocked, pick_two-aware fallback |
| C4 | Dedup 5-entry window too short — covers <20% of long episodes | 002a | ⏳ queued |
| C5 | Dedup exact string match bypassed by synonyms ("Refrigerator" vs "Fridge") | 002a | ✅ fixed — verb normalization + canonical type matching via spatial memory |
| C6 | Task-agnostic fallbacks — "approach Window" for food task | 002a | ✅ fixed — task-aware priority (parent_target > unvisited > visible) |
| C7 | LLM critic dead code — `chat_text_json()` doesn't exist | 002b | 🟢 false alarm — method already exists at vlm_client.py:226 |
| B13 | Regen loop wastes call on second rejection | 002b | ✅ fixed — forces "explore area" fallback on second reject |
| C8 | Search-Dedup-Critic cascade deadlock (all 3 trigger on same target) | all | 🟡 |
| C9 | Floor marked as searched — blocks all floor interaction | 001 | 🟡 |
| C10 | Fork inherits nothing — clean slate repeats same failures | all | 🟢 |

### Refinements Queued
- R1: Evidence-gated search marking (require actual interior inspection)
- R2: Instance-level marking via objectId
- R3: Unmark pathway via Planner Phase 3 diagnosis
- R4: Exclude task-target receptacles from marking
- R5: Semantic normalization (Fridge↔Refrigerator)
- R6: Window reset on successful different intent
- R7: Task-aware fallback generation
- R8: Track fallback outcomes
- R9: Unify tracking into single source of truth
- R10: Fix or remove LLM critic path
- R11: Feed critic outcomes to Oracle
- R12: Graduated response to regen failures
