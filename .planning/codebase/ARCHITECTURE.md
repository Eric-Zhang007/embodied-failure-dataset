# ARCHITECTURE.md — System Architecture

## High-Level Design

The system generates a dataset of embodied agent failures with counterfactual reasoning. It runs ALFRED household tasks in AI2-THOR, introduces failures via trap injection, and records the agent's attempts to recover — including what it should have done differently.

## Dual-Agent Four-Phase Loop

Every interaction step runs through Phase 1-4:

```
┌─────────────────────────────────────────────────────────┐
│                    BranchRunner.run()                    │
│                                                         │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌────────┐ │
│  │ Phase 1 │──▶│ Phase 2  │──▶│ Execute │──▶│Success?│ │
│  │  EB     │   │  Oracle  │   │ action  │   └───┬────┘ │
│  │ propose │   │  inject? │   │         │       │      │
│  └─────────┘   └──────────┘   └─────────┘   Yes│  No  │
│       ▲                                         │  │   │
│       │                                   ┌─────┘  │   │
│       │                                   │        ▼   │
│       │                              continue  ┌────────┐
│       │                                       │Phase 3 │  │
│       │                                       │  EB    │  │
│       │                                       │diagnose│  │
│       │                                       └───┬────┘  │
│       │                                           │      │
│       │                                      ┌────▼────┐ │
│       │                                      │Phase 4  │ │
│       └──────────────────────────────────────│ Oracle  │ │
│              next iteration                  │evaluate │ │
│                                              └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Phase 1 — EB Agent Proposes Action

- **File**: `src/eb_agent.py` → `EBAgent.propose_action()`
- **Input**: task goal, current screenshot, visible objects list, full history, hand status, task completion criteria, spatial memory
- **Output**: `{action, params, reasoning}`
- Prompt includes full action catalog (navigation, object interaction, task control)
- Validated against `_VALID_ACTIONS` set; invalid actions trigger retry (max 3)
- `LookAround` is a special case: executes a 4-direction scan, then calls `propose_action_lookaround()` with all 4 views

### Phase 2 — Oracle Injection Decision

- **File**: `src/oracle_agent.py` → `OracleAgent.decide_injection()`
- **Guard**: `cascade_level <= 1` AND remaining injections > 0
- Oracle sees full environment state and decides whether to inject a trap
- 6 injection methods available: `set_object_property`, `close_container`, `occlude_object`, `hide_object`, `swap_object`, `remove_object`
- Injection failures retry up to 3 times with different approaches
- Maximum 3 injections per episode
- Hard guard: never inject on task-critical objects (irreversible props blocked)

### Phase 3 — EB Failure Diagnosis

- **File**: `src/eb_agent.py` → `EBAgent.diagnose_failure()`
- Triggered when environment step returns `success: false`
- EB must: diagnose cause, propose recovery action, optionally provide counterfactual
- Counterfactual format: `{target_step, alternative_action, reasoning}`
- All text fields must use first-person perspective

### Phase 4 — Oracle Evaluation

- **File**: `src/oracle_agent.py` → `OracleAgent.evaluate_failure()`
- Evaluates: diagnosis correctness, counterfactual grade (WA/PA/AC), recovery verdict
- Decides whether to create a fork branch (only for AC-grade counterfactuals)
- Detects dead loops and unrecoverable states

## Information Permissions

| Information | EB Agent | Oracle Agent |
|-------------|----------|--------------|
| Current screenshot | ✓ | ✓ |
| Visible objects | ✓ | ✓ |
| Own action history | ✓ (EB fields only) | ✓ (all fields) |
| Oracle injection decisions | ✗ | ✓ |
| Fork existence | ✗ | ✓ |
| Trap existence | ✗ | ✓ |
| Cascade level | ✓ (via error context) | ✓ |

This is enforced by `context_builder.py`:
- `build_eb_history_context()` filters to EB-visible fields only
- `build_oracle_history_context()` includes all Oracle fields

## Branch/Fork Mechanism

```
main branch:  s0 → s1 → s2(fail) → s3(recover) → s4 → ...
                              │
                              └── fork_s2_main: s0(alt) → s1 → ...
```

- Fork created when counterfactual grade = AC and `enable_fork=True`
- Fork root step: Oracle rewrites reasoning to sound natural (agent doesn't know it's a fork)
- Fork shares parent context up to divergence point, then isolated
- `fork_manager.py` handles reasoning rewrite
- `scheduler.py` manages both main queue and fork queue (serial execution)

## Cascade Levels

| Level | Meaning | Phase 2 Behavior |
|-------|---------|-----------------|
| 0 | Normal execution | Oracle may inject |
| 1 | First failure, recovering | Oracle may inject (different trap) |
| 2+ | Multiple consecutive failures | Injection blocked |

## Key Architectural Decisions

1. **Dual-agent design**: EB and Oracle as separate agents with different information access, mirroring real-world supervisor-worker relationship
2. **VLM as both agents**: Same 32B model, different system prompts — no specialized models per role
3. **Trap-before-execution**: Oracle injects failures BEFORE EB's action executes, so EB encounters them naturally
4. **JSONL failure logging**: Separate from episode JSON — failures are events, steps are state
5. **Incremental flush**: Episode JSON written after every step for crash resilience
6. **Counterfactual at failure time**: EB proposes what it should have done; Oracle validates — not post-hoc analysis
