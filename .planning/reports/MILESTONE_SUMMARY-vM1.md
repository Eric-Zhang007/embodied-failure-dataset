# MILESTONE SUMMARY — vM1: System Stabilization & Semantic Memory

**Generated:** 2026-07-15 | **Status:** P1+P2 modules complete, Runtime-Debug complete  
**Next milestone:** vM2 — Scaled Data Generation (target: >30% E2E pass rate)

---

## 1. Overview

**What, Why, How, For whom**

Embodied Failure Dataset generates cascading failure + counterfactual reasoning datasets for embodied AI research. An LLM-driven agent (Planner + Executor + Oracle) runs ALFRED household tasks in AI2-THOR 5.0.0 simulation, with an Oracle Agent injecting environmental traps to create failure scenarios. The system records complete diagnostics, recovery attempts, and counterfactual annotations.

**Target users:** Embodied AI / robot learning researchers needing failure recovery training data and counterfactual reasoning benchmarks.

**M1 focused on:** Stabilizing the core Planner→Executor→Review→Execute loop, fixing critical runtime bugs (VLM reliability, scene restoration, parallel orchestration), and upgrading spatial memory from a flat geometric list to a receptacle-grouped semantic system with first-person perspective.

**Key outcomes:**
- `pick_and_place_simple`: `task_complete` in 6 steps (SoapBottle→Toilet) and 19 steps (Book→Desk)
- `pick_heat_then_place_in_recep`: partial progress (63 steps, Apple→Fridge navigation, object ID resolution issue identified)
- All VLM calls reliable via HTTP-200 hardening, reasoning_effort fallback, and provider migration
- Parallel scheduler per-worker isolation validated with 9 deterministic tests
- Semantic memory module live and rendering receptacle-grouped output
- 58-unit test suite passing

---

## 2. Architecture

**Planner→Executor→Review→Execute four-phase loop with Oracle supervision**

```
s0: Initial LookAround → 4-direction scan → VLM analysis → rotate to target
while True:
  Planner (gpt-5.5) → high-level intent
    ├── "scan room" → 4-view capture → VLM analysis → rotation
    ├── "Done" → task_conditions hard verification → complete or reject
    └── normal intent → Executor (gpt-5.5) → 1-5 action sequence with repeat
         → Planner Review → approve or corrected_actions
         → MoveSequence execution (sequential, stops on first failure)
         ├── success → continue
         └── failure → Phase 3 Planner diagnosis + Phase 4 Oracle evaluation
              ├── unrecoverable → terminate
              ├── recovered → continue
              └── recoverable → execute recovery → continue
```

### Core Modules (22 source files, ~4000 lines)

| Module | Lines | Role |
|--------|-------|------|
| `branch_runner.py` | 2390 | Main 4-phase loop, MoveSequence, recovery, resume |
| `eb_agent.py` | 1760 | Planner: plan_intent, review_actions, diagnose_failure |
| `executor.py` | 178 | Executor: intent → action sequence with repeat |
| `oracle_agent.py` | 288 | Oracle: Phase 2 injection + Phase 4 evaluation |
| `vlm_client.py` | 719 | VLM API calls, HTTP-200 validation, retry logic |
| `semantic_memory.py` | 403 | NEW: receptacle-grouped memory with task tracking |
| `geometric_memory.py` | 871 | REFACTORED: flat-list geometric memory (backward compat) |
| `memory_interface.py` | 93 | NEW: shared memory interface |
| `scheduler.py` | 325 | Per-worker agent isolation, parallel main + serial fork |
| `task_conditions.py` | 322 | 7 ALFRED task completion checkers |
| `action_adapter.py` | 123 | objectType→objectId resolution, param cleanup |
| `alfred_scene.py` | 192 | ALFRED scene restore, task state tracking |

### Simulation Stack
- **AI2-THOR 5.0.0**, gridSize=0.125m, visibilityDistance=100
- WSL2 Ubuntu + WSLg/Xvfb for rendering
- VLM: gpt-5.5 via www.9527code.com/v1 OpenAI-compatible API (reasoning_effort=medium)

### Memory Architecture (new in vM1)
```
memory_interface.py (base API)
├── geometric_memory.py  (flat list, "SPATIAL MEMORY — what you remember seeing")
└── semantic_memory.py   (receptacle groups, "=== WHAT I SEE === / === WHAT I REMEMBER ===")
```
Selectable via `--memory semantic` (default) or `--memory geometric`. First-person "I" perspective throughout all prompts.

---

## 3. What Changed Per Phase

### Phase 1: System Stability + Intent Memory ✅

| Change | Files |
|--------|-------|
| `_META_ACTIONS` blocks Done/LookAround in MoveSequence | branch_runner |
| `detect_dead_loop` supports nested param hashing | task_conditions |
| `analyze_scan_room` — 4-image VLM analysis → direction + intent | eb_agent |
| Intent-level history — Planner sees intent tree, Executor sees current intent | branch_runner, eb_agent |
| Executor `repeat` support — batch same-direction moves | executor |
| Executor SOLID obstacle labels — furniture marked "do NOT walk through" | executor |
| Recovery execution — Phase 3 recovery actions actually executed | branch_runner |
| `replay_steps` supports MoveSequence/LookAround replay | branch_runner |
| `resume` supports executor_agent + correct replay | branch_runner |

### Runtime-Debug Phase ✅ (this session)

| Task | What | Status |
|------|------|--------|
| T1 | Validate strict Executor/action boundaries | 35/35 tests |
| T2 | Real no-trap E2E | Exposed /v1 + restoration bugs |
| T3 | Fix ALFRED pose restoration | Type+nearest-position fallback |
| T4 | **HTTP-200 response hardening** | 5-stage validation, reasoning_effort auto-strip fallback, 400 retry, 12 tests |
| T5 | **Scheduler parallel isolation** | Per-worker `_create_agents()`, fixed accounting, 9 tests |
| T6 | **Action-path E2E probe** | task_complete (SoapBottle→Toilet, 6 steps) |

### Key runtime fixes:
- **Provider migration**: `api.fullcupai.com` → `www.9527code.com/v1` (eliminated intermittent 400 errors)
- **Reasoning_effort**: xhigh→medium default (xhigh multi-image returns content=None on gpt-5.5)
- **Camera horizon epsilon**: `30.0 + 30.0 = 60.00000000000001` exceeded boundary check, fixed with 0.01 epsilon
- **Object ID resolution**: Priority 4 fallback removed for PickupObject (prevents resolving to invisible objects inside containers)
- **4-direction scan metadata**: Previously only stored ahead-view objects in memory; now captures objects from all 4 scan directions

### P2: Semantic Memory (partial — core module complete)

| Change | Files |
|--------|-------|
| `memory_interface.py` — shared base API | NEW |
| `geometric_memory.py` — renamed from egocentric, `GeometricMemory(MemoryInterface)` | REFACTORED |
| `semantic_memory.py` — receptacle-grouped, freshness-aware, task-tracking, first-person "I" | NEW |
| `egocentric_memory.py` — backward-compat shim (`EgocentricMemory` → `GeometricMemory`) | REFACTORED |
| System prompts shifted to first-person: "I am an embodied agent" | eb_agent, executor |
| `--memory` CLI flag added to all scripts + SchedulerConfig | e2e_test, run_pipeline, resume_episode, scheduler |

---

## 4. Key Technical Decisions

1. **Planner+Executor split**: Planner handles high-level intent (WHAT), Executor handles concrete actions (HOW). Reduces 32B model call frequency. Both share same spatial memory context.

2. **Oracle-blinded architecture**: Oracle sees unified EB history only, unaware of Planner/Executor internal split. EB never knows fork/injection exist.

3. **Per-worker VLM isolation**: Each thread creates its own VLMClient + EBAgent + ExecutorAgent + OracleAgent via factory pattern. Prevents mutable state sharing across parallel workers.

4. **Semantic memory with receptacle grouping**: Objects grouped by parent receptacle (e.g., "On CounterTop: Tomato, Egg"). Target tracking shows "WHERE MY TARGET IS" with freshness hints. Disambiguation warns when a remembered target may be confused with a similar-looking visible object.

5. **First-person "I" perspective**: Consistent egocentric narrative — "I saw the Apple inside the Fridge", "I should MoveLeft". Replaces inconsistent "you"/3rd-person mix.

6. **Reasoning_effort auto-strip fallback**: When gpt-5.5 returns reasoning_content with null content, retry without reasoning_effort parameter (fallback to non-thinking mode).

7. **Four-directional scan metadata**: Objects visible from left/behind/right during LookAround are now recorded in memory, not just ahead view.

8. **Strict failure semantics**: Model/API contract errors are logged, retried (up to 5× validation + 3× transport = 15 total), then raised. Never become simulator failures or fallback actions.

---

## 5. Requirements Coverage

| Req | Description | Status |
|-----|-------------|--------|
| R1 | Execute precision (repeat, SOLID, LookAround/Done filtering, recovery) | ✅ |
| R2 | Spatial memory semanticization (parent receptacles, freshness, task tracking) | 🟡 Core module done; task-level integration pending |
| R3 | Fuzzy intent decomposition (locate X → approach <location>) | ⬜ Depends on R2 |
| R4 | Systematic exploration (quadrant search, scan→scan fix) | ⬜ Depends on R2 |
| R5 | scan→scan loop fix | ✅ Initial LookAround interception done |
| R6 | Fork validation | ⬜ Code exists; never E2E tested |

---

## 6. Known Gaps & Tech Debt

**Untested paths:**
- Fork mechanism never end-to-end tested (code exists but defaults to `enable_fork=False`)
- heat/cool/clean task types not E2E validated
- Phase 2 trap injection not exercised in recent runs

**Known bugs (carried forward):**
- 8B Executor under-counts repeat values and ignores SOLID labels (mitigated: now using 32B gpt-5.5 for all roles)
- Phase 3 Planner occasionally hallucinates task completion after navigation failure
- Object ID resolution for PickupObject can resolve to invisible objects inside containers (partially fixed: Priority 4 removed for PickupObject)

**Performance:**
- ~3-4 VLM calls per step (~$0.50-1.00 per episode at ~20 steps)
- Serial fork processing despite `max_parallel` for main branches

**Infrastructure:**
- No CI/CD pipeline
- No type checking (mypy/pyright)
- Hardcoded limits distributed across codebase (step limit, retry counts, aging thresholds)

---

## 7. Getting Started

### Quick start
```bash
# In WSL2 Ubuntu:
cd ~/embodied-failure-dataset
export PATH="$HOME/.local/bin:$PATH"

# Single-task E2E test (semantic memory, default)
uv run python scripts/e2e_test.py \
  --api-key sk-IxXjSiRZ3IhCd4hxMajweiXamF0R1U1cHmNECBrRbIz8v0hy \
  --task pick_and_place_simple --random --no-traps

# With geometric (legacy) memory
uv run python scripts/e2e_test.py ... --memory geometric

# Parallel pipeline (2 workers)
uv run python scripts/run_pipeline.py \
  --api-key sk-xxx --max 2 --parallel 2 --no-fork

# Resume a crashed episode
uv run python scripts/resume_episode.py <episode.json> --api-key sk-xxx --no-traps
```

### Prerequisites
- WSL2 with WSLg (for AI2-THOR GPU rendering)
- Python 3.10 via `uv`
- AI2-THOR 5.0.0
- API key for OpenAI-compatible endpoint (9527code.com)
- ALFRED json_2.1.0 data in `data/json_2.1.0/`

### Key files to read first
1. `CLAUDE.md` — environment setup, architecture overview
2. `src/branch_runner.py` — main orchestration (lines 141-900)
3. `src/semantic_memory.py` — new memory format
4. `src/vlm_client.py` — VLM client with retry/validation
5. `.planning/STATE.md` — project status
6. `.planning/ROADMAP.md` — remaining phases

---

## 8. Next Milestone Preview (vM2)

**Target:** Scaled data generation with >30% E2E pass rate

**Prerequisites from M1:**
- [ ] P2: Full semantic memory integration with task-aware Planner prompting
- [ ] P3: Fuzzy intent decomposition (leverage semantic memory for locate→approach)
- [ ] P4: Systematic exploration strategy + scan→scan fix
- [ ] P5: Fork validation with E2E tests
- [ ] P6: 7-task baseline (5 episodes each)

---

*Milestone summary: 2026-07-15*
