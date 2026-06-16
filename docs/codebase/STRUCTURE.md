# Codebase Structure
**Analysis Date:** 2026-06-16

## Directory Layout

```
embodied-failure-dataset/
│
├── pyproject.toml                  # Project metadata, deps (ai2thor==5.0.0), hatchling build
├── uv.lock                         # Locked dependency versions (uv)
│
├── data/
│   └── failure_type_library.json   # Catalog of injectable failure types with placeholders
│
├── src/                            # Core library (18 .py files)
│   ├── scheduler.py                # Global orchestrator: episode queue, fork queue, parallel execution
│   ├── branch_runner.py            # Phase 1-4 loop engine for a single branch (~1500 lines)
│   ├── eb_agent.py                 # Embodied Agent: Planner (intents, review, scan), failure diagnosis
│   ├── oracle_agent.py             # Oracle Agent: injection decisions (Phase 2), failure evaluation (Phase 4)
│   ├── executor.py                 # Low-level action Executor (8B model): intent → action sequence
│   ├── env_controller.py           # AI2-THOR wrapper: scene reset, step, state snapshot, Xvfb
│   ├── env_injector.py             # Environment mutation: set property, swap, occlude, hide, remove
│   ├── episode_manager.py          # Per-episode JSON persistence with incremental auto-flush
│   ├── step_recorder.py            # Build step dict entries, save PNG screenshots
│   ├── context_builder.py          # Compact JSONL history rendering for EB/Oracle prompts
│   ├── action_adapter.py           # VLM→AI2-THOR action translation: type→ID resolution, param cleaning
│   ├── task_conditions.py          # Hard-coded task completion checkers, dead-loop detection
│   ├── trap_planner.py             # Selects initial traps from failure_type_library.json
│   ├── fork_manager.py             # Rewrites EB reasoning for fork branch self-consistency
│   ├── egocentric_memory.py        # Compressed spatial memory of all ever-seen objects
│   ├── alfred_parser.py            # ALFRED traj_data.json parser: metadata, scene state, low actions
│   ├── alfred_scene.py             # ALFRED scene restoration: object poses, toggles, init action
│   └── vlm_client.py               # VLM API client: SiliconFlow, Ollama, OpenRouter; multi-image chat
│
├── scripts/                        # Entry points and utilities (10 .py files)
│   ├── run_pipeline.py             # ★ Main CLI: args → SchedulerConfig → Scheduler.run()
│   ├── e2e_test.py                 # End-to-end test harness
│   ├── e2e_test_output.py          # E2E test output utilities
│   ├── replay_demo.py              # Replay recorded episodes with screenshots
│   ├── alfred_replay.py            # Replay raw ALFRED expert trajectories
│   ├── bench_api.py                # VLM API latency/throughput benchmark
│   ├── generate_ft_data.py         # Convert episode JSON → fine-tuning data
│   ├── finetune_qwen3vl.py         # Fine-tune Qwen3-VL on generated data
│   ├── download_alfred.py          # Download ALFRED json_2.1.0 dataset
│   └── regenerate_screenshots.py   # Rebuild PNGs from existing episode JSON
│
├── tests/                          # Unit and integration tests (8 .py files)
│   ├── test_eb_agent_prompts.py    # EB Agent prompt construction tests
│   ├── test_vlm_client.py          # VLM API client tests
│   ├── test_context_builder.py     # Context builder rendering tests
│   ├── test_env_injector.py        # Environment injection method tests
│   ├── test_alfred_scene.py        # ALFRED scene restoration tests
│   ├── test_task_conditions_alfred.py  # Task completion checker tests
│   ├── test_error_handling.py      # Error handling and edge case tests
│   └── test_training_prompt.py     # Training data prompt format tests
│
└── docs/
    └── codebase/
        ├── ARCHITECTURE.md         # This file's sibling — architecture analysis
        └── STRUCTURE.md            # This file — directory layout and conventions
```

## Key Locations

### Where to Start Reading

| What | Path |
|------|------|
| **Main entry point** | `scripts/run_pipeline.py` |
| **Pipeline orchestrator** | `src/scheduler.py` |
| **Core agent loop** | `src/branch_runner.py` (method: `BranchRunner.run()`) |
| **Planner agent** | `src/eb_agent.py` (methods: `plan_intent()`, `diagnose_failure()`, `analyze_scan_room()`) |
| **Oracle supervisor** | `src/oracle_agent.py` (methods: `decide_injection()`, `evaluate_failure()`) |
| **Action executor** | `src/executor.py` (method: `ExecutorAgent.execute_intent()`) |
| **Environment control** | `src/env_controller.py` (class: `EnvController`) |
| **VLM communication** | `src/vlm_client.py` (class: `VLMClient`) |

### Where to Add New Code

| What you want to add | Where to put it |
|----------------------|-----------------|
| **New task type support** | `src/task_conditions.py` — add checker function + register in `_CHECKERS` and `_CRITERIA_RULES` |
| **New failure injection type** | `data/failure_type_library.json` — add entry; `src/env_injector.py` — add handler function + register in `_INJECTORS` |
| **New agent capability** | `src/eb_agent.py` — add method on `EBAgent`; `src/branch_runner.py` — wire into the Phase 1-4 loop |
| **New VLM backend** | `src/vlm_client.py` — add backend in `__init__` and `_call_api` |
| **New CLI script** | `scripts/` — add `your_script.py`, import from `src/` |
| **New environment action** | `src/action_adapter.py` — add to `_VALID_ACTIONS` in `branch_runner.py`, add adapter if needed |
| **New test** | `tests/` — name `test_<module>.py`, use pytest |
| **New prompt strategy** | `src/eb_agent.py` or `src/oracle_agent.py` — modify the `*_SYSTEM` string constants and `build_*_prompt()` functions |
| **New spatial memory feature** | `src/egocentric_memory.py` — extend `EgocentricMemory.update()` and `render()` |

## Naming Conventions

### Files
- **Source modules:** lowercase with underscores (`eb_agent.py`, `env_controller.py`)
- **Scripts:** lowercase with underscores, descriptive verb-noun (`run_pipeline.py`, `generate_ft_data.py`)
- **Tests:** `test_<module_name>.py`

### Classes
- **PascalCase:** `EBAgent`, `OracleAgent`, `ExecutorAgent`, `EnvController`, `EpisodeManager`, `BranchRunner`, `Scheduler`
- **Config dataclasses** suffix `Config`: `SchedulerConfig`, `BranchConfig`
- **Result dataclasses** suffix `Result`: `BranchResult`

### Functions
- **Public API methods:** `snake_case` verbs (`plan_intent()`, `decide_injection()`, `diagnose_failure()`)
- **Internal helpers:** prefixed with `_` (`_build_step_entry()`, `_execute_move_sequence()`, `_resolve()`)
- **Module-level free functions:** `snake_case` (`run_single_branch()`, `replay_steps()`, `build_phase1_prompt()`)

### Episode Data Fields
- **Step IDs:** `s{index}` (e.g., `s0`, `s1`, `s23`)
- **Branch IDs:** `main` for primary; `fork_s{step}_{parent_branch}` for forks (e.g., `fork_s5_main`)
- **Trap IDs:** `trap_{i}` for initial traps; `trap_{branch}_{step}` for runtime traps

### Prompt System Messages
- Capitalized module-level constants: `PHASE1_SYSTEM`, `PHASE2_SYSTEM`, `PHASE3_SYSTEM`, `PHASE4_SYSTEM`, `PLANNER_SYSTEM`, `PLANNER_REVIEW_SYSTEM`, `EXECUTOR_SYSTEM`, `SCAN_ANALYSIS_SYSTEM`

## Data Formats

### ALFRED Input
- **Location:** `data/json_2.1.0/{split}/{task_id}/traj_data.json`
- **Format:** ALFRED 2.1.0 trajectory with `task_type`, `pddl_params`, `scene`, `plan.low_actions`

### Episode Output
- **Location:** `output_{timestamp}/{episode_id}.json`
- **Schema:** top-level keys: `episode_id`, `task_goal`, `scene`, `task_type`, `alfred_task_type`, `alfred_task_id`, `pddl_params`, `alfred_scene`, `initial_traps[]`, `runtime_traps[]`, `steps[]`, `final_outcome`
- **Step entry keys:** `step_id`, `branch_id`, `parent_step_id`, `step_index_in_branch`, `action`, `action_params`, `success`, `error_message`, `image_path`, `eb_reasoning`, `eb_diagnosis`, `eb_recovery_reasoning`, `eb_counterfactual`, `eb_proposed_recovery_action`, `oracle_injection_decision`, `oracle_diagnosis_correct`, `oracle_ground_truth`, `oracle_counterfactual_grade`, `oracle_counterfactual_gold`, `oracle_recovery_verdict`, `fork_decision`, `fork_metadata`

### Screenshots
- **Location:** `output_{timestamp}/{episode_id}/s{step_index}.png`
- **LookAround views:** `s{step_index}_look_{ahead|left|behind|right}.png`

### Failure Logs
- **Location:** `output_{timestamp}/{episode_id}/failures_{branch_id}.jsonl`
- **Format:** One JSON object per line; fields vary by `failure_type`

### Failure Type Library
- **Location:** `data/failure_type_library.json`
- **Entry schema:** `failure_type`, `category`, `description`, `injection` (with `<target>`, `<appliance>`, `<container>`, `<receptacle>`, `<tool>`, `<lamp>` placeholders), `applicable_tasks[]`, `severity`

## Dependencies
- **Runtime:** `ai2thor==5.0.0` (3D simulation), `numpy`, `Pillow`, `requests`
- **Optional:** `openai` (for SiliconFlow API), `torch`, `transformers`, `qwen-vl-utils` (for fine-tuning)
- **Build:** `hatchling`

## Development Workflow

```
# Install deps
uv sync

# Run pipeline
uv run python scripts/run_pipeline.py --max 5

# Run tests
uv run python -m pytest tests/ -v

# Run single test
uv run python -m pytest tests/test_env_injector.py -v
```
