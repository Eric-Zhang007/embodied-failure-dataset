# STRUCTURE.md — Codebase Structure

## Directory Tree

```
embodied-failure-dataset/
├── pyproject.toml                  # Project config, single dependency (ai2thor)
├── CLAUDE.md                       # Project instructions for Claude Code
├── check_task.py                   # Ad-hoc task checking script
│
├── src/                            # Core library (19 files)
│   ├── __init__.py                 # Empty
│   ├── vlm_client.py              # VLM API client (500 lines)
│   ├── eb_agent.py                # EB Agent: Phase 1 & 3 (441 lines)
│   ├── oracle_agent.py            # Oracle Agent: Phase 2 & 4 (286 lines)
│   ├── branch_runner.py           # Main loop + run_single_branch() (944 lines) ★ largest
│   ├── env_controller.py          # AI2-THOR wrapper + Xvfb (108 lines)
│   ├── env_injector.py            # 6 injection methods (217 lines)
│   ├── scheduler.py               # Global queue + fork scheduling (262 lines)
│   ├── episode_manager.py         # Episode JSON I/O (82 lines)
│   ├── context_builder.py         # History formatting for prompts (125 lines)
│   ├── egocentric_memory.py       # Compressed spatial memory (513 lines)
│   ├── task_conditions.py         # 7 task completion checkers (312 lines)
│   ├── trap_planner.py            # Failure type selection (147 lines)
│   ├── action_adapter.py          # objectType→objectId resolution (123 lines)
│   ├── step_recorder.py           # Step standardization + screenshot (53 lines)
│   ├── fork_manager.py            # Counterfactual reasoning rewrite (56 lines)
│   ├── alfred_parser.py           # ALFRED traj_data.json parser (148 lines)
│   ├── alfred_scene.py            # ALFRED scene restore utilities (192 lines)
│   └── executor.py                # Action execution helper (111 lines)
│
├── scripts/                        # Entry points (10 files)
│   ├── e2e_test.py                # Single-trajectory E2E test (83 lines)
│   ├── e2e_test_output.py         # E2E test with fixed output dir (104 lines)
│   ├── run_pipeline.py            # Batch pipeline runner (48 lines)
│   ├── bench_api.py               # VLM API latency benchmark (135 lines)
│   ├── download_alfred.py         # ALFRED data downloader (43 lines)
│   ├── alfred_replay.py           # ALFRED expert trajectory replay (131 lines)
│   ├── replay_demo.py             # Demo replay script (95 lines)
│   ├── generate_ft_data.py        # Fine-tuning data generation (164 lines)
│   ├── finetune_qwen3vl.py        # Qwen3-VL fine-tuning script (362 lines)
│   └── regenerate_screenshots.py  # Screenshot regeneration (135 lines)
│
├── data/
│   ├── failure_type_library.json  # 8 hand-crafted failure types
│   └── json_2.1.0/               # ALFRED dataset (train/valid_seen/valid_unseen)
│       └── {split}/{task}/trial_*/traj_data.json
│
├── logs/                          # Runtime logs (gitignored)
│   ├── api_calls.jsonl           # Full API exchange log
│   └── api_calls.log             # Human-readable API summary
│
├── output/                        # Episode outputs (gitignored)
│   └── {episode_id}/
│       ├── {episode_id}.json     # Full episode data
│       ├── s*.png                # Step screenshots
│       └── failures_*.jsonl      # Failure event logs
│
└── .planning/                     # GSD planning artifacts
    └── codebase/                  # Codebase map (this document set)
```

## Module Dependency Graph

```
scripts/e2e_test.py ─────┐
scripts/run_pipeline.py ─┤
                          ▼
                  src/branch_runner.py (run_single_branch)
                     │
         ┌───────────┼───────────────┐
         ▼           ▼               ▼
   eb_agent.py  oracle_agent.py  env_controller.py
         │           │               │
         └─────┬─────┘               │
               ▼                     ▼
         vlm_client.py        env_injector.py
               │                     │
               ▼                     ▼
         (SiliconFlow API)    (AI2-THOR Controller)

Supporting modules:
  context_builder.py  ← used by eb_agent, oracle_agent
  egocentric_memory.py ← used by branch_runner, eb_agent
  task_conditions.py  ← used by branch_runner
  episode_manager.py  ← used by branch_runner, scheduler
  action_adapter.py   ← used by branch_runner
  trap_planner.py     ← used by branch_runner (via run_single_branch)
  alfred_parser.py    ← used by branch_runner, scheduler
  fork_manager.py     ← used by scheduler
  step_recorder.py    ← used by branch_runner, scheduler
  alfred_scene.py     ← used by env_controller
  executor.py         ← used by scripts (ft data generation)
```

## File Size Breakdown

| Size | Files |
|------|-------|
| 900+ lines | `branch_runner.py` (944) — the core loop |
| 400-600 | `egocentric_memory.py` (513), `vlm_client.py` (500), `eb_agent.py` (441) |
| 200-400 | `finetune_qwen3vl.py` (362), `task_conditions.py` (312), `oracle_agent.py` (286), `scheduler.py` (262), `env_injector.py` (217) |
| 100-200 | `alfred_scene.py` (192), `generate_ft_data.py` (164), `alfred_parser.py` (148), `trap_planner.py` (147), others |
| <100 | `step_recorder.py` (53), `fork_manager.py` (56), `episode_manager.py` (82), etc. |

## Entry Points

| Script | Purpose | Key Args |
|--------|---------|----------|
| `e2e_test.py` | Single trajectory test | `--api-key`, `--task`, `--no-traps`, `--enable-fork` |
| `run_pipeline.py` | Batch processing | `--max`, `--api-key`, `--task`, `--no-fork` |
| `bench_api.py` | API latency benchmark | `--api-key`, image count/size |
| `download_alfred.py` | ALFRED data download | (none) |
| `generate_ft_data.py` | Fine-tuning dataset generation | output dir |
| `alfred_replay.py` | Expert trajectory replay | traj_data.json path |
