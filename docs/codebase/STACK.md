# Technology Stack
**Analysis Date:** 2026-06-16

## Overview

embodied-failure-dataset is a Python project that generates a dataset of embodied agent cascading failures and counterfactual reasoning traces. It runs a Planner→Executor→Review agent cycle inside AI2-THOR 3D household environments, with an external Oracle Agent injecting traps and evaluating recovery.

## Languages

| Language | Version       | Usage                     |
|----------|---------------|---------------------------|
| Python   | >= 3.10       | Entire codebase           |

## Package Manager & Build System

- **Package manager:** `uv` (lock file at `uv.lock`, created 2026)
- **Build system:** `hatchling` (PEP 517 backend)
- **Source layout:** `src/` package (flat, no namespace packages)
- **No virtual environment:** uv creates .venv automatically

Configuration from `pyproject.toml`:
```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
requires-python = ">=3.10"
dependencies = [
    "ai2thor==5.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

## Runtime Environment

- **Target OS:** Linux (WSL2 required, see script comments: "WSL2 中")
- **Graphics:** Xvfb virtual framebuffer for headless AI2-THOR rendering (display :99)
- **WSLg support:** Auto-detects XWayland at :0 or :0.0 using `xdpyinfo`
- **Shell:** POSIX (bash/git-bash/MSYS on Windows host)

## Key Dependencies

### Direct

| Package     | Version   | Purpose                              |
|-------------|-----------|--------------------------------------|
| `ai2thor`   | 5.0.0     | 3D household simulator (Unity-based) |

All other dependencies are transitive through ai2thor.

### Transitive (from uv.lock)

| Package              | Version    | Brought in by     | Purpose                    |
|----------------------|------------|--------------------|----------------------------|
| `numpy`              | 2.2.6/2.4.6| ai2thor           | Numerical arrays, images  |
| `pillow`             | (trans)    | ai2thor           | Image I/O (PNG encode)    |
| `opencv-python`      | (trans)    | ai2thor           | Computer vision           |
| `flask`              | (trans)    | ai2thor           | Web server (THOR backend) |
| `werkzeug`           | (trans)    | ai2thor           | WSGI (Flask dependency)   |
| `requests`           | (trans)    | ai2thor           | HTTP client               |
| `botocore`           | (trans)    | ai2thor           | AWS SDK (S3 downloads)    |
| `msgpack`            | (trans)    | ai2thor           | Binary serialization      |
| `pyyaml`             | (trans)    | ai2thor           | YAML parsing              |
| `python-xlib`        | (trans)    | ai2thor           | X11 display management    |
| `progressbar2`       | (trans)    | ai2thor           | Progress bars             |
| `aws-requests-auth`  | (trans)    | ai2thor           | AWS SigV4 auth            |

### Runtime-only (lazy imports, not in pyproject.toml)

| Package    | Imported lazily via          | Used in files                        |
|------------|------------------------------|--------------------------------------|
| `requests` | `import requests` inside methods | `vlm_client.py` (both OpenAI and Ollama backends) |

**Note:** `requests` is NOT declared in pyproject.toml but is imported at runtime. It works because ai2thor transitively depends on it.

## Core Frameworks & Libraries (by actual imports)

### Standard Library
```
import json, os, base64, io, time, logging    (vlm_client.py)
import subprocess, atexit                     (env_controller.py)
import math                                   (eb_agent.py, alfred_scene.py, egocentric_memory.py, env_injector.py)
import threading                              (scheduler.py)
import glob                                   (scheduler.py)
import re                                     (alfred_parser.py, egocentric_memory.py)
import random                                 (trap_planner.py)
import argparse, sys                          (run_pipeline.py, bench_api.py, download_alfred.py, e2e_test.py, finetune_qwen3vl.py, generate_ft_data.py)
from pathlib import Path                      (vlm_client.py, finetune_qwen3vl.py)
from dataclasses import dataclass, field      (branch_runner.py, egocentric_memory.py, scheduler.py)
from collections import OrderedDict, deque    (egocentric_memory.py, scheduler.py)
from concurrent.futures import ThreadPoolExecutor, as_completed  (scheduler.py, e2e_test.py, generate_ft_data.py)
from typing import Optional                   (vlm_client.py, trap_planner.py, branch_runner.py)
from __future__ import annotations            (context_builder.py, executor.py, egocentric_memory.py)
```

### Third-Party Libraries
```python
import numpy as np                            (vlm_client.py, env_controller.py, eb_agent.py, executor.py, oracle_agent.py, step_recorder.py, bench_api.py, generate_ft_data.py, fork_manager.py)
from PIL import Image                         (vlm_client.py, step_recorder.py, bench_api.py, generate_ft_data.py)
from ai2thor.controller import Controller     (env_controller.py, env_injector.py, trap_planner.py)
```

### Internal Imports (src.*)
```python
from src.vlm_client import VLMClient          (eb_agent.py, scheduler.py, oracle_agent.py, run_pipeline.py, bench_api.py, generate_ft_data.py, fork_manager.py, e2e_test.py)
from src.eb_agent import EBAgent              (scheduler.py, branch_runner.py, e2e_test.py)
from src.oracle_agent import OracleAgent      (scheduler.py, branch_runner.py, e2e_test.py)
from src.executor import ExecutorAgent        (scheduler.py, e2e_test.py)
from src.branch_runner import BranchRunner, BranchConfig, BranchResult, run_single_branch  (scheduler.py, e2e_test.py)
from src.env_controller import EnvController  (scheduler.py, branch_runner.py)
from src.episode_manager import EpisodeManager (scheduler.py, branch_runner.py)
from src.fork_manager import ForkManager      (scheduler.py)
from src.step_recorder import StepRecorder    (scheduler.py, branch_runner.py)
from src.alfred_parser import load_traj, extract_metadata, extract_low_actions  (scheduler.py)
from src.alfred_scene import ...              (env_controller.py)
from src.context_builder import ...           (eb_agent.py, executor.py, oracle_agent.py, branch_runner.py, fork_manager.py, finetune_qwen3vl.py, generate_ft_data.py)
from src.action_adapter import adapt, resolve_object_ids, clean_history_params  (branch_runner.py, alfred_scene.py, context_builder.py)
from src.env_injector import inject           (branch_runner.py, trap_planner.py)
from src.task_conditions import ...           (branch_runner.py)
from src.egocentric_memory import EgocentricMemory  (branch_runner.py)
from src.trap_planner import TrapPlanner      (e2e_test.py)
```

## File Formats

| Format | Extension     | Usage                                        |
|--------|---------------|----------------------------------------------|
| JSON   | `.json`       | Episode output, traj_data.json (ALFRED), failure_type_library.json |
| JSONL  | `.jsonl`      | API call logs (api_calls.jsonl), failure logs (failures_*.jsonl) |
| PNG    | `.png`        | Step screenshots (encoded from numpy arrays via PIL) |
| TOML   | `.toml`       | Project configuration (pyproject.toml)       |
| 7z     | `.7z`         | ALFRED data archive (downloaded from AWS S3) |
| YAML   | `.yaml`       | LLaMA-Factory fine-tuning config             |

## Logging & Observability

- **Python logging:** `logging.getLogger("vlm_client")` with DEBUG to file, INFO to console
- **File logs:** `logs/api_calls.log` (all API exchanges), `logs/failure_*.json` (failure dumps)
- **Per-episode logs:** `output/<ep_id>/failures_*.jsonl`, `output/<ep_id>/api_calls.jsonl`
- **Environment variables:** `EFD_LOG_FULL_API` (enable/disable full API logging)

## Testing

- **Framework:** pytest (inferred from test file naming convention `tests/test_*.py`)
- **Test files:**
  - `tests/test_vlm_client.py`
  - `tests/test_eb_agent_prompts.py`
  - `tests/test_env_injector.py`
  - `tests/test_task_conditions_alfred.py`
  - `tests/test_context_builder.py`
  - `tests/test_error_handling.py`
  - `tests/test_alfred_scene.py`
  - `tests/test_training_prompt.py`

## Source Modules (17 modules in src/)

| Module               | Lines | Purpose                                                  |
|----------------------|-------|----------------------------------------------------------|
| `vlm_client.py`      | 590   | Multi-backend VLM client (OpenAI/Ollama/OpenRouter/SiliconFlow) |
| `branch_runner.py`   | 1503  | Phase 1-4 execution loop, Planner→Executor→Review cycle  |
| `eb_agent.py`        | 759   | EB Agent: Phase 1 action proposal, Phase 3 failure diagnosis, Planner, Reviewer |
| `egocentric_memory.py`| 410  | Spatial memory: object tracking, obstacle tracking, area labels |
| `oracle_agent.py`    | 288   | Oracle Agent: Phase 2 trap injection decisions, Phase 4 evaluation & fork |
| `task_conditions.py` | 328   | Task completion checkers, dead-loop detection, unrecoverable detection |
| `scheduler.py`       | 309   | Global scheduler: parallel episode dispatch, fork queue |
| `env_injector.py`    | 217   | AI2-THOR environment mutators (6 injection methods) |
| `alfred_scene.py`    | 192   | ALFRED scene restoration, object pose merging, task state tracking |
| `alfred_parser.py`   | 188   | ALFRED traj_data.json parser, PDDL→task goal generation |
| `executor.py`        | 177   | Executor Agent: low-level action chunk generation (8B model) |
| `trap_planner.py`    | 160   | Trap selection from failure_type_library, placeholder resolution |
| `context_builder.py` | 125   | Compact JSONL history rendering for EB and Oracle prompts |
| `env_controller.py`  | 124   | AI2-THOR Controller wrapper, Xvfb management, bounds fix |
| `action_adapter.py`  | 123   | AI2-THOR action parameter adaptation and objectId resolution |
| `episode_manager.py` | 82    | Episode JSON read/write with incremental flush |
| `fork_manager.py`    | 56    | Counterfactual fork reasoning rewriting |
| `step_recorder.py`   | 55    | Step entry builder, frame → PNG saver |

## Scripts (8 scripts in scripts/)

| Script                    | Purpose                                           |
|---------------------------|---------------------------------------------------|
| `run_pipeline.py`         | Main entry point: run full data generation pipeline |
| `e2e_test.py`             | End-to-end test with parallel execution           |
| `e2e_test_output.py`      | E2E test output formatter                         |
| `bench_api.py`            | SiliconFlow API latency benchmarking              |
| `finetune_qwen3vl.py`     | Fine-tuning data generation (LLaMA-Factory format) |
| `generate_ft_data.py`     | Batch reasoning generation from existing episodes |
| `download_alfred.py`      | ALFRED dataset downloader (S3 → local 7z extract) |
| `regenerate_screenshots.py`| Screenshot regeneration from recorded episodes    |
| `replay_demo.py`          | Episode replay demo                               |
| `alfred_replay.py`        | ALFRED trajectory replay                          |
