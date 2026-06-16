# Coding Conventions
**Analysis Date:** 2026-06-16

## Formatter / Linter Configuration

The project does not use formatter or linter config files. There is no `.editorconfig`, `ruff.toml`, `.flake8`, or `[tool.ruff]` section in `pyproject.toml`. The build config in `pyproject.toml` uses Hatchling with no linter/formatter tooling:

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "embodied-failure-dataset"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["ai2thor==5.0.0"]
```

## Naming Conventions

### Classes (PascalCase)

All classes use PascalCase. Agent classes, utility classes, and test classes all follow this convention:

```python
# src/vlm_client.py
class VLMClient:
    ...

# src/eb_agent.py
class EBAgent:
    ...

# src/oracle_agent.py
class OracleAgent:
    ...

# src/executor.py
class ExecutorAgent:
    ...

# src/env_controller.py
class EnvController:
    ...

# src/branch_runner.py
class BranchRunner:
    ...

# src/scheduler.py
class Scheduler:
    ...
```

Test classes also use PascalCase, suffixed with `Test`:

```python
# tests/test_context_builder.py
class ContextBuilderTest(unittest.TestCase):
    ...

# tests/test_error_handling.py
class ErrorHandlingTest(unittest.TestCase):
    ...

# tests/test_vlm_client.py
class VLMClientTest(unittest.TestCase):
    ...
```

### Functions and Methods (snake_case)

All functions and methods use snake_case:

```python
# src/eb_agent.py
def _egocentric_angle(agent_pos, agent_rot_y, obj_pos):
    ...

def _is_in_front(angle: float) -> bool:
    ...

def build_phase1_prompt(
    task_goal: str,
    visible_objects: list[dict],
    ...
) -> str:
    ...

# src/env_controller.py
def get_state_snapshot(self) -> dict:
    ...

def reset_scene(self, scene: str = None):
    ...

# src/action_adapter.py
def adapt(action: str, params: dict) -> tuple[str, dict]:
    ...

def resolve_object_ids(action: str, params: dict, visible_objects: list[dict]):
    ...
```

Private/helper functions are prefixed with a single underscore:

```python
# src/alfred_scene.py
def _apply_object_toggles(controller, object_toggles: list[dict]):
    ...

def _apply_dirty_and_empty(controller):
    ...

def _merge_object_poses_for_current_scene(controller, alfred_object_poses: list[dict]) -> list[dict]:
    ...

def _object_name(obj: dict) -> str:
    ...

# src/context_builder.py
def _render_step(step: dict, include_oracle: bool) -> str:
    ...

def _public_oracle_key(name: str) -> str:
    ...

# src/vlm_client.py
def _summarize_messages(messages: list[dict]) -> str:
    ...
```

### Test Functions (snake_case, test_ prefix)

Test methods always start with `test_`:

```python
# tests/test_context_builder.py
def test_eb_history_uses_all_steps_without_oracle_fields(self):
    ...

# tests/test_error_handling.py
def test_vlm_json_parse_failure_raises_instead_of_pass_fallback(self):
    ...

# tests/test_alfred_scene.py
def test_restore_uses_official_alfred_order(self):
    ...

def test_init_action_runs_after_restore(self):
    ...
```

### Constants (UPPER_SNAKE_CASE)

Module-level constants for system prompts and configuration use UPPER_SNAKE_CASE:

```python
# src/eb_agent.py
PHASE1_SYSTEM = """You are an embodied agent ..."""
PHASE3_SYSTEM = """You are an embodied agent that has just encountered a failure ..."""
PLANNER_SYSTEM = """You are an embodied agent in a 3D household ..."""

# src/executor.py
EXECUTOR_SYSTEM = """You are an embodied agent in a 3D household ..."""

# src/oracle_agent.py
PHASE2_SYSTEM = """You are a supervisor agent ..."""
PHASE2_SYSTEM_FORK = """You are a supervisor agent ... This is a counterfactual branch ..."""
PHASE4_SYSTEM = """You are a supervisor agent evaluating an embodied agent's response ..."""
```

Class-level constants also use UPPER_SNAKE:

```python
# src/egocentric_memory.py
class EgocentricMemory:
    AGING_THRESHOLD: int = 20
    BLOCK_THRESHOLD: int = 2

# src/alfred_scene.py
ALFRED_INIT_SETTINGS = {
    "gridSize": 0.125,
    "cameraY": 0.75,
    ...
}
```

### Variables (snake_case)

All local variables, parameters, and instance attributes use snake_case:

```python
# src/step_recorder.py
step_id: str
branch_id: str
action_params: dict
image_dir: str

# src/branch_runner.py
last_error: Optional[str] = None
cascade_level = 0
fork_tasks: list[dict] = []

# src/context_builder.py
eb_history: list[dict]
shared_ids = set(shared_step_ids or [])
```

### Dataclass Configs (PascalCase)

Small config/result dataclasses use PascalCase as well:

```python
# src/branch_runner.py
@dataclass
class BranchConfig:
    episode_id: str
    branch_id: str
    parent_branch_id: Optional[str]
    ...

@dataclass
class BranchResult:
    branch_id: str
    termination_reason: str
    total_steps: int
    ...

# src/scheduler.py
@dataclass
class SchedulerConfig:
    data_dir: str = "data/json_2.1.0"
    output_dir: str = "output"
    max_episodes: int = 0
    ...
```

## Type Annotations

The project uses extensive type annotations. Three files use `from __future__ import annotations` for PEP 604 union syntax (`str | None`):

```python
# src/context_builder.py, src/executor.py, src/egocentric_memory.py
from __future__ import annotations
```

Annotations are used on function parameters, return types, and class attributes:

```python
# src/task_conditions.py
def check_task_complete(metadata: dict, ep_data: dict, task_state: dict | None = None):
    ...

def check_unrecoverable(metadata: dict, ep_data: dict) -> str | None:
    ...

def detect_dead_loop(action_history: list[dict], window: int = 10, threshold: int = 5) -> bool:
    ...

# src/env_injector.py
def inject(controller: Controller, method: str, pddl_params: dict = None, **params) -> dict:
    ...

def _guard_task_critical(method: str, params: dict, pddl_params: dict) -> str | None:
    ...

# src/alfred_scene.py
def find_closest_object_of_type(object_type: str, ref_object_id: str, metadata: dict) -> dict:
    ...

def _merge_object_poses_for_current_scene(controller, alfred_object_poses: list[dict]) -> list[dict]:
    ...
```

Common patterns for type annotations:
- `str | None` (or `Optional[str]` in files without `from __future__`) for optional values
- `list[dict]` for lists of dictionaries
- `dict` for generic dictionaries
- `set[str]` for string sets
- `Iterable[dict]` from `typing` module
- `np.ndarray` for numpy arrays

## Error Handling Patterns

### ValueError for semantic/validation errors

ValueError is raised when an action or input is semantically invalid:

```python
# src/vlm_client.py
def chat(self, messages, json_mode=False, temperature=0.2, max_tokens=1024):
    if self.backend == "openai":
        return self._chat_openai(...)
    elif self.backend == "ollama":
        return self._chat_ollama(...)
    else:
        raise ValueError(f"Unknown backend: {self.backend}")

# src/alfred_scene.py
def restore_alfred_scene(controller, scene_state: dict):
    object_poses = scene_state.get("object_poses") or []
    if not object_poses:
        raise ValueError("ALFRED scene_state.object_poses is required to restore the expert initial scene")

# src/env_injector.py
def _find_first(objects: list[dict], object_type: str, *, visible: bool | None = None) -> dict:
    if not candidates:
        raise ValueError(f"No{suffix} {object_type} found")
```

### RuntimeError for system/infrastructure errors

RuntimeError is used when something fails at the system level (controller crashes, AI2-THOR failures):

```python
# src/alfred_scene.py
def _step_or_raise(controller, action: str, **params):
    event = controller.step(action=action, **params)
    if not event.metadata.get("lastActionSuccess", False):
        error = event.metadata.get("errorMessage") or "unknown error"
        raise RuntimeError(f"AI2-THOR action {action} failed during ALFRED scene handling: {error}")

# src/branch_runner.py
if nonexecuted_retry_count >= max_nonexecuted_retries:
    raise RuntimeError(error_msg)
```

### Dictionary-based success/error results

For operations that can fail without crashing, a dict with `success` / `error` keys is returned:

```python
# src/env_injector.py
def inject(controller, method, pddl_params=None, **params) -> dict:
    try:
        handler(controller, **params)
        return {"success": True, "error": None, "blocked": None}
    except ValueError as e:
        return {"success": False, "error": str(e), "blocked": None}

# src/env_controller.py
def step(self, action: str, **params) -> dict:
    event = self.controller.step(action=action, **params)
    success = event.metadata["lastActionSuccess"]
    return {
        "success": success,
        "error": event.metadata.get("errorMessage") if not success else None,
        "frame": event.frame,
        "metadata": event.metadata,
        ...
    }
```

### Retry-with-feedback pattern

Several components use retry loops where the model is told what went wrong:

```python
# src/vlm_client.py — chat_text_json retries on parse failure, telling the model
# what was wrong with the previous response

# src/branch_runner.py — nonexecuted_retry_count tracks invalid actions
# and raises RuntimeError after max retries
max_nonexecuted_retries = 3
```

### assert statements

The source code under `src/` does NOT use bare `assert` statements. Error conditions are handled via `raise ValueError(...)` or `raise RuntimeError(...)` instead.

There is no `logging.error()` or `logger.exception()` usage; errors flow through exceptions or the dict-based result pattern above.

## Logging Patterns

### Structured logging (jsonl failure logs)

The primary error tracking mechanism writes structured JSONL failure logs:

```python
# src/branch_runner.py
failure_log_path = os.path.join(
    self.output_dir, ep.episode_id,
    f"failures_{config.branch_id}.jsonl",
)
self._write_failure_log(failure_log_path, {
    "step_index": step_index,
    "branch_id": config.branch_id,
    "failure_type": "model_invalid_action",
    "proposed_action": proposed_action,
    "error_message": error_msg,
    "retry_attempt": nonexecuted_retry_count,
})
```

### Python logging module

Configured centrally in `vlm_client.py`:

```python
# src/vlm_client.py
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("vlm_client")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOG_DIR / "api_calls.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)
```

Other modules reuse this logger:

```python
# src/oracle_agent.py
import logging
logger = logging.getLogger("vlm_client")

# src/executor.py
import logging
logging.warning("Executor output %d actions — rejected as excessive", len(actions))
```

### API call logging

Environment-variable-gated per-instance API call logging:

```python
# src/vlm_client.py
LOG_FULL_API = os.environ.get("EFD_LOG_FULL_API", "1") != "0"
_api_log_dir: Path | None = None

@staticmethod
def set_api_log_dir(log_dir: str | Path | None):
    """Set per-test directory for api_calls.jsonl."""
    global _api_log_dir
    _api_log_dir = Path(log_dir) if log_dir else None
```

### Print statements

`print()` is used sparingly for user-facing status output (e.g., scheduler warmup time, e2e test progress). Not used for internal diagnostics.

## Import Organization

Imports follow a clear ordering: standard library first, then third-party, then project-internal:

```python
# src/vlm_client.py
import json
import os
import base64
import io
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# (No project-internal imports; this is a base module)

# src/executor.py
from __future__ import annotations
import numpy as np
from src.vlm_client import VLMClient
from src.context_builder import build_eb_history_context

# src/scheduler.py
import os
import glob
import threading
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import OracleAgent
from src.executor import ExecutorAgent
from src.branch_runner import (...)
from src.fork_manager import ForkManager
from src.env_controller import EnvController
from src.episode_manager import EpisodeManager
from src.alfred_parser import load_traj, extract_metadata, extract_low_actions
from src.step_recorder import StepRecorder
```

All project-internal imports use absolute imports from `src.`.

## Common Patterns

### Factory Methods (@classmethod)

VLMClient provides classmethod factories for different backends:

```python
# src/vlm_client.py
@classmethod
def ollama(cls, model: str = "qwen2.5-vl:7b", host: str = "http://localhost:11434") -> "VLMClient":
    return cls(backend="ollama", model=model, base_url=host)

@classmethod
def openrouter(cls, model: str = "anthropic/claude-opus-4", api_key: str = None) -> "VLMClient":
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    return cls(backend="openai", model=model, base_url="https://openrouter.ai/api/v1", api_key=key)

@classmethod
def siliconflow(cls, model: str, api_key: str) -> "VLMClient":
    return cls(backend="openai", model=model, base_url="https://api.siliconflow.cn/v1", api_key=api_key)
```

### Static Factory Methods

EpisodeManager uses `__new__` for loading from disk:

```python
# src/episode_manager.py
@staticmethod
def load(file_path: str) -> "EpisodeManager":
    with open(file_path, "r") as f:
        data = json.load(f)
    mgr = EpisodeManager.__new__(EpisodeManager)
    mgr.episode_id = data["episode_id"]
    mgr.file_path = file_path
    mgr.data = data
    return mgr
```

### Dataclasses for Configuration

Configuration objects use `@dataclass` with sensible defaults:

```python
# src/scheduler.py
@dataclass
class SchedulerConfig:
    data_dir: str = "data/json_2.1.0"
    output_dir: str = "output"
    max_episodes: int = 0
    max_parallel: int = 1
    task_filter: str = ""
    splits: str = "train,valid_seen,valid_unseen"
    eb_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    oracle_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    siliconflow_key: str = ""
    enable_fork: bool = True
```

### Dict-based Data Models

The project does not use Pydantic or other schema libraries. All structured data is passed as plain `dict` objects with known keys:

```python
# src/step_recorder.py
return {
    "step_id": step_id,
    "branch_id": branch_id,
    "parent_step_id": parent_step_id,
    "step_index_in_branch": step_index,
    "action": action,
    "action_params": action_params,
    "success": result["success"],
    "error_message": result.get("error"),
    "image_path": image_path,
    "eb_reasoning": None,
    "eb_diagnosis": None,
    ...
}
```

### Agent Wrapper Pattern

Agent classes wrap VLMClient and add domain-specific prompt building:

```python
# src/eb_agent.py
class EBAgent:
    def __init__(self, client: VLMClient):
        self.client = client

    def plan_intent(self, task_goal: str, image: np.ndarray, ...) -> dict:
        prompt = "\n".join(lines)
        return self.client.chat_with_image_json(
            system_prompt=PLANNER_SYSTEM,
            user_text=prompt,
            image=image,
            ...
        )

# src/oracle_agent.py
class OracleAgent:
    def __init__(self, client: VLMClient):
        self.client = client

    def decide_injection(self, ...) -> dict:
        ...

    def evaluate_failure(self, ...) -> dict:
        ...
```

### Function Dispatch via Dicts

Handler lookup tables for both action dispatching and task condition checking:

```python
# src/env_injector.py
_INJECTORS = {
    "set_object_property": _set_object_property,
    "swap_object": _swap_object,
    "occlude_object": _occlude_object,
    "close_container": _close_container,
    "remove_object": _remove_object,
    "hide_object": _hide_object,
}

# src/action_adapter.py
_ADAPTERS = {
    "TeleportFull": _adapt_teleport_full,
    "PutObject": _adapt_put_object,
}
```

### State Flushing

EpisodeManager flushes JSON state to disk on every change:

```python
# src/episode_manager.py
def add_step(self, step: dict):
    self.data["steps"].append(step)
    self._flush()

def _flush(self):
    with open(self.file_path, "w") as f:
        json.dump(self.data, f, indent=2, ensure_ascii=False)
```
