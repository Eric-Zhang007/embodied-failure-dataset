# External Integrations
**Analysis Date:** 2026-06-16

## Overview

embodied-failure-dataset integrates with multiple external services and data sources to run embodied agent experiments. The VLM client (`src/vlm_client.py`) supports 4 backends, the environment controller wraps AI2-THOR, and the data pipeline consumes the ALFRED dataset.

---

## 1. VLM/LLM API Providers

### 1.1 SiliconFlow (Primary)
- **URL:** `https://api.siliconflow.cn/v1`
- **Protocol:** OpenAI-compatible Chat Completions API
- **Auth:** API key via `--api-key` CLI argument → `SILICONFLOW_API_KEY` env var
- **Models used:**
  - `Qwen/Qwen3-VL-32B-Instruct` — EB Agent (Planner + Phase 1 action proposal + Phase 3 diagnosis)
  - `Qwen/Qwen3-VL-32B-Instruct` — Oracle Agent (Phase 2 injection decisions + Phase 4 evaluation)
  - `Qwen/Qwen3-VL-8B-Instruct` — Executor Agent (low-level action chunk generation)
- **Endpoint:** `POST {base_url}/chat/completions`
- **Features used:**
  - Multi-modal (text + base64-encoded PNG images via `data:image/png;base64,...`)
  - JSON mode (`response_format: {"type": "json_object"}`)
  - Configurable temperature (0.3 default) and max_tokens (up to 16384 for Executor)
- **Retry logic:** 3 attempts with exponential backoff for 429 rate limits
- **Timeout:** 300s initially, increases by 100s per retry
- **Import:** `import requests` (lazy, inside `_chat_openai`)

Relevant code from `src/vlm_client.py`:
```python
@classmethod
def siliconflow(cls, model: str, api_key: str) -> "VLMClient":
    return cls(backend="openai", model=model,
               base_url="https://api.siliconflow.cn/v1", api_key=api_key)
```

Usage in `src/scheduler.py`:
```python
eb_client = VLMClient.siliconflow(config.eb_model, config.siliconflow_key)
oracle_client = VLMClient.siliconflow(config.oracle_model, config.siliconflow_key)
executor_client = VLMClient.siliconflow("Qwen/Qwen3-VL-8B-Instruct", config.siliconflow_key)
```

### 1.2 OpenAI (Alternative)
- **URL:** `https://api.openai.com/v1`
- **Protocol:** OpenAI Chat Completions API
- **Auth:** `OPENAI_API_KEY` environment variable or explicit `api_key` parameter
- **Model:** `gpt-5` (default)
- **Import:** `import requests` (lazy)

```python
@classmethod
def openai(cls, model: str = "gpt-5", api_key: str = None, base_url: str = None) -> "VLMClient":
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    url = base_url or "https://api.openai.com/v1"
    return cls(backend="openai", model=model, base_url=url, api_key=key)
```

### 1.3 OpenRouter (Alternative)
- **URL:** `https://openrouter.ai/api/v1`
- **Protocol:** OpenAI-compatible
- **Auth:** `OPENROUTER_API_KEY` environment variable
- **Model:** `anthropic/claude-opus-4` (default)
- **Import:** `import requests` (lazy)

```python
@classmethod
def openrouter(cls, model: str = "anthropic/claude-opus-4", api_key: str = None) -> "VLMClient":
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    return cls(backend="openai", model=model,
               base_url="https://openrouter.ai/api/v1", api_key=key)
```

### 1.4 Ollama (Local)
- **URL:** `http://localhost:11434` (configurable)
- **Protocol:** Ollama native Chat API
- **Auth:** None (local)
- **Model:** `qwen2.5-vl:7b` (default)
- **Endpoint:** `POST {base_url}/api/chat`
- **Image format:** Ollama `images` array (base64 strings, not inline data URIs)
- **Timeout:** 40s
- **Import:** `import requests` (lazy)

```python
@classmethod
def ollama(cls, model: str = "qwen2.5-vl:7b", host: str = "http://localhost:11434") -> "VLMClient":
    return cls(backend="ollama", model=model, base_url=host)
```

### API Call Flow Summary

```
VLMClient.chat()
├── backend == "openai" → _chat_openai()
│   POST {base_url}/chat/completions
│   Headers: Authorization: Bearer {api_key}, Content-Type: application/json
│   Body: {model, messages, temperature, max_tokens, [response_format]}
│   Retry: 3 attempts, 429 backoff, ConnectionError backoff, Timeout escalation
│   Logging: api_calls.jsonl + failure dumps
│
└── backend == "ollama" → _chat_ollama()
    POST {base_url}/api/chat
    Body: {model, messages, stream: false, options: {temperature, num_predict}}
    Image conversion: data:image/png;base64 → Ollama images[] array
    Retry: 3 attempts
    Timeout: 40s
```

---

## 2. AI2-THOR Simulation Environment

### Connection
- **Library:** `ai2thor==5.0.0` (pinned)
- **Architecture:** Python client ↔ Unity game engine (bundled with package)
- **Import:** `from ai2thor.controller import Controller`

### Controller Setup (`src/env_controller.py`)
```python
from ai2thor.controller import Controller

self.controller = Controller(
    scene=scene,            # FloorPlan1..FloorPlan10
    width=300, height=300,  # Render resolution
    makeAgentsVisible=False,
)
```

### Grid System
- Grid size: 0.125m per step
- Camera height: 0.75m (ALFRED default)
- Navigation: 4-directional movement + 90° rotation
- Interaction range: 0.5m

### AI2-THOR Actions Used

**Navigation:**
- `MoveAhead`, `MoveBack`, `MoveLeft`, `MoveRight` (0.125m each)
- `RotateLeft`, `RotateRight` (90°)
- `LookUp`, `LookDown` (camera tilt, +30°/-60° limits)
- `TeleportFull` (initial scene setup, ALFRED init_action)

**Object Interaction:**
- `PickupObject`, `PutObject`, `DropHandObject`
- `OpenObject`, `CloseObject`
- `ToggleObjectOn`, `ToggleObjectOff`
- `SliceObject`, `BreakObject`
- `FillObjectWithLiquid`, `EmptyLiquidFromObject`

**Environment Mutation (Oracle injections):**
- `SetObjectStatic`, `DisableObject`, `PlaceObjectAtPoint`
- `DirtyObject`, `CookObject`, `UseUpObject`
- `SetObjectPoses` (ALFRED scene restoration)

**Metadata:**
- `Pass` (no-op, get current state)
- `Initialize` (scene setup with ALFRED_INIT_SETTINGS)

### Known Bug Workaround
AI2-THOR 5.0.0 has a bug where `process_visible_bounds2D` runs before `instance_detections2D` is populated. Fixed manually in `EnvController._fix_visible_bounds()`:
```python
for obj in event.metadata["objects"]:
    obj["visibleBounds2D"] = (
        obj.get("visible", False)
        and obj["objectId"] in det
    )
```

---

## 3. ALFRED Dataset

### Data Source
- **URL:** `https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z`
- **Format:** 7z archive containing JSON trajectory data (no images/features)
- **Size:** ~200MB compressed
- **Download script:** `scripts/download_alfred.py` (uses `wget` + `7z`)
- **Local path:** `data/json_2.1.0/`

### Directory Structure
```
data/json_2.1.0/
├── train/
│   └── <episode_id>/
│       └── traj_data.json
├── valid_seen/
│   └── <episode_id>/
│       └── traj_data.json
└── valid_unseen/
    └── <episode_id>/
        └── traj_data.json
```

### traj_data.json Schema (fields used)

| Field | Type | Usage |
|-------|------|-------|
| `task_type` | string | ALFRED task type (pick_and_place_simple, look_at_obj_in_light, etc.) |
| `task_id` | string | ALFRED task identifier |
| `scene.floor_plan` | string | AI2-THOR floor plan name (FloorPlan1..10) |
| `scene.scene_num` | int | Scene number |
| `scene.random_seed` | int | Random seed for reproducibility |
| `scene.object_poses` | list | Initial object positions/rotations |
| `scene.object_toggles` | list | Initial toggle states |
| `scene.dirty_and_empty` | bool | Clean scene flag |
| `scene.init_action` | dict/string | Initial agent teleport |
| `pddl_params` | dict | PDDL task parameters (object_target, parent_target, etc.) |
| `plan.low_actions` | list | Expert trajectory actions with high_idx |
| `turk_annotations` | list | Human task descriptions (not used; PDDL-based goals instead) |

### PDDL Parameters Used
- `object_target` — main task object
- `parent_target` — target receptacle
- `mrecep_target` — movable receptacle (for pick_two tasks)
- `toggle_target` — appliance to toggle (for examine tasks)
- `object_sliced` — whether object needs slicing

### Task Type Mapping
```python
TASK_TYPE_MAP = {
    "pick_and_place_simple": "pick_and_place",
    "pick_and_place_with_movable_recep": "pick_and_place",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "examine",
    "pick_two_obj_and_place": "pick_two",
}
```

### ARNOLD S3 Bucket (ALFRED source)
- **Bucket:** `ai2-vision-alfred`
- **Region:** `us-west-2`
- **Accessed via:** `botocore` + `aws-requests-auth` (transitive deps of ai2thor)

---

## 4. Data Formats & File Storage

### 4.1 Episode Output Format

**Path:** `output/<episode_id>.json`

```json
{
  "episode_id": "...",
  "task_goal": "Pick up the Apple and put it on the Counter.",
  "scene": "FloorPlan1",
  "task_type": "pick_and_place",
  "alfred_task_type": "pick_and_place_simple",
  "alfred_task_id": "...",
  "pddl_params": {"object_target": "Apple", "parent_target": "Counter"},
  "alfred_scene": { "floor_plan": "...", "object_poses": [...], ... },
  "initial_traps": [{ "trap_id": "trap_0", "failure_type": "...", "injection": {...} }],
  "runtime_traps": [{ "trap_id": "trap_main_5", "injection": {...}, "status": "active" }],
  "steps": [
    {
      "step_id": "s0",
      "branch_id": "main",
      "parent_step_id": null,
      "step_index_in_branch": 0,
      "action": "LookAround",
      "action_params": {},
      "success": true,
      "image_path": "output/<ep_id>/s0.png",
      "lookaround_views": [
        {"label": "ahead", "image_path": "output/<ep_id>/s0_look_ahead.png"},
        ...
      ],
      "eb_reasoning": "...",
      "eb_diagnosis": null,
      "oracle_injection_decision": null,
      "fork_metadata": null
    }
  ],
  "final_outcome": { "result": "task_complete", "branches": [...] }
}
```

### 4.2 Step Screenshots
- **Format:** PNG (RGB, uint8)
- **Saving:** `PIL.Image.fromarray(frame).save(path)`
- **Naming:** `output/<ep_id>/s{step_index}.png`
- **LookAround views:** `output/<ep_id>/s{step_index}_look_{direction}.png`

### 4.3 API Call Logs
- **File:** `output/<ep_id>/api_calls.jsonl` or `logs/api_calls.jsonl`
- **Format:** JSONL, one line per API exchange
- **Sanitized:** Image data replaced with size indicator in logs

### 4.4 Failure Logs
- **File:** `output/<ep_id>/failures_{branch_id}.jsonl`
- **Contents:** Failed step details, error messages, injection decisions

### 4.5 Failure Type Library
- **File:** `data/failure_type_library.json`
- **8 failure types:** immovable_object, broken_appliance, locked_container, occluded_target, missing_object_light, wrong_receptacle_closed, appliance_off, removed_tool
- **Categories:** physical, perceptual, semantic
- **Injection methods:** set_object_property, close_container, occlude_object, remove_object, hide_object, swap_object

---

## 5. Environment Injection Methods

Environment mutations are applied via `src/env_injector.py` using AI2-THOR low-level actions:

| Method               | AI2-THOR Actions Used                    | Purpose                          |
|----------------------|------------------------------------------|----------------------------------|
| `set_object_property`| SetObjectStatic, BreakObject, OpenObject/CloseObject, ToggleObjectOn/Off, DirtyObject, CookObject, SliceObject, UseUpObject, FillObjectWithLiquid | Modify object state |
| `close_container`    | CloseObject (forceAction)                | Close a cabinet/fridge/drawer   |
| `occlude_object`     | PlaceObjectAtPoint                       | Block object from view          |
| `remove_object`      | DisableObject                            | Remove object from scene        |
| `hide_object`        | PlaceObjectAtPoint                       | Move object to non-target receptacle |
| `swap_object`        | PlaceObjectAtPoint (×2)                  | Replace object with another     |

### Safety Guards (`_guard_task_critical`)
- Never apply irreversible properties (broken, static, sliced, cooked, used_up) to task-critical objects
- Never remove/hide/occlude/swap task-critical objects (object_target, parent_target, mrecep_target, toggle_target)

---

## 6. Fine-Tuning Pipeline Integration

### LLaMA-Factory
- **Tool:** LLaMA-Factory CLI (`llamafactory-cli`)
- **Format:** ShareGPT JSON format
- **Data generation:** `scripts/finetune_qwen3vl.py` converts episode JSONs → LLaMA-Factory training data
- **Training:** QLoRA fine-tuning of Qwen3-VL-8B-Instruct
- **Config path:** `ft_data/training/train_config.yaml`
- **Model output:** `ft_model/` (adapter weights), `ft_model/merged/` (merged model)
- **Resume:** `--resume_from_checkpoint ft_model/checkpoint-N`

### Model Architecture
```
EB Agent (Qwen3-VL-32B, via API)
Oracle Agent (Qwen3-VL-32B, via API)
Executor Agent (Qwen3-VL-8B, via API; optionally fine-tuned locally)
```

---

## 7. External Tools (System Dependencies)

| Tool | Purpose | Used In |
|------|---------|---------|
| `Xvfb` | Virtual framebuffer for headless AI2-THOR | `src/env_controller.py` |
| `xdpyinfo` | Detect existing X display (WSLg) | `src/env_controller.py` |
| `wget` | Download ALFRED dataset | `scripts/download_alfred.py` |
| `7z` | Extract ALFRED 7z archive | `scripts/download_alfred.py` |
| `llamafactory-cli` | Fine-tuning Qwen3-VL | `scripts/finetune_qwen3vl.py` |

---

## 8. Configuration & Environment Variables

| Variable | Purpose | Default | Used In |
|----------|---------|---------|---------|
| `EFD_LOG_FULL_API` | Enable full API call logging (set to "0" to disable) | "1" | `vlm_client.py` |
| `OPENAI_API_KEY` | OpenAI API key | "" | `vlm_client.py` |
| `OPENROUTER_API_KEY` | OpenRouter API key | "" | `vlm_client.py` |
| `DISPLAY` | X11 display for AI2-THOR | auto-detected | `env_controller.py` |

**No `.env` file found** in the repository — configuration is passed via CLI arguments at runtime.

CLI parameters for `run_pipeline.py`:
```
--data-dir     data/json_2.1.0   (ALFRED data directory)
--output       auto-generated    (output directory with timestamp)
--max          0 (all)           (maximum episodes)
--task         ""                (filter by task type)
--splits       train,valid_seen,valid_unseen
--eb-model     Qwen/Qwen3-VL-32B-Instruct
--oracle-model Qwen/Qwen3-VL-32B-Instruct
--api-key      ""                (SiliconFlow API key)
--no-fork      false             (disable counterfactual forks)
--parallel     1                 (parallel workers)
```

---

## 9. Network Endpoints Summary

| Endpoint | Protocol | Auth | Purpose |
|----------|----------|------|---------|
| `https://api.siliconflow.cn/v1/chat/completions` | HTTPS | Bearer token | Primary VLM inference (Qwen3-VL models) |
| `https://api.openai.com/v1/chat/completions` | HTTPS | Bearer token | Alternative VLM (GPT-5) |
| `https://openrouter.ai/api/v1/chat/completions` | HTTPS | Bearer token | Alternative VLM (Claude Opus 4) |
| `http://localhost:11434/api/chat` | HTTP | None | Local Ollama VLM (qwen2.5-vl:7b) |
| `https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/` | HTTPS (S3) | None | ALFRED dataset download |
| `https://pypi.org/simple` | HTTPS | None | Package registry (uv/pip) |
| `https://files.pythonhosted.org/` | HTTPS | None | Package wheels (uv/pip) |
