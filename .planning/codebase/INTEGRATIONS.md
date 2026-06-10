# INTEGRATIONS.md — External Integrations

## 1. AI2-THOR Simulator

**File**: `src/env_controller.py`

- Wraps `ai2thor.controller.Controller` in `EnvController` class
- Each episode creates a fresh controller with a specific scene
- `step(action, **params)` returns `{success, error, frame, metadata, task_state}`
- `get_state_snapshot()` captures current frame + metadata via `Pass` action
- `reset_to_alfred_scene()` restores scene from ALFRED scene state JSON
- Xvfb auto-start on class first use (display `:99`, 1024×768×24)
- Scene restoration uses `src/alfred_scene.py` utilities (object placement, init action)
- `alfred_task_state` tracks cleaned/heated/cooled sets across steps
- `clean_sink_contents_after_faucet()` handles an AI2-THOR edge case

## 2. VLM API (SiliconFlow)

**File**: `src/vlm_client.py`

- `VLMClient` class with factory methods: `ollama()`, `openai()`, `openrouter()`, `siliconflow()`
- Primary use: `VLMClient.siliconflow(model, api_key)`
- OpenAI-compatible protocol via `requests.post(f"{base_url}/chat/completions", ...)`
- Image handling: numpy array → PIL → PNG bytes → base64 → `data:image/png;base64,...` inline URI
- Key methods:
  - `chat_with_image()` — single image + text → string
  - `chat_with_image_json()` — single image → parsed dict with JSON retry
  - `chat_with_images()` — multi-image with VIEW labels
  - `chat_with_images_json()` — multi-image → parsed dict
  - `chat_text()` / `chat_text_json()` — text-only variants
- JSON retry: up to 2 retries with explicit error feedback in prompt
- Network retry: 3 attempts for connection errors and timeouts
- Rate limit: detects 429, waits 5s×(attempt+1)
- All exchanges logged to `logs/api_calls.jsonl` (base64 images replaced with length stubs)
- Failed exchanges dumped to `logs/failure_*.json` for debugging

## 3. ALFRED Dataset

**Files**: `src/alfred_parser.py`, `scripts/download_alfred.py`

- Data stored in `data/json_2.1.0/{split}/{task_type}-{object}-{receptacle}-{idx}/trial_*/traj_data.json`
- `load_traj(path)` — loads a single traj_data.json
- `extract_metadata(traj)` — extracts task_goal, scene, task_type, pddl_params, alfred_scene, alfred_task_id
- `extract_low_actions(traj)` — extracts expert low-level actions for step count baseline
- Supports both old (string) and new (dict) `api_action` formats
- Scene state (`alfred_scene`) contains object_poses and init_action for environment restoration
- `data/failure_type_library.json` — 8 hand-crafted failure types for trap injection (separate from ALFRED)

## 4. File System

| Path | Format | Writer | Purpose |
|------|--------|--------|---------|
| `output/{episode_id}.json` | JSON (pretty-printed) | `EpisodeManager._flush()` | Episode state, flushed after every step |
| `output/{episode_id}/*.png` | PNG | `StepRecorder.save_frame()` | Per-step and LookAround screenshots |
| `output/{episode_id}/failures_{branch}.jsonl` | JSONL | `BranchRunner._write_failure_log()` | Structured failure events |
| `logs/api_calls.jsonl` | JSONL | `VLMClient._write_api_exchange()` | Full API request/response pairs |
| `logs/api_calls.log` | Text | Python logging | Human-readable API call summary |
| `logs/failure_*.json` | JSON | `VLMClient._dump_failure()` | Failed API call dump |

## 5. Environment Variable Dependencies

| Variable | Used In | Purpose |
|----------|---------|---------|
| `EFD_LOG_FULL_API` | `vlm_client.py` | Set to "0" to disable full API JSONL logging |
| `DISPLAY` | `env_controller.py` | X display for AI2-THOR; auto-set to `:99` if unset |
| `OPENAI_API_KEY` | `vlm_client.py` | Fallback for openai factory method |
| `OPENROUTER_API_KEY` | `vlm_client.py` | Fallback for openrouter factory method |

## 6. No Other Integrations

- No database
- No message queue
- No cloud storage
- No monitoring/observability (beyond local log files)
- No authentication service
- No web server or API
