# External Integrations

**Analysis Date:** 2026-06-12

## APIs & External Services

### VLM Inference (SiliconFlow)
- **Service:** SiliconFlow API (`api.siliconflow.cn/v1`) — OpenAI-compatible `/chat/completions` endpoint.
- **SDK/Client:** Custom lightweight client in `src/vlm_client.py` using `requests` directly (no OpenAI SDK).
- **Auth:** API key passed as `--api-key` CLI argument. Stored in `VLMClient.api_key`, sent as `Authorization: Bearer <key>` header.
- **Models:**
  - Planner: `Qwen/Qwen3-VL-32B-Instruct` (intent proposal, review, diagnosis, scan analysis)
  - Executor: `Qwen/Qwen3-VL-8B-Instruct` (concrete action generation)
  - Oracle: `Qwen/Qwen3-VL-32B-Instruct` (injection decisions, failure evaluation, counterfactual grading)
- **Endpoint:** `POST {base_url}/chat/completions`
- **Retry logic (`_chat_openai()`):**
  1. Normal attempt with 300s timeout.
  2. Timeout: escalate by 100s (400s, then 500s), up to 3 tries.
  3. Connection error: exponential backoff (1s, 2s, 4s), up to 3 tries.
  4. HTTP 429 rate limit: wait 5s x (attempt+1), up to 3 tries.
  5. JSON parse failure: up to 2 retries with error feedback in prompt.
- **Image encoding:** numpy array → PIL → PNG → base64 → `data:image/png;base64,...` inline URI.
- **Multi-image support:** `chat_with_images()` sends direction labels ("VIEW 1: ahead", "VIEW 2: left", etc.) as text blocks before each image.
- **Thinking model CoT compatibility:** `parse_json_response()` extracts JSON `{...}` from surrounding text (`src/vlm_client.py` lines 467-477).
- **Logging:**
  - Full API exchanges → `logs/api_calls.jsonl` (sanitized: images replaced with `<image_base64 len=N>` stubs).
  - Failed calls → `logs/failure_<ts>_<model>.json` with full request body.
  - Controlled by env var `EFD_LOG_FULL_API` (default enabled).

### Alternative Backends (factory methods in `src/vlm_client.py`)
| Factory | Default Model | Base URL | Status |
|---------|--------------|----------|--------|
| `siliconflow()` | configurable | `https://api.siliconflow.cn/v1` | **Active** (primary) |
| `openai()` | `gpt-5` | `https://api.openai.com/v1` | Available, not actively used |
| `openrouter()` | `anthropic/claude-opus-4` | `https://openrouter.ai/api/v1` | Available, not actively used |
| `ollama()` | `qwen2.5-vl:7b` | `http://localhost:11434` | Available, not actively used (tested historically) |

## Data Storage

**Databases:**
- None — no SQL/NoSQL database used. All state is in-memory during execution.

**File Storage:**
- **Local filesystem only** — output directories named `output_e2e_YYYYMMDD_HHMMSS/`. Contents:
  - Per-episode JSON files (`EpisodeManager` writes incremental episode data as JSON, flushed each step).
  - Step screenshots (PNG, named `s<step_index>.png`).
  - LookAround view images (PNG, named `s<step_index>_look_<dir>.png` for ahead/left/behind/right).
  - Failure logs (`failures_<branch_id>.jsonl` — JSON Lines format per branch).
- **No cloud storage integration** (no S3, GCS, Azure Blob).

**Caching:**
- None — no Redis, memcached, or on-disk cache. Every VLM call is fresh.

## Data Sources

### ALFRED Dataset
- **Source:** Downloaded from ALFRED repository to `data/json_2.1.0/`.
- **Format:** `json_2.1.0` — each episode is a subdirectory containing `traj_data.json`.
- **Structure:**
  ```
  data/json_2.1.0/
    train/
    valid_seen/
    valid_unseen/
      <episode_id>/
        traj_data.json
  ```
- **Parser:** `src/alfred_parser.py` — `load_traj()`, `extract_metadata()`, `extract_scene_state()`, `extract_low_actions()`.
- **7 task types:**
  | ALFRED Type | Mapped Type | Checker in `task_conditions.py` |
  |-------------|-------------|--------------------------------|
  | `pick_and_place_simple` | `pick_and_place` | `_pick_and_place_simple` |
  | `pick_and_place_with_movable_recep` | `pick_and_place` | `_pick_and_place_with_movable_recep` |
  | `pick_clean_then_place_in_recep` | `clean` | `_pick_clean_then_place` |
  | `pick_heat_then_place_in_recep` | `heat` | `_pick_heat_then_place` |
  | `pick_cool_then_place_in_recep` | `cool` | `_pick_cool_then_place` |
  | `look_at_obj_in_light` | `examine` | `_look_at_obj_in_light` |
  | `pick_two_obj_and_place` | `pick_two` | `_pick_two` |

### Failure Type Library
- **File:** `data/failure_type_library.json` — 8 hand-written failure types.
- **Loader:** `TrapPlanner` in `src/trap_planner.py`.
- **Format:** Each entry has `failure_type`, `description`, `applicable_tasks`, `injection` (method + params with `<target>`, `<appliance>`, `<container>` type placeholders that are resolved against scene objects at runtime).
- **Constraints:** Supports `exclude_types` to protect task-critical objects; has `Blinds` blocklist for close operations (AI2-THOR timeout guard).

## Input / Output

### Entry Points

| Script | Purpose | Command |
|--------|---------|---------|
| `scripts/e2e_test.py` | Single/multi-episode end-to-end test | `uv run python scripts/e2e_test.py --api-key sk-xxx --task pick_and_place_simple` |
| `scripts/run_pipeline.py` | Batch data generation pipeline | `uv run python scripts/run_pipeline.py --max 10 --api-key sk-xxx --parallel 3` |

### e2e_test.py CLI Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` | (required) | SiliconFlow API key |
| `--task` | `""` | Single task type |
| `--all` | `False` | Run all 7 task types in parallel |
| `--random` | `False` | Random trajectory selection |
| `--no-traps` | `False` | Disable trap injection |
| `--parallel` | `3` | Max ThreadPoolExecutor workers |
| `--output` | timestamp | Output directory override |
| `--data-dir` | `data/json_2.1.0` | ALFRED data path |
| `--eb-model` | `Qwen/Qwen3-VL-32B-Instruct` | Planner model override |
| `--oracle-model` | `Qwen/Qwen3-VL-32B-Instruct` | Oracle model override |
| `--executor-model` | `Qwen/Qwen3-VL-8B-Instruct` | Executor model override |

### Output Layout
| Path | Format | Writer | Purpose |
|------|--------|--------|---------|
| `output_e2e_<ts>/<ep_id>.json` | JSON (incremental) | `EpisodeManager._flush()` | Complete episode data with all steps, metadata, traps, outcome |
| `output_e2e_<ts>/<ep_id>/s<N>.png` | PNG | `StepRecorder.save_frame()` | Per-step first-person screenshots |
| `output_e2e_<ts>/<ep_id>/s<N>_look_<dir>.png` | PNG | `BranchRunner.run()` | LookAround multi-directional views |
| `output_e2e_<ts>/<ep_id>/failures_<branch>.jsonl` | JSONL | `BranchRunner._write_failure_log()` | Structured failure events |

### Resume Capability
- `BranchRunner.resume()` (`src/branch_runner.py` line 940) loads existing episode JSON, replays successful steps, continues from last step.
- Requires `alfred_scene` state in episode JSON (for `reset_to_alfred_scene`).

## Authentication & Identity

**Auth Provider:**
- **Custom** — API key authentication for SiliconFlow.
- Implementation: `Authorization: Bearer <key>` header in `VLMClient._chat_openai()` (`src/vlm_client.py` line 289).
- The API key value is documented (with value) in `CLAUDE.md`. This is a known security concern.

## Monitoring & Observability

**Error Tracking:**
- None — no Sentry, Datadog, or similar service.

**Logs:**
- Python `logging` module with two handlers per API logger:
  - Console handler (INFO level): API call summaries.
  - File handler (DEBUG level): `logs/api_calls.log`.
- Full API journal: `logs/api_calls.jsonl` (disablable via `EFD_LOG_FULL_API=false`).
- Failure dump: `logs/failure_<ts>_<model>.json` with full request body on catastrophic failures.

## CI/CD & Deployment

**Hosting:**
- Not deployed — runs locally on developer machine (WSL2 Ubuntu 24.04).

**CI Pipeline:**
- None — no GitHub Actions, Jenkins, or other CI configuration files found.

## Environment Configuration

| Variable | Source File | Purpose |
|----------|-------------|---------|
| `DISPLAY` | `src/env_controller.py` | X display for AI2-THOR. Auto-detected: WSLg `:0` → fallback Xvfb `:99`. |
| `EFD_LOG_FULL_API` | `src/vlm_client.py` | Set to `"0"` to disable full API exchange logging. |
| `OPENAI_API_KEY` | `src/vlm_client.py` | Fallback for `VLMClient.openai()` factory (not actively used). |
| `OPENROUTER_API_KEY` | `src/vlm_client.py` | Fallback for `VLMClient.openrouter()` factory (not actively used). |

**Secrets:**
- SiliconFlow API key passed via `--api-key` CLI argument. Not read from environment variables or `.env` files.
- No `.env` file used.

## Webhooks & Callbacks

**Incoming:**
- None — no HTTP server running.

**Outgoing:**
- None — no webhook/callback integration.

## No Other Integrations

- No database (all persistence is JSON files on local filesystem).
- No message queue.
- No cloud storage.
- No authentication service beyond the API key header.
- No web server or API endpoints.
- No gRPC or protobuf.

---

*Integration audit: 2026-06-12*
