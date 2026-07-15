# Technology Stack

**Analysis Date:** 2026-07-15

## Languages

**Primary:**
- Python 3.10 — All source code in `src/`, scripts in `scripts/`, test files in `tests/`. Pinned in `.python-version`.

**Secondary:**
- None detected — no TypeScript, JS, Rust, C++, or other languages present

## Runtime

**Environment:**
- WSL2 (Ubuntu 24.04) — AI2-THOR requires Linux; WSL2 with WSLg for GPU-accelerated rendering. No native Windows build for AI2-THOR.

**Package Manager:**
- **uv** — Modern Python package manager (`uv run`, `uv sync`). Lockfile `uv.lock` present (183KB).
- `.python-version` set to `3.10`.

**Build System:**
- **hatchling** — Build backend (`pyproject.toml`: `[build-system] requires = ["hatchling"]`).
- Wheel build target: `src/` package directory.

## Frameworks

**Core:**
- None — the project is a research prototype. No web framework (Flask/FastAPI/Django). No ML framework (PyTorch/JAX/TensorFlow). Custom synchronous pipeline.

**Simulation:**
- **AI2-THOR 5.0.0** — 3D household simulation for embodied agent research.
  - Imported as `from ai2thor.controller import Controller` in `src/env_controller.py`.
  - Controller instantiation config: width=300, height=300, makeAgentsVisible=False.
  - Grid size: 0.125m per Move step (`src/alfred_scene.py` `ALFRED_INIT_SETTINGS`).
  - Visibility distance: 100.0 (`src/alfred_scene.py` `ALFRED_INIT_SETTINGS`).
  - `renderObjectImage=True` for per-object instance segmentation support.
  - Bug workaround: `EnvController._fix_visible_bounds()` manually sets `visibleBounds2D` from `visible` flag + `instance_detections2D` (AI2-THOR 5.0.0 processes bounds before detections are populated).

**VLM Client:**
- `VLMClient` class in `src/vlm_client.py` with four factory methods: `siliconflow()`, `openai()`, `openrouter()`, `ollama()`.
- Uses raw `requests` (no OpenAI SDK) for full control over timeout/retry/logging.

**Testing:**
- No test framework configured. A `tests/` directory exists with minimal content.

## Key Dependencies

**Critical:**
- **ai2thor==5.0.0** — The sole explicit dependency in `pyproject.toml`. Entire simulation environment. No fallback.
- **requests** — HTTP client for VLM API calls. Used in `VLMClient._chat_openai()` and `VLMClient._chat_ollama()`.
- **numpy** — Image array handling (`np.ndarray` for frames), geometric math for egocentric direction calculations.
- **Pillow (PIL)** — Image encoding (numpy array → PNG bytes → base64), screenshot saving (`StepRecorder.save_frame()`).

**Infrastructure:**
- **logging** — Standard Python `logging` module. Loggers: `vlm_client` (API calls — file + console handlers), plus general logging.
- **concurrent.futures.ThreadPoolExecutor** — Parallel episode execution in `src/scheduler.py` and `scripts/e2e_test.py`.
- **dataclasses** — Used for `BranchConfig`, `BranchResult`, `SchedulerConfig`, `_ObjectEntry`, `_ObstacleEntry` in `src/branch_runner.py`, `src/scheduler.py`, `src/egocentric_memory.py`.
- **argparse** — CLI argument parsing in `scripts/e2e_test.py`.
- **glob** — ALFRED trajectory discovery in `data/json_2.1.0/{split}/**/traj_data.json`.

**Notable absences:**
- No web framework (Flask/FastAPI)
- No ML framework (PyTorch/JAX/TensorFlow)
- No database client (SQLite/PostgreSQL/Redis)
- No async runtime (asyncio/trio)
- No gRPC/protobuf
- No linter, formatter, or type checker configured
- No CI/CD configuration

## VLM Model Configuration

Single model accessed via OpenAI-compatible API (www.9527code.com/v1):

| Role | Agent Class | Model | Reasoning Effort | Purpose |
|------|-------------|-------|-------------------|---------|
| Planner (EB) | `EBAgent` in `src/eb_agent.py` | `gpt-5.5` | medium | Phase 1 intent, review, Phase 3 diagnosis, scan room analysis |
| Executor | `ExecutorAgent` in `src/executor.py` | `gpt-5.5` | medium | Intent-to-actions decomposition |
| Oracle | `OracleAgent` in `src/oracle_agent.py` | `gpt-5.5` | medium | Phase 2 injection decisions, Phase 4 evaluation |

- All three roles use the same model with different system prompts.
- Model default: `--planner-model gpt-5.5` / `--executor-model gpt-5.5` / `--oracle-model gpt-5.5`.
- Reasoning effort default: `medium` (xhigh causes content=None on multi-image gpt-5.5).

## VLM Client Features

**Response validation** (`src/vlm_client.py` `_chat_openai()`):
- HTTP-200 5-stage validation: JSON parse → non-empty choices → dict message → non-empty string content
- reasoning_effort auto-strip fallback: when reasoning_content present but content=None, retry without reasoning_effort
- Malformed 200 retry: 5 attempts with exponential backoff (1s→2s→4s→8s), max_tokens doubling at attempt 3+ (→4096→8192)
- Exhaustion falls through to transport retries (3×5=15 total)
- 400 transport retry: 3 attempts with backoff (proxy glitch workaround)

**Transport retry:**
- Connection errors: 3 attempts, exponential backoff (1s, 2s, 4s)
- Timeout escalation: 300s → 400s → 500s (3 attempts)
- Rate limiting (HTTP 429): waits 5s, 10s, 15s
- Server errors (5xx): 3 attempts with backoff

**Image handling:**
- Pipeline: numpy array (HWC uint8) → PIL → PNG bytes → base64 → inline `data:image/png;base64,...` URI.
- Multi-image support: `chat_with_images()` sends direction labels ("VIEW 1: ahead") as text blocks before each image.

## Logging System

| Path | Format | Content |
|------|--------|---------|
| `logs/api_calls.jsonl` | JSONL | Full API request/response (images sanitized to `<image_base64 len=N>` stubs). Gated by env var `EFD_LOG_FULL_API`. |
| `logs/api_calls.log` | Text | Timestamped DEBUG-level summary of each API call |
| `logs/failure_<ts>_<model>.json` | JSON | Full request dump on API failure (timeout, connection error, non-200 status). Written by `_dump_failure()`. |
| `output/<ep_id>/failures_<branch>.jsonl` | JSONL | Structured failure events per branch (invalid action, done rejected, injection failed, etc.) |

## Configuration

**Python build:**
- `pyproject.toml` — project metadata, hatchling build config
- `.python-version` — Python version pin

**Environment variables:**
- `DISPLAY` — X display target. Set automatically by `EnvController._start_xvfb()`: tries WSLg `:0` first, falls back to Xvfb `:99`.
- `EFD_LOG_FULL_API` — Controls full API call JSONL logging. Default: `"1"` (enabled). Set to `"0"` to disable.
- `OPENAI_API_KEY` / `OPENROUTER_API_KEY` — Fallbacks for alternate factory methods (not actively used).

**No other config files** — no Dockerfile, no CI config, no linter config.

## Platform Requirements

**Development:**
- WSL2 with WSLg (for GPU-accelerated X display).
- `uv` installed in PATH (`$HOME/.local/bin/uv`).
- Python 3.10.
- SiliconFlow API account with API key.

**Production:**
- Not applicable — this is a research dataset generation tool. Runs on local machine or cloud VM with X server.

---

*Stack analysis: 2026-06-12*
