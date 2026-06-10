# STACK.md — Technology Stack

## Runtime

| Component | Detail |
|-----------|--------|
| Language | Python 3.10+ |
| Package Manager | uv |
| OS | Ubuntu 24.04 (WSL2 on Windows 11) |
| Build System | hatchling |

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ai2thor | 5.0.0 | 3D household simulator (Unity-based) |
| numpy | (ai2thor dep) | Image array handling, math |
| Pillow | (ai2thor dep) | Image encoding for VLM API |
| requests | (ai2thor dep) | HTTP client for VLM API calls |

The project has minimal dependencies — only `ai2thor==5.0.0` is declared directly. All other packages (numpy, Pillow, requests) come transitively through AI2-THOR.

## External Services

| Service | Endpoint | Model | Purpose |
|---------|----------|-------|---------|
| SiliconFlow API | `https://api.siliconflow.cn/v1` | Qwen/Qwen3-VL-32B-Instruct | Both EB Agent and Oracle Agent VLM calls |

- Protocol: OpenAI-compatible `/chat/completions` endpoint
- Auth: Bearer token (`sk-umtq...`)
- The same 32B model is used for both EB (embodied) and Oracle roles
- No local model hosting — all inference goes through SiliconFlow cloud

## Infrastructure

| Component | Detail |
|-----------|--------|
| Simulator | AI2-THOR 5.0.0 (Unity backend) |
| Headless Rendering | Xvfb on display `:99`, auto-started by `EnvController` |
| Resolution | 300×300 (configurable in `EnvController.__init__`) |
| Data Source | ALFRED dataset (json_2.1.0 format) |

## Dev Tools

- **uv**: Python package management (`uv run python ...`)
- **hatchling**: Build backend (PEP 621)
- No linter, formatter, or type checker configured
- No CI/CD configuration present
- No Docker configuration

## Logging

- Python `logging` module (DEBUG to file, INFO to console)
- Full API request/response logged to `logs/api_calls.jsonl`
- Summary API log to `logs/api_calls.log`
- Failed API calls dumped to `logs/failure_<timestamp>_<model>.json`
- Branch failure events logged to per-episode `failures_<branch>.jsonl`

## API Client Features

- Connection retry (3 attempts, exponential backoff)
- Rate-limit handling (429 → wait 5/10/15s)
- Timeout escalation (120s → 150s → 180s)
- JSON parse retry (2 attempts with error feedback)
- Base64 image encoding (PNG format, inline data URIs)
- Multi-image support with view labels (VIEW 1: ahead, etc.)
