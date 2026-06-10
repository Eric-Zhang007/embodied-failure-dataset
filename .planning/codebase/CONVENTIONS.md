# CONVENTIONS.md — Code Conventions & Patterns

## Naming

- **Files**: `snake_case.py` throughout
- **Classes**: `PascalCase` — `BranchRunner`, `VLMClient`, `EpisodeManager`
- **Functions/methods**: `snake_case` — `run_single_branch()`, `build_phase1_prompt()`
- **Constants**: `UPPER_SNAKE_CASE` at module level — `_VALID_ACTIONS`, `PHASE1_SYSTEM`
- **Private module members**: `_leading_underscore` — `_encode_image()`, `_find_first()`
- **Dataclass fields**: `snake_case`
- **JSON keys**: `snake_case` in episode data (matches Python attribute style)

## Code Organization

- One class per file for major components (`EBAgent`, `OracleAgent`, `BranchRunner`, `Scheduler`)
- Utility modules group related functions without a class (`context_builder.py`, `task_conditions.py`)
- `src/__init__.py` is empty — no public API surface defined
- Scripts are thin wrappers that import from `src/`
- Dataclasses used for config objects (`BranchConfig`, `BranchResult`, `SchedulerConfig`)
- No abstract base classes or interfaces — duck typing throughout

## Error Handling

- **Let it crash**: RuntimeErrors raised for unrecoverable states (invalid action after 3 retries, injection setup failure after 3 attempts, replay mismatch)
- **Retry with backoff**: VLM API calls retry on connection error (3× with 1s/2s wait), timeout (3× with escalating timeout), rate limit (3× with 5s/10s/15s wait)
- **JSON parse retry**: VLM responses retried up to 2× with explicit error feedback in prompt
- **Validation before execution**: Invalid actions blocked before reaching AI2-THOR
- **try/finally**: Used in `run_single_branch()` to guarantee `env.close()` cleanup
- No custom exception hierarchy — all errors use built-in types (`ValueError`, `RuntimeError`, `FileNotFoundError`)

## Logging

- **Python logging**: `vlm_client.py` sets up file handler (DEBUG) + stream handler (INFO)
- **Structured JSONL**: API calls → `logs/api_calls.jsonl`, failures → per-episode `failures_*.jsonl`
- **Failure dumps**: Complete API request bodies dumped to `logs/failure_*.json` on non-retryable failures
- **Console progress**: `scheduler.py` uses `\r` carriage return for in-place progress updates
- No structured logging library (no structlog, no loguru)

## Type Hints

- Used inconsistently:
  - Modern syntax: `str | None`, `dict | None`, `list[dict]`
  - Legacy syntax: `Optional[str]`, `Optional[dict]` still appears
  - Some functions have no type hints at all
  - `from __future__ import annotations` used in `context_builder.py`
- No mypy/pyright configuration

## Docstrings

- Module-level: triple-quote description at top of most files
- Class-level: sparse, some classes have docstrings (`EnvController`), others don't (`BranchRunner`)
- Function-level: minimal inline comments, no formal docstrings on most functions
- Section separators: `# ---` comment blocks used as visual separators
- No Sphinx/Google/NumPy docstring format

## Import Style

- Standard library first, then third-party, then local (`src.`)
- `from src.xxx import yyy` preferred over `import src.xxx`
- Some modules import inside functions (lazy imports for `numpy`, `PIL`, `requests`)
- `from __future__ import annotations` used in one file

## Prompt Management

- System prompts are module-level string constants (`PHASE1_SYSTEM`, `PHASE3_SYSTEM`, etc.)
- Prompt builder functions named `build_phaseN_prompt()` or `build_<role>_prompt()`
- EB prompt assembly is multi-step with conditional blocks (memory, error, failed objects)
- Multi-image prompts add direction labels: `VIEW 1: ahead`, `VIEW 2: left`, etc.

## Commit Conventions

- Messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- No conventional commits prefix (no `feat:`, `fix:`, etc.)
- Commits appear to be direct to `master` branch

## Comment Style

- Block comments with `# ---` separators
- Inline comments are sparse and functional
- Chinese appears in some comments (module docstrings)
- Section labels in `branch_runner.py`: `# ===== Phase 1: ... =====`

## Anti-Patterns Observed

- Large monolithic function: `BranchRunner.run()` is ~550 lines
- String-based dispatch: `_CHECKERS` dict maps task type strings to functions
- Hardcoded paths: `LOG_DIR = Path("logs")` at module level (no config)
- Module-level mutable state: `EnvController._xvfb_proc` as class variable
