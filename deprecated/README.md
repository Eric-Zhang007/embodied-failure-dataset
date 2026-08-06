# Deprecated

Archived pre-action-aware trap code and tests. These artifacts used random,
initial environment mutations, or exhaustive phase-specific prompt/candidate
assertions. They are intentionally outside the production runtime and default
test discovery.

Archived tests are historical records and are not maintained against current APIs.

The active path is `src/env_injector.py` plus Oracle-selected raw runtime
injections in `src/branch_runner.py`.
