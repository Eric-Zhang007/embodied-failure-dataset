# Agent Context History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make EB see its own complete branch history and make Oracle see complete EB plus Oracle history.

**Status as of 2026-06-08:** Partially implemented and superseded by the current source code plus `CLAUDE.md`. Current behavior includes shared EB/Oracle history builders, LookAround as a recorded step, Phase 3 pending failed-step context, shared `HAND STATUS` / `TASK COMPLETION CRITERIA` prompt context, single-pass LookAround scanning, and per-image direction labels in multi-image requests. This file remains a historical implementation plan, not the current source of truth.

**Architecture:** Add one small context builder module. EB and Oracle stop manually slicing history. BranchRunner passes branch-aware history and records failures before diagnosis.

**Tech Stack:** Python 3.10, pytest-style unit tests with plain asserts, existing VLM prompt builders.

---

### Task 1: Shared Context Builder

**Files:**
- Create: `src/context_builder.py`
- Test: `tests/test_context_builder.py`

- [ ] Add tests for EB full history, Oracle full history, fork shared history, and current failure inclusion.
- [ ] Add compact JSONL formatting with stable key order and no truncation.
- [ ] Keep EB context free of Oracle-only fields.
- [ ] Include Oracle injection, verdict, ground truth, counterfactual grade, and fork metadata in Oracle context.

### Task 2: EB Prompt Alignment

**Files:**
- Modify: `src/eb_agent.py`
- Test: `tests/test_context_builder.py`

- [ ] Replace `action_history[-10:]` and `action_history[-20:]` prompt construction with context builder output.
- [ ] Give LookAround the same EB history, last error, and failed object context as normal action selection.
- [ ] Remove duplicated per-step history formatting from EB prompt builders.

### Task 3: Oracle Prompt Alignment

**Files:**
- Modify: `src/oracle_agent.py`
- Test: `tests/test_context_builder.py`

- [ ] Replace recent-history snippets with full Oracle context.
- [ ] Ensure Phase 2 sees proposed EB action and current EB reasoning.
- [ ] Ensure Phase 4 sees the current failed step, EB diagnosis, and previous Oracle decisions.
- [ ] Remove duplicated per-step history formatting from Oracle prompt builders.

### Task 4: BranchRunner History Correctness

**Files:**
- Modify: `src/branch_runner.py`
- Modify: `src/scheduler.py`
- Modify: `src/episode_manager.py`
- Test: `tests/test_context_builder.py`

- [ ] Build branch history from shared main steps plus current branch steps.
- [ ] Record LookAround as a real step.
- [ ] Convert object resolution failure into a failed step instead of an uncounted retry.
- [ ] Build a pending failed step before EB Phase 3 and Oracle Phase 4.
- [ ] Keep changes small; do not redesign unrelated task-completion or trap logic.

### Task 5: Training Prompt Reuse

**Files:**
- Modify: `scripts/finetune_qwen3vl.py`
- Modify: `scripts/generate_ft_data.py`

- [ ] Use EB-style history formatting for training inputs.
- [ ] Remove the old `scene | goal | plan` generator format.
- [ ] Keep output JSON shape compatible with runtime: `action`, `params`, `reasoning`.

### Task 6: Verification

**Files:**
- All touched files

- [ ] Run unit tests.
- [ ] Run Python syntax compilation for touched modules.
- [ ] Check there are no remaining `action_history[-10:]`, `action_history[-20:]`, or `action_history[-5:]` prompt slices.
- [ ] Review diff for over-engineering and delete redundant helpers.
