"""Compact context rendering for EB and Oracle prompts."""

from __future__ import annotations

import json
from typing import Iterable

from src.action_adapter import clean_history_params


EB_FIELDS = (
    "eb_reasoning",
    "eb_diagnosis",
    "eb_recovery_reasoning",
    "eb_counterfactual",
    "eb_proposed_recovery_action",
)

ORACLE_FIELDS = (
    "oracle_injection_decision",
    "oracle_diagnosis_correct",
    "oracle_ground_truth",
    "oracle_counterfactual_grade",
    "oracle_counterfactual_gold",
    "oracle_recovery_verdict",
    "fork_decision",
    "fork_metadata",
)


def build_branch_history(
    episode_steps: Iterable[dict],
    branch_id: str,
    parent_branch_id: str | None = None,
    shared_step_ids: Iterable[str] | None = None,
) -> list[dict]:
    """Return shared parent steps followed by current-branch steps."""
    steps = list(episode_steps)
    shared_ids = set(shared_step_ids or [])
    history = []

    if parent_branch_id and shared_ids:
        for step in steps:
            if step.get("branch_id") == parent_branch_id and step.get("step_id") in shared_ids:
                history.append(step)

    for step in steps:
        if step.get("branch_id") == branch_id:
            history.append(step)

    return history


def build_eb_history_context(action_history: Iterable[dict], current_step: dict | None = None) -> str:
    """Render the complete EB-visible history as compact JSONL."""
    return _build_history_context(
        action_history,
        current_step=current_step,
        include_oracle=False,
        empty_text="No previous EB actions.",
        header="Full EB history:",
    )


def build_oracle_history_context(action_history: Iterable[dict], current_step: dict | None = None) -> str:
    """Render the complete Oracle-visible history as compact JSONL."""
    return _build_history_context(
        action_history,
        current_step=current_step,
        include_oracle=True,
        empty_text="No previous EB or Oracle history.",
        header="Full EB + Oracle history:",
    )


def _build_history_context(
    action_history: Iterable[dict],
    current_step: dict | None,
    include_oracle: bool,
    empty_text: str,
    header: str,
) -> str:
    steps = list(action_history)
    if current_step is not None:
        steps.append(current_step)
    if not steps:
        return empty_text

    lines = [header]
    for step in steps:
        lines.append(_render_step(step, include_oracle=include_oracle))
    return "\n".join(lines)


def _render_step(step: dict, include_oracle: bool) -> str:
    row = {
        "branch": step.get("branch_id"),
        "step": step.get("step_index_in_branch"),
        "id": step.get("step_id"),
        "action": step.get("action"),
        "params": clean_history_params(step.get("action_params", {}) or {}),
        "success": bool(step.get("success")),
    }

    if step.get("error_message"):
        row["error"] = step["error_message"]

    eb = {name[3:]: step.get(name) for name in EB_FIELDS if step.get(name) is not None}
    if eb:
        row["eb"] = eb

    if include_oracle:
        oracle = {
            _public_oracle_key(name): step.get(name)
            for name in ORACLE_FIELDS
            if step.get(name) is not None
        }
        if oracle:
            row["oracle"] = oracle

    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _public_oracle_key(name: str) -> str:
    return name[7:] if name.startswith("oracle_") else name
