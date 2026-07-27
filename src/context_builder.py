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
    """Return the ordered ancestor lineage followed by current-branch steps."""
    steps = list(episode_steps)
    history = []
    seen_ids = set()

    if parent_branch_id and shared_step_ids:
        steps_by_id = {
            step.get("step_id"): step
            for step in steps
            if step.get("step_id")
        }
        for step_id in shared_step_ids:
            if not step_id or step_id in seen_ids:
                continue
            step = steps_by_id.get(step_id)
            if step is not None:
                history.append(step)
                seen_ids.add(step_id)

    for step in steps:
        if (
            step.get("branch_id") == branch_id
            and step.get("step_id") not in seen_ids
        ):
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


# ------------------------------------------------------------------
# Unified intent tree rendering (replaces separate "Intent history" +
# "Full EB history" JSONL sections)
# ------------------------------------------------------------------

def _format_action_summary(action: str, params: dict) -> str:
    """Create a compact action summary like 'MoveSequence(MoveAhead×5)'
    or 'OpenObject(Fridge)'."""
    if action == "MoveSequence":
        steps = params.get("steps", [])
        if steps:
            parts: list[str] = []
            for s in steps:
                a = s.get("action", "?")
                r = s.get("repeat", 1)
                if r > 1:
                    parts.append(f"{a}×{r}")
                else:
                    sp = s.get("params", {})
                    ot = sp.get("objectType", "")
                    if ot:
                        parts.append(f"{a}({ot})")
                    else:
                        parts.append(a)
                if len(parts) >= 4:
                    break
            suffix = "..." if len(steps) > 4 else ""
            return f"MoveSequence({', '.join(parts)}{suffix})"
        return "MoveSequence"
    ot = params.get("objectType", "")
    rt = params.get("receptacleType", "")
    if ot and rt:
        return f"{action}({ot}, {rt})"
    if ot:
        return f"{action}({ot})"
    return action


def render_intent_tree(
    intent_history: list[dict] | None,
    action_history: list[dict] | None = None,
) -> str:
    """Render a unified intent-level tree for the Planner.

    Replaces both the old "Intent history" list and the "Full EB history"
    JSONL dump with a single compact, visually-structured tree.

    Returns empty string when intent_history is empty or None (caller
    should fall back to raw JSONL history).
    """
    if not intent_history:
        return ""

    lines = ["=== INTENT TREE ===", ""]

    for intent_entry in intent_history[-20:]:  # last 20 intents max
        intent_str = intent_entry.get("intent", "?")
        target_str = intent_entry.get("target", "")
        completed = intent_entry.get("completed", False)
        steps: list[dict] = intent_entry.get("steps", [])
        n_steps = len(steps)

        status = "OK" if completed else "INCOMPLETE"

        # ── Intent header ──
        desc = intent_str
        if target_str:
            desc += f" {target_str}"
        lines.append(
            f"▼ {desc} ({status}, {n_steps} step"
            f"{'s' if n_steps != 1 else ''})"
        )

        for step in steps:
            step_id = step.get("step_id", "?")
            action = step.get("action", "?")
            success = bool(step.get("success"))
            is_recovery = bool(step.get("recovery_step"))

            action_summary = _format_action_summary(
                action, step.get("action_params", {})
            )
            status_mark = "OK" if success else "FAIL"
            tags = []
            if is_recovery:
                tags.append("recovery")
            tag_str = f" [{' | '.join(tags)}]" if tags else ""

            lines.append(
                f"  ▸ {step_id}: {action_summary} [{status_mark}]{tag_str}"
            )

            if not success:
                error_msg = (step.get("error_message") or "")[:150]
                if error_msg:
                    lines.append(f"    └─ error: {error_msg}")
                diagnosis = (step.get("eb_diagnosis") or "")[:150]
                if diagnosis:
                    lines.append(f"    └─ diagnosed: {diagnosis}")
                recovery = (step.get("eb_recovery_reasoning") or "")[:150]
                if recovery:
                    lines.append(f"    └─ recovery: {recovery}")
                # Oracle evaluation (only if present)
                oracle_parts = []
                if step.get("oracle_diagnosis_correct") is not None:
                    oracle_parts.append(
                        f"diagnosis_correct={step['oracle_diagnosis_correct']}"
                    )
                if step.get("oracle_counterfactual_grade"):
                    oracle_parts.append(
                        f"grade={step['oracle_counterfactual_grade']}"
                    )
                if oracle_parts:
                    lines.append(
                        f"    └─ oracle: {', '.join(oracle_parts)}"
                    )
            else:
                reasoning = (step.get("eb_reasoning") or "")[:120]
                if reasoning:
                    lines.append(f"    └─ reasoning: {reasoning}")
                # Oracle fields can appear on successful steps too
                oracle_parts = []
                if step.get("oracle_diagnosis_correct") is not None:
                    oracle_parts.append(
                        f"diagnosis_correct={step['oracle_diagnosis_correct']}"
                    )
                if step.get("oracle_counterfactual_grade"):
                    oracle_parts.append(
                        f"grade={step['oracle_counterfactual_grade']}"
                    )
                if oracle_parts:
                    lines.append(
                        f"    └─ oracle: {', '.join(oracle_parts)}"
                    )

        lines.append("")

    # ── Done rejections from action_history ──
    if action_history:
        done_rejections = [
            s
            for s in action_history
            if s.get("action") == "Done"
            and not s.get("success", False)
            and s.get("error_type") == "done_rejected"
        ]
        if done_rejections:
            latest = done_rejections[-1]
            error_msg = (latest.get("error_message") or "")[:200]
            lines.append("▼ Done (REJECTED)")
            if error_msg:
                lines.append(f"  └─ rejected: {error_msg}")
            lines.append("")

    if len(lines) == 2:  # only header + blank line
        return ""

    return "\n".join(lines)


def render_current_intent_steps(
    intent: str,
    target: str,
    planner_reasoning: str,
    steps: list[dict],
    task_goal: str = "",
) -> str:
    """Render the current intent's context for the Executor.

    The Executor sees ONLY the current intent (not the full tree) plus
    the Planner's reasoning for choosing this intent.
    """
    lines = ["=== CURRENT INTENT ==="]
    lines.append(f"Intent: {intent}")
    if target:
        lines.append(f"Target: {target}")
    if planner_reasoning:
        lines.append(f"Planner's reasoning: {planner_reasoning[:200]}")
    if task_goal:
        lines.append(f"Task goal: {task_goal}")

    if steps:
        lines.append("")
        lines.append("=== STEPS IN THIS INTENT ===")
        for step in steps:
            step_id = step.get("step_id", "?")
            action = step.get("action", "?")
            success = bool(step.get("success"))
            is_recovery = bool(step.get("recovery_step"))

            action_summary = _format_action_summary(
                action, step.get("action_params", {})
            )
            status_mark = "OK" if success else "FAIL"
            tags = []
            if is_recovery:
                tags.append("recovery")
            tag_str = f" [{' | '.join(tags)}]" if tags else ""

            lines.append(
                f"  {step_id}: {action_summary} [{status_mark}]{tag_str}"
            )

            if not success:
                error_msg = (step.get("error_message") or "")[:150]
                if error_msg:
                    lines.append(f"    → error: {error_msg}")
                diagnosis = (step.get("eb_diagnosis") or "")[:150]
                if diagnosis:
                    lines.append(f"    → diagnosed: {diagnosis}")

    return "\n".join(lines)
