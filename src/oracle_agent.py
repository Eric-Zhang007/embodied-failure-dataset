"""
外部 Agent（Oracle Agent）。

负责：
- Phase 2: 决定是否注入失败（相机注入）
- Phase 4: 评估 EB Agent 的诊断/反事实/恢复，决定是否 fork
"""

import json
import logging
import numpy as np

from src.vlm_client import VLMClient
from src.context_builder import build_oracle_history_context

logger = logging.getLogger("vlm_client")


# ------------------------------------------------------------------
# Phase 2: 相机注入决策
# ------------------------------------------------------------------

CONSTRAINED_PHASE2_SYSTEM = """You are a supervisor agent designing recovery-focused failures for an embodied agent.

You choose an injection method and its object parameters. The runtime accepts only the known methods below and rejects invalid AI2-THOR actions or destructive changes to task-critical objects. Do not invent methods or object types.

Allowed methods:
- set_object_property: {"object_type": "<scene type>", "property": "is_open"|"is_toggled"|"fillLiquid"|"emptyLiquid", "value": ...}
- close_container: {"object_type": "<openable scene type>"}
- hide_object: {"object_type": "<visible pickupable scene type>", "container_type": "<different closed openable receptacle type>"}
- drop_held_object: {}
- stash_held_object: {"container_type": "<non-destination closed openable receptacle type>"}

Every object-targeting method also accepts the matching visible `object_id` (and container methods accept `container_id`). Use IDs when multiple instances share a type.

TRAP STATE is the only ground truth for prior mutations. At a normal action, do not add a new trap while one is active. At recovery time, an earlier trap is still triggered: you may install one different later trap BEFORE that recovery executes. That is a cascade. The second trap may affect a later task action; it does not need to interrupt the recovery action itself.

When injecting, describe the expected failed action/error marker and the recovery action. Use real action names and objectType/objectId parameters. Skip when the state cannot stay recoverable or a similar intervention adds no value.
For a closed receptacle, use the error marker "closed"; AI2-THOR reports CLOSED rather than "not open".
For a hidden pickup target, use "specified visibility"; AI2-THOR does not report "not visible" verbatim.

OUTPUT FORMAT: valid JSON only.
{
  "inject": true/false,
  "reasoning": "brief timing and causal rationale",
  "injection": {"method": "<allowed method>", "params": {}} or null,
  "expected_failure": {"action": "<AI2-THOR action>", "error_marker": "<lowercase substring expected in error>"} or null,
  "recovery_action": {"action": "<AI2-THOR action>", "params": {}} or null
}"""


CONSTRAINED_PHASE2_SYSTEM_FORK = CONSTRAINED_PHASE2_SYSTEM + """

This is a counterfactual branch. Prefer an effect not already used on the parent path."""


def build_phase2_prompt(
    task_goal: str,
    proposed_action: str,
    proposed_params: dict,
    eb_reasoning: str,
    env_state: dict,
    action_history: list[dict],
    cascade_level: int,
    is_fork: bool,
    injection_attempt_errors: list[dict] | None = None,
    trap_state: list[dict] | None = None,
    recovery_time: bool = False,
) -> str:
    lines = [f"Task goal: {task_goal}\n"]
    lines.append(f"Agent proposes: {proposed_action}({json.dumps(proposed_params)})")
    if eb_reasoning:
        lines.append(f"Agent reasoning: {eb_reasoning}")

    objects = env_state.get("objects", [])
    inventory = env_state.get("inventoryObjects", [])
    if inventory:
        lines.append("\nAgent is holding:")
        for obj in inventory:
            lines.append(f"  {obj.get('objectId')} ({obj.get('objectType')})")
    visible = [o for o in objects if o.get("visible")]
    if visible:
        lines.append(f"\nVisible objects ({len(visible)}):")
        for o in visible:
            props = []
            if o.get("pickupable"): props.append("pickupable")
            if o.get("openable"): props.append("open" if o.get("isOpen") else "closed")
            if o.get("toggleable"): props.append("on" if o.get("isToggled") else "off")
            if o.get("canFillWithLiquid"): props.append("filled" if o.get("isFilledWithLiquid") else "empty")
            if o.get("isPickedUp"): props.append("held")
            if o.get("receptacle"): props.append("receptacle")
            prop_str = f" ({', '.join(props)})" if props else ""
            lines.append(f"  {o['objectId']} ({o['objectType']}){prop_str}")

    lines.append(f"\nCascade level: {cascade_level}")
    lines.append(f"Fork branch: {is_fork}")
    lines.append(f"Recovery-time injection: {recovery_time}")

    lines.append("")
    lines.append(build_oracle_history_context(action_history))

    lines.append("\nTRAP STATE (environment-confirmed):")
    if trap_state:
        for trap in trap_state:
            summary = {
                key: trap.get(key)
                for key in (
                    "trap_id", "created_by", "status", "injection", "expected_failure",
                    "recovery_action", "setup_result", "modification_success", "recovered_by_action",
                    "created_at_step_id", "triggered_at_step_id",
                    "recovered_at_step_id", "env_error", "trigger_events", "recovery_events",
                )
                if trap.get(key) is not None
            }
            lines.append("  " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        lines.append("  None. No prior trap has been applied in this branch.")

    actionable_types = {}
    for obj in objects:
        if obj.get("pickupable") or obj.get("openable") or obj.get("toggleable"):
            actionable_types.setdefault(obj.get("objectType"), []).append(obj)
    if actionable_types:
        lines.append("\nSCENE TOOL INVENTORY (may be outside the current view):")
        for object_type, entries in sorted(actionable_types.items()):
            props = set()
            for obj in entries:
                if obj.get("pickupable"):
                    props.add("pickupable")
                if obj.get("openable"):
                    props.add("openable")
                if obj.get("receptacle"):
                    props.add("receptacle")
                if obj.get("toggleable"):
                    props.add("toggleable")
                if obj.get("breakable"):
                    props.add("breakable")
                if obj.get("canFillWithLiquid"):
                    props.add("fillable")
            lines.append(f"  {object_type} x{len(entries)} ({', '.join(sorted(props))})")
    if injection_attempt_errors:
        lines.append("\nPrevious injection setup attempts for THIS SAME EB action failed:")
        for item in injection_attempt_errors:
            lines.append(
                "  - method="
                + str(item.get("method"))
                + " params="
                + json.dumps(item.get("params", {}), ensure_ascii=False)
                + " error="
                + str(item.get("error"))
            )
        lines.append("Choose a different injection that can actually be applied, or set inject=false.")
    lines.append("\nDecide whether to inject. Use only an allowed method and scene object type. OUTPUT: valid JSON only, { first char, } last char.")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Phase 4: 评估与 fork 决策
# ------------------------------------------------------------------

PHASE4_SYSTEM = """You are a supervisor agent evaluating an embodied agent's response to a failure.

You will receive:
1. What the agent saw (image, error message, task goal, action history)
2. The agent's diagnosis of the failure
3. The agent's recovery reasoning
4. The agent's counterfactual reasoning (if any)
5. The agent's proposed recovery action

Your job:
1. Judge whether the agent's diagnosis is correct
2. Provide the ground truth explanation of what really caused the failure
3. Grade the counterfactual reasoning (WA / PA / AC)
4. If WA or PA, provide a corrected counterfactual — as a STRUCTURED object so it can be replayed as a fork
5. Judge whether the recovery is reasonable and whether the task is still recoverable
6. Decide whether to create a fork branch to test the counterfactual

Recovery verdict meanings:
- "recoverable": the proposed recovery action should resolve the current failure and the task is still achievable
- "unrecoverable": the task goal is permanently unreachable OR the agent is stuck in a dead loop.

DEAD LOOP DETECTION: If the agent has repeated the SAME failed action 5+ times in recent history (e.g., MoveAhead blocked by the same obstacle 5+ times, or PickupObject failing on the same objectId 3+ times), and continues to try the same approach without changing strategy, the task is UNRECOVERABLE. A stuck agent that cannot adapt its behavior is effectively deadlocked. Mark such cases as "unrecoverable".

COUNTERFACTUAL_GOLD FORMAT — this is the CORRECTED counterfactual that a fork will actually execute, so it must be concrete and replayable:
- target_step: the integer step index (step_index_in_branch) where the agent should have acted differently.
- alternative_action: {"action": "<action>", "params": {...}} — the SINGLE action that should have been taken at target_step.
  Use objectType for interactions (e.g. {"action": "PickupObject", "params": {"objectType": "Egg"}}).
  ALLOWED actions: MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight, LookUp, LookDown,
  PickupObject, PutObject, OpenObject, CloseObject, ToggleObjectOn, ToggleObjectOff,
  SliceObject, BreakObject, FillObjectWithLiquid, EmptyLiquidFromObject, DropHandObject,
  MoveSequence (use params: {"steps": [{"action": "...", "repeat": N}, ...]} for multi-step moves).
  FORBIDDEN: Done, LookAround — these are meta-actions that cannot be executed as a fork step.
- reasoning: 1 sentence on why this alternative would have prevented the failure.
Set counterfactual_gold to null ONLY when the grade is AC (the agent's own counterfactual was already correct) or when no earlier decision could have prevented the failure.

OUTPUT FORMAT — VALID JSON ONLY:
- { must be FIRST char, } must be LAST char. NO text outside braces. NO markdown.
- The closing } is REQUIRED. Truncated JSON = failed evaluation.

{
  "diagnosis_correct": true/false,
  "ground_truth": "<real cause of the failure>",
  "counterfactual_grade": "WA" / "PA" / "AC",
  "counterfactual_gold": {
    "target_step": <step_index_integer>,
    "alternative_action": {"action": "<action>", "params": {}},
    "reasoning": "<1 sentence: why this would have prevented the failure>"
  } or null,
  "recovery_verdict": "recoverable" / "unrecoverable",
  "should_fork": true/false,
  "fork_reasoning": "<why fork or not>" or null
}"""


def build_phase4_prompt(
    task_goal: str,
    error_message: str,
    action_history: list[dict],
    eb_diagnosis: str,
    eb_recovery_reasoning: str,
    eb_counterfactual: str | None,
    eb_proposed_recovery: dict,
    trap_state: list[dict] | None = None,
) -> str:
    lines = [f"Task goal: {task_goal}\n"]
    lines.append(f"Error encountered: {error_message}\n")

    lines.append(build_oracle_history_context(action_history))

    lines.append(f"\nAgent's diagnosis: {eb_diagnosis}")
    lines.append(f"Agent's recovery reasoning: {eb_recovery_reasoning}")
    lines.append(f"Agent's counterfactual: {eb_counterfactual or 'None'}")
    lines.append(f"Agent's proposed recovery: {json.dumps(eb_proposed_recovery)}")

    lines.append("\nENVIRONMENT-CONFIRMED TRAP STATE:")
    if trap_state:
        for trap in trap_state:
            summary = {
                key: trap.get(key)
                for key in (
                    "trap_id", "created_by", "status", "injection", "expected_failure",
                    "recovery_action", "setup_result", "env_error",
                    "triggered_at_step_id", "recovered_at_step_id",
                    "trigger_events", "recovery_events",
                )
                if trap.get(key) is not None
            }
            lines.append("  " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        lines.append("  None.")
    lines.append(
        "Treat a failure as injection-caused only when a triggered trap's "
        "recorded effect and environment error match the current failure."
    )

    lines.append("\nEvaluate the agent's response and decide whether to fork. OUTPUT: valid JSON only, { first char, } last char.")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Oracle Agent 顶层
# ------------------------------------------------------------------

class OracleAgent:
    def __init__(self, client: VLMClient):
        self.client = client

    def decide_injection(
        self,
        task_goal: str,
        image: np.ndarray,
        env_state: dict,
        proposed_action: str,
        proposed_params: dict,
        eb_reasoning: str,
        action_history: list[dict],
        cascade_level: int,
        is_fork: bool = False,
        recovery_time: bool = False,
        injection_attempt_errors: list[dict] | None = None,
        trap_state: list[dict] | None = None,
    ) -> dict:
        """Phase 2: 决定是否注入失败。"""
        system = CONSTRAINED_PHASE2_SYSTEM_FORK if is_fork else CONSTRAINED_PHASE2_SYSTEM
        prompt = build_phase2_prompt(
            task_goal, proposed_action, proposed_params, eb_reasoning,
            env_state, action_history, cascade_level, is_fork,
            injection_attempt_errors, trap_state, recovery_time,
        )
        result = self.client.chat_with_image_json(
            system_prompt=system,
            user_text=prompt,
            image=image,
            required_fields=("inject", "reasoning", "injection"),
        )
        if not result.get("inject"):
            result["injection"] = None
            result["expected_failure"] = None
            result["recovery_action"] = None
        return result

    def evaluate_failure(
        self,
        task_goal: str,
        error_message: str,
        image: np.ndarray,
        action_history: list[dict],
        eb_diagnosis: str,
        eb_recovery_reasoning: str,
        eb_counterfactual: str | None,
        eb_proposed_recovery: dict,
        trap_state: list[dict] | None = None,
    ) -> dict:
        """Phase 4: 评估 EB 的诊断和恢复，决定是否 fork。"""
        prompt = build_phase4_prompt(
            task_goal, error_message, action_history,
            eb_diagnosis, eb_recovery_reasoning, eb_counterfactual, eb_proposed_recovery,
            trap_state,
        )
        result = self.client.chat_with_image_json(
            system_prompt=PHASE4_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=(
                "diagnosis_correct",
                "ground_truth",
                "counterfactual_grade",
                "counterfactual_gold",
                "recovery_verdict",
                "should_fork",
                "fork_reasoning",
            ),
        )
        return result
