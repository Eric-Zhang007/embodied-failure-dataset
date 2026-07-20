"""
外部 Agent（Oracle Agent）。

负责：
- Phase 2: 决定是否注入失败（相机注入）
- Phase 4: 评估 EB Agent 的诊断/反事实/恢复，决定是否 fork
"""

import json
import logging
from pathlib import Path
import numpy as np

from src.vlm_client import VLMClient
from src.context_builder import build_oracle_history_context

logger = logging.getLogger("vlm_client")


# ------------------------------------------------------------------
# Phase 2: 相机注入决策
# ------------------------------------------------------------------

PHASE2_SYSTEM = """You are a supervisor agent overseeing an embodied agent performing a household task. Your role is to occasionally introduce failures to test the agent's recovery ability.

When the agent proposes an action, you decide whether to inject a failure into the environment. The injection modifies the environment state BEFORE the agent's proposed action executes. The modification may cause the agent's action to fail, or the effect may surface later.

Injection methods available:
- set_object_property: change a property of an object (e.g., make it static, broken, open/closed, toggled on/off)
  params: {"object_type": "<type>", "property": "<prop>", "value": true/false}
- close_container: close an open container (cabinet, fridge, drawer, etc.)
  params: {"object_type": "<type>"}
- occlude_object: block an object from the agent's view
  params: {"object_type": "<type>"}
- hide_object: move a visible pickupable object to a different non-target receptacle
  params: {"object_type": "<type>"}
- swap_object: replace a visible object with an existing object of another type
  params: {"target_type": "<type>", "new_type": "<type>"}
- remove_object: disable one object of a type so it disappears from the scene
  params: {"object_type": "<type>"}

When to inject — YOUR GOAL IS TO CREATE A CASCADING FAILURE:
A cascading failure means: the agent hits a first obstacle, begins recovering from it, and THEN a second, different obstacle appears during recovery. The second failure interrupts the recovery from the first one. This tests whether the agent can handle compound setbacks without losing track of the task.

Strategy for building a cascade:
- First injection (cascade_level=0): apply a moderate trap early — close a container, hide an object, occlude a path. The agent will encounter this naturally when it tries to interact.
- Then WAIT. Let the agent diagnose the failure and begin recovery.
- Second injection (cascade_level=1, during recovery): apply a DIFFERENT type of trap. If the first was occlusion, the second could be closing a container or swapping an object. The key is that the agent is now dealing with TWO problems at once.
- Third injection (cascade_level=1, after second recovery begins): apply yet another DIFFERENT trap — but only if the agent is still making progress. Three failures stacked is aggressive.

When NOT to inject:
- Do NOT inject when the agent is about to complete the task (PickupObject on the target object, PutObject to the target receptacle, Done). Let success happen.
- NEVER occlude_object, hide_object, or swap_object on the TASK TARGET. If the task says "move the alarm clock", you must NOT hide/occlude/swap the AlarmClock — the agent must be able to find it. Occluding non-target objects (blocking a path, obscuring a desk) is acceptable.
- Do NOT inject if cascade_level >= 2 — the agent is already buried in problems.
- Do NOT inject on purely passive actions: Done, LookDown, RotateLeft, RotateRight.
- Do NOT repeat the same injection method on the same object type — each trap should be a NOVEL challenge.

Limit: maximum 3 injections per episode total, across all cascade levels.

INJECTION COUNTDOWN: each time you inject, your remaining quota decreases. You will see "INJECTIONS REMAINING: N" in the prompt. When it reaches 0, your injections are blocked automatically — the system will reject any further inject=true responses.

OUTPUT FORMAT — VALID JSON ONLY:
- { must be the FIRST character of your response, } must be the LAST character.
- NO text outside braces. NO markdown fences. NO "//" comments (JSON does not allow comments).
- The closing } is REQUIRED. Truncated JSON = failed action.

{
  "inject": true/false,
  "reasoning": "<why you chose to inject or not>",
  "injection": {
    "method": "<method name>",
    "params": {...}
  } or null
}"""


PHASE2_SYSTEM_FORK = """You are a supervisor agent overseeing an embodied agent. This is a counterfactual branch — the agent is trying an alternative path after realizing a past mistake.

The agent DOES NOT know it is in a counterfactual branch. Do not reveal this.

Your injection rules are the same as usual, EXCEPT:
- Do NOT inject failures that would repeat the same type of failure the agent is trying to avoid
- Focus on NEW types of failures that test different recovery skills
- Same 3-injection countdown applies — check "INJECTIONS REMAINING" in the prompt

OUTPUT FORMAT — VALID JSON ONLY:
- { must be the FIRST character, } must be the LAST character.
- NO text outside braces. NO markdown. The closing } is REQUIRED.

{
  "inject": true/false,
  "reasoning": "<why>",
  "injection": {"method": "<method>", "params": {...}} or null
}"""


def build_phase2_prompt(
    task_goal: str,
    proposed_action: str,
    proposed_params: dict,
    eb_reasoning: str,
    env_state: dict,
    action_history: list[dict],
    cascade_level: int,
    is_fork: bool,
    remaining_injections: int = 3,
    injection_attempt_errors: list[dict] | None = None,
) -> str:
    lines = [f"Task goal: {task_goal}\n"]
    lines.append(f"Agent proposes: {proposed_action}({json.dumps(proposed_params)})")
    if eb_reasoning:
        lines.append(f"Agent reasoning: {eb_reasoning}")

    objects = env_state.get("objects", [])
    visible = [o for o in objects if o.get("visibleBounds2D")]
    if visible:
        lines.append(f"\nVisible objects ({len(visible)}):")
        for o in visible:
            props = []
            if o.get("pickupable"): props.append("pickupable")
            if o.get("openable"): props.append("open" if o.get("isOpen") else "closed")
            if o.get("toggleable"): props.append("on" if o.get("isToggled") else "off")
            if o.get("isPickedUp"): props.append("held")
            if o.get("receptacle"): props.append("receptacle")
            prop_str = f" ({', '.join(props)})" if props else ""
            lines.append(f"  {o['objectId']} ({o['objectType']}){prop_str}")

    lines.append(f"\nCascade level: {cascade_level}")
    lines.append(f"Fork branch: {is_fork}")

    lines.append("")
    lines.append(build_oracle_history_context(action_history))

    if remaining_injections <= 0:
        lines.append(f"\nINJECTION LIMIT REACHED: you have used all your injections. You MUST set inject=false.")
    else:
        lines.append(f"\nINJECTIONS REMAINING: {remaining_injections}. Use them wisely — each one counts.")
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
    lines.append("\nDecide whether to inject. OUTPUT: valid JSON only, { first char, } last char.")
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
- "recovered": the proposed recovery action successfully resolves the current failure
- "recoverable": recovery not yet achieved but the task is still achievable
- "unrecoverable": the task goal is permanently unreachable OR the agent is stuck in a dead loop.

DEAD LOOP DETECTION: If the agent has repeated the SAME failed action 5+ times in recent history (e.g., MoveAhead blocked by the same obstacle 5+ times, or PickupObject failing on the same objectId 3+ times), and continues to try the same approach without changing strategy, the task is UNRECOVERABLE. A stuck agent that cannot adapt its behavior is effectively deadlocked. Mark such cases as "unrecoverable".

COUNTERFACTUAL_GOLD FORMAT — this is the CORRECTED counterfactual that a fork will actually execute, so it must be concrete and replayable:
- target_step: the integer step index (step_index_in_branch) where the agent should have acted differently.
- alternative_action: {"action": "<exact AI2-THOR action>", "params": {...}} — the action that should have been taken at target_step. Use objectType for interactions (e.g. {"action": "PickupObject", "params": {"objectType": "Egg"}}).
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
  "recovery_verdict": "recovered" / "recoverable" / "unrecoverable",
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
) -> str:
    lines = [f"Task goal: {task_goal}\n"]
    lines.append(f"Error encountered: {error_message}\n")

    lines.append(build_oracle_history_context(action_history))

    lines.append(f"\nAgent's diagnosis: {eb_diagnosis}")
    lines.append(f"Agent's recovery reasoning: {eb_recovery_reasoning}")
    lines.append(f"Agent's counterfactual: {eb_counterfactual or 'None'}")
    lines.append(f"Agent's proposed recovery: {json.dumps(eb_proposed_recovery)}")

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
        remaining_injections: int = 3,
        injection_attempt_errors: list[dict] | None = None,
    ) -> dict:
        """Phase 2: 决定是否注入失败。"""
        system = PHASE2_SYSTEM_FORK if is_fork else PHASE2_SYSTEM
        prompt = build_phase2_prompt(
            task_goal, proposed_action, proposed_params, eb_reasoning,
            env_state, action_history, cascade_level, is_fork, remaining_injections,
            injection_attempt_errors,
        )
        result = self.client.chat_with_image_json(
            system_prompt=system,
            user_text=prompt,
            image=image,
            required_fields=("inject", "reasoning"),
        )
        if "injection" not in result:
            result["injection"] = None
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
    ) -> dict:
        """Phase 4: 评估 EB 的诊断和恢复，决定是否 fork。"""
        prompt = build_phase4_prompt(
            task_goal, error_message, action_history,
            eb_diagnosis, eb_recovery_reasoning, eb_counterfactual, eb_proposed_recovery,
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


_AGENTIC_FAILURE_LIBRARY = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "failure_type_library.json").read_text(encoding="utf-8")
)
_AGENTIC_FAILURE_LIBRARY_TEXT = json.dumps(_AGENTIC_FAILURE_LIBRARY, ensure_ascii=False, indent=2)

_AGENTIC_TOOLS = """Available intervention tools:
- set_object_property: {"object_type": "<type>", "property": "<prop>", "value": true/false}
- close_container: {"object_type": "<type>"}
- occlude_object: {"object_type": "<type>"}
- hide_object: {"object_type": "<type>"}
- swap_object: {"target_type": "<type>", "new_type": "<type>"}
- remove_object: {"object_type": "<type>"}"""

_AGENTIC_REVIEW_SYSTEM = f"""You are an autonomous Oracle supervising an embodied household agent.
After sustained successful progress, decide whether a useful, recoverable intervention would create an informative failure, or whether to keep observing. Never make the task permanently impossible and do not target an object when doing so would make the stated goal unachievable.

{_AGENTIC_TOOLS}

The following failure library is inspiration, not a restriction:
{_AGENTIC_FAILURE_LIBRARY_TEXT}

Return valid JSON only:
{{"action": "intervene" or "observe", "intervention": {{"tool": "<tool>", "params": {{...}}}} or null, "reasoning": "<brief reason>"}}"""

_AGENTIC_FAILURE_SYSTEM = f"""You are an autonomous Oracle supervising an embodied household agent after an environment action failed.
Choose fork to test a concrete alternative at an earlier existing history step, terminate only when continued execution is not useful or the task is unrecoverable, or observe to let the agent continue planning without Oracle diagnosis or recovery.

A fork_plan must use this exact replayable shape:
{{"target_step": <existing step_index_in_branch>, "alternative_action": {{"action": "<AI2-THOR action>", "params": {{...}}}}, "reasoning": "<brief reason>"}}

The following failure library is inspiration, not a restriction:
{_AGENTIC_FAILURE_LIBRARY_TEXT}

Return valid JSON only:
{{"action": "fork" or "terminate" or "observe", "fork_plan": <structured fork plan> or null, "reasoning": "<brief reason>"}}"""


class AgenticOracle:
    def __init__(self, client: VLMClient):
        self.client = client
        self._success_count = 0

    def periodic_review(
        self,
        image: np.ndarray,
        task_goal: str,
        visible_objects: list[dict],
        action_history: list[dict],
        memory_text: str,
    ) -> dict:
        self._success_count += 1
        if self._success_count % 3:
            return {
                "action": "observe",
                "intervention": None,
                "reasoning": "Periodic review is due after three successful steps.",
            }

        prompt = "\n".join([
            f"Task goal: {task_goal}",
            f"Visible objects: {json.dumps(visible_objects, ensure_ascii=False)}",
            memory_text,
            build_oracle_history_context(action_history),
            "Review current progress and decide whether to intervene.",
        ])
        result = self.client.chat_with_image_json(
            system_prompt=_AGENTIC_REVIEW_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("action", "intervention", "reasoning"),
        )
        if result.get("action") not in ("intervene", "observe"):
            result["action"] = "observe"
        if result["action"] != "intervene":
            result["intervention"] = None
        return result

    def analyze_failure(
        self,
        image: np.ndarray,
        task_goal: str,
        error_message: str,
        action_history: list[dict],
        memory_text: str,
    ) -> dict:
        prompt = "\n".join([
            f"Task goal: {task_goal}",
            f"Environment error: {error_message}",
            memory_text,
            build_oracle_history_context(action_history),
            "Choose whether to fork, terminate, or observe.",
        ])
        result = self.client.chat_with_image_json(
            system_prompt=_AGENTIC_FAILURE_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("action", "fork_plan", "reasoning"),
        )
        if result.get("action") not in ("fork", "terminate", "observe"):
            result["action"] = "observe"
        if result["action"] != "fork":
            result["fork_plan"] = None
        return result
