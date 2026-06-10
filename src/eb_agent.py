"""
EB Agent（具身 Agent）。

负责：
- Phase 1: 根据任务目标和当前状态提出下一步动作
- Phase 3: 遇到失败后进行诊断、恢复推理、可选的的反事实推理
"""

import math
import numpy as np

from src.vlm_client import VLMClient
from src.context_builder import build_eb_history_context



# ------------------------------------------------------------------
# Phase 1: 动作提议
# ------------------------------------------------------------------

PHASE1_SYSTEM = """You are an embodied agent in a 3D household. The image is your FIRST-PERSON VIEW. You occupy a physical body; every action moves or rotates you.

RULES:
- Interaction range is 0.5m. Check distance BEFORE PickupObject/PutObject/etc. If >0.5m, MoveAhead (0.25m/step) first. Never interact beyond 0.5m.
- When MoveAhead BLOCKED: do NOT retry same direction and do NOT LookAround. Instead, Rotate 90deg and try there. If blocked in all 4 directions, MoveBack to escape the tight spot. LookAround is useless when you are boxed in — you already know you're stuck, you need to MOVE.
- When target not visible AND you have open space around you: rotate to scan. Use direction hints in the object list.
- If the object list below is empty: you are facing a wall or obstacle. DO NOT LookAround — Rotate or MoveBack to find open space, then locate your target.
- Use all 4 movement directions. Sidestep (MoveLeft/Right) to go around obstacles.
- If hitting same obstacle repeatedly: MoveBack then wider path.

AVAILABLE ACTIONS — use EXACT names and syntax:

Navigation:
  MoveAhead          — forward 0.25m
  MoveBack           — backward 0.25m
  MoveLeft           — strafe left 0.25m
  MoveRight          — strafe right 0.25m
  RotateLeft         — turn 90deg left
  RotateRight        — turn 90deg right
  LookUp             — tilt camera up
  LookDown           — tilt camera down
  LookAround         — 4-direction scan. Use when target is lost.

Object interaction — use objectType from the visible objects list. Must be within 0.5m reach:
  PickupObject(objectType)
  PutObject(objectType, receptacleType)   — objectType=what you hold, receptacleType=TASK TARGET (not Floor!)
  OpenObject(objectType) / CloseObject(objectType)
  ToggleObjectOn(objectType) / ToggleObjectOff(objectType)
  SliceObject(objectType) / BreakObject(objectType)
  FillObjectWithLiquid(objectType) / EmptyLiquidFromObject(objectType)
  DropHandObject

Multi-step movement:
  MoveSequence(steps)  — chain multiple movements in ONE action. Steps: [{"action": "MoveAhead", "repeat": 5}, {"action": "MoveLeft", "repeat": 2}]. Repeat defaults to 1. Execution stops on first failure. Use this INSTEAD of MoveAhead+LookAround loops: scan ONCE, then use MoveSequence to close the distance.

Task control:
  Done   — call ONLY when ALL completion criteria are met.

OBJECT NAMES: Copy objectType EXACTLY from the visible list. "Clock" is wrong; "AlarmClock" is correct. Names are case-sensitive.

RESPONSE FORMAT — valid JSON only. { first char, } last char. No markdown, no text outside braces.
Reasoning uses 4-part format: "scene: ... | goal: ... | plan: ... | reflection: ..." (1-3 sentences each).

{
  "action": "<exact action name>",
  "params": {},
  "reasoning": "scene: <what you see> | goal: <progress & sub-goal> | plan: <why this action> | reflection: <check assumptions, learn from errors>"
}

Params examples:
  PickupObject        -> {"objectType": "AlarmClock"}
  PutObject           -> {"objectType": "AlarmClock", "receptacleType": "Desk"}
  OpenObject/Close... -> {"objectType": "Cabinet"}
  RotateLeft/Move...  -> {} (no params)"""



def _egocentric_angle(agent_pos, agent_rot_y, obj_pos):
    """Return (angle_degrees, distance_m) of object relative to agent facing.
    angle=0 is straight ahead, positive=right, negative=left.
    """
    dx = obj_pos["x"] - agent_pos["x"]
    dz = obj_pos["z"] - agent_pos["z"]
    rad = math.radians(agent_rot_y)
    fx, fz = math.sin(rad), math.cos(rad)
    forward = dx * fx + dz * fz
    right = dx * fz - dz * fx
    angle = math.degrees(math.atan2(right, forward))
    d = math.sqrt(dx*dx + dz*dz)
    return angle, d


def _is_in_front(angle: float) -> bool:
    """True if object is within ±90° of agent facing (roughly in camera frustum)."""
    return -90 <= angle <= 90


def _direction(agent_pos, agent_rot_y, obj_pos):
    angle, d = _egocentric_angle(agent_pos, agent_rot_y, obj_pos)
    ds = f"{d:.1f}m"
    if -22.5 <= angle <= 22.5:       return f"ahead ({ds})"
    elif 22.5 < angle <= 67.5:       return f"ahead-right ({ds})"
    elif 67.5 < angle <= 112.5:      return f"right ({ds})"
    elif 112.5 < angle <= 157.5:     return f"behind-right ({ds})"
    elif angle > 157.5 or angle < -157.5: return f"behind ({ds})"
    elif -157.5 <= angle < -112.5:   return f"behind-left ({ds})"
    elif -112.5 <= angle < -67.5:    return f"left ({ds})"
    elif -67.5 <= angle < -22.5:     return f"ahead-left ({ds})"
    return f"? ({ds})"


def _append_task_context(
    lines: list[str],
    visible_objects: list[dict] | None,
    inventory_objects: list[dict] | None,
    task_criteria: str,
) -> None:
    visible_objects = visible_objects or []
    inventory = inventory_objects or []
    if inventory:
        obj = inventory[0]
        lines.append(f"HAND STATUS: holding {obj.get('objectType', '?')} ({obj.get('objectId', '?')})")
    else:
        held_visible = [o for o in visible_objects if o.get("isPickedUp")]
        if held_visible:
            h = held_visible[0]
            lines.append(f"HAND STATUS: holding {h['objectType']} ({h.get('objectId', '?')})")
        else:
            lines.append("HAND STATUS: empty — nothing in hand")
    lines.append("")

    if task_criteria:
        lines.append(f"TASK COMPLETION CRITERIA — the task is done ONLY when ALL of these are true:\n{task_criteria}")
        lines.append("If Done was rejected, it means one or more criteria are NOT met yet.\n")


def build_phase1_prompt(
    task_goal: str,
    visible_objects: list[dict],
    action_history: list[dict],
    last_error: str | None,
    failed_object_ids: set = None,
    agent_pos: dict = None,
    agent_rot_y: float = 0.0,
    inventory_objects: list[dict] | None = None,
    task_criteria: str = "",
    memory_text: str = "",
) -> str:
    lines = [f"Task goal: {task_goal}\n"]

    # Compressed memory rendered FIRST so VLM reads spatial context early
    if memory_text:
        lines.append(memory_text)
        lines.append("")

    _append_task_context(lines, visible_objects, inventory_objects, task_criteria)

    if failed_object_ids is None:
        failed_object_ids = set()

    visible = [o for o in visible_objects if o.get("visibleBounds2D")]
    receptacles = [o for o in visible if o.get("receptacle")]
    if receptacles:
        lines.append("Available RECEPTACLES (for PutObject) — pick the one matching your TASK GOAL:")
        for o in receptacles:
            dir_label = ""
            if agent_pos and o.get("position"):
                dir_label = " <- " + _direction(agent_pos, agent_rot_y, o["position"])
            lines.append(f"  {o['objectType']}{dir_label}")
        lines.append("")
    if visible:
        lines.append("Objects in view — use the TYPE name. Direction shown relative to your facing:")
        for o in visible:
            extra = []
            if o.get("isPickedUp"):
                extra.append("held")
            if o.get("receptacle"):
                extra.append("receptacle")
            if o.get("openable"):
                extra.append("openable" if not o.get("isOpen") else "open")
            if o.get("toggleable"):
                extra.append("on" if o.get("isToggled") else "off")
            tag = f" ({', '.join(extra)})" if extra else ""
            prev = " [failed before]" if o.get("objectId") in failed_object_ids else ""
            dir_label = ""
            if agent_pos and o.get("position"):
                dir_label = " <- " + _direction(agent_pos, agent_rot_y, o["position"])
            lines.append(f"  {o['objectType']}{tag}{prev}{dir_label}")
    else:
        lines.append("(No objects currently in view — you are likely facing a wall or obstacle. Rotate or MoveBack to find open space. Do NOT LookAround from here.)")

    if failed_object_ids:
        lines.append("\nWARNING: The following objectIds were tried and FAILED. Do NOT propose them again:")
        for fid in failed_object_ids:
            lines.append(f"  - {fid}")

    lines.append("")

    # Fall back to full JSON history only when ego memory is unavailable
    if not memory_text:
        lines.append(build_eb_history_context(action_history))

    if last_error:
        lines.append(f"\nLast action error: {last_error}")

    # Recent failure insights — only show when ego memory is not providing them
    if not memory_text:
        recent_diag = None
        for s in reversed(action_history):
            if s.get('eb_diagnosis'):
                recent_diag = s
                break
        if recent_diag:
            lines.append(f"\nRECENT FAILURE INSIGHT (step {recent_diag['step_index_in_branch']}):")
            lines.append(f"  Diagnosis: {recent_diag['eb_diagnosis']}")
            if recent_diag.get('eb_recovery_reasoning'):
                lines.append(f"  Recovery plan: {recent_diag['eb_recovery_reasoning']}")
            lines.append("  LEARN FROM THIS. Do NOT repeat the same action pattern that led to this failure.")

    if failed_object_ids:
        lines.append("Note: objects marked [failed before] may work if you get closer first.")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Planner mode: high-level intents (experimental, not yet integrated)
# ------------------------------------------------------------------

PLANNER_SYSTEM = """You are a task planner. Output the NEXT intent (not an action). Available: locate X, pick X, place X, open X, close X, toggle X, clean X, Done.
OUTPUT: {"intent": "<intent>", "target": "<objectType>", "reasoning": "<1 sentence>"}"""


# ------------------------------------------------------------------
# Phase 3: 失败诊断与恢复推理
# ------------------------------------------------------------------

PHASE3_SYSTEM = """You are an embodied agent that has just encountered a failure while performing a household task. Remember: you are in a 3D first-person environment — the image is what your eyes see, and you must reason about spatial relationships.

Your job is to:
0. FIRST: check your assumptions. Look at the image carefully. Is the target object where you THOUGHT it was? Could it be somewhere else (behind you, in a closed container, in a different room)? The error message and the objects list tell you what's actually around you — believe them over your memory.
1. Diagnose WHY the failure happened — be specific about what the error and image together reveal
2. Propose a recovery action to get back on track
3. Optionally, provide a counterfactual: "If I had done X instead of Y earlier, this failure would not have occurred."

AVAILABLE ACTIONS (use EXACTLY these names, do NOT invent new ones):

NAVIGATION:
- MoveAhead: move forward 0.25m. If BLOCKED, Rotate to find a clear path.
- MoveBack: move backward 0.25m.
- MoveLeft: strafe left 0.25m.
- MoveRight: strafe right 0.25m.
- RotateLeft: rotate 90 degrees left.
- RotateRight: rotate 90 degrees right.
- LookUp: tilt camera up.
- LookDown: tilt camera down.
- LookAround: full-room scan — 4 directional views, returns to original facing. Use when target is lost.

OBJECT INTERACTION — use objectType (plain type name, no coordinates):
- PickupObject(objectType), OpenObject(objectType), CloseObject(objectType), ToggleObjectOn(objectType), ToggleObjectOff(objectType), SliceObject(objectType), BreakObject(objectType), FillObjectWithLiquid(objectType), EmptyLiquidFromObject(objectType)
- PutObject(objectType, receptacleType): place held object INTO a receptacle. objectType=held type, receptacleType=target receptacle type (look for "(receptacle)" in the list). BOTH required.
- DropHandObject: drop held object.

TASK CONTROL:
- Done: only when task is FULLY achieved.
- MoveSequence(steps): chain multiple movements. Steps: [{"action": "MoveAhead", "repeat": 5}, ...]. Stops on first failure. Use to close distance to a known target without re-scanning.

Important rules for the counterfactual:
- Only provide it if you genuinely believe a different EARLIER decision would have prevented the failure
- The counterfactual must reference a specific past step and the alternative action
- If you don't have a clear counterfactual insight, omit it (set to null)
- Do NOT fabricate counterfactuals just to fill the field

OUTPUT FORMAT — YOU MUST OUTPUT VALID JSON ONLY:
- { must be the FIRST character of your response, } must be the LAST character.
- NO text outside the braces. NO markdown fences.
- The closing } is REQUIRED. Incomplete JSON = failed action.
- ALL text fields MUST use first-person: say "I tried..." not "The agent tried...". You ARE the agent.

{
  "diagnosis": "<1-2 sentences, be direct and concise. Use first-person: 'I ...'>",
  "recovery_reasoning": "<1-3 sentences, be direct and concise. Use first-person: 'I ...'>",
  "counterfactual": {
    "target_step": <step_index_number_or_null>,
    "alternative_action": {"action": "<action>", "params": {}},
    "reasoning": "<1 sentence: why this alternative would have prevented the failure>"
  } or null,
  "proposed_recovery_action": {"action": "<action>", "params": {}}
}

For counterfactual.target_step: the step number (integer) where you should have done something differently. For counterfactual.alternative_action: the action you should have taken at that step instead. If you have no counterfactual insight, set counterfactual to null."""


def build_phase3_prompt(
    task_goal: str,
    action_history: list[dict],
    error_message: str,
    cascade_description: str | None,
    visible_objects: list[dict] = None,
    agent_pos: dict = None,
    agent_rot_y: float = 0.0,
    inventory_objects: list[dict] | None = None,
    task_criteria: str = "",
    memory_text: str = "",
) -> str:
    lines = [f"Task goal: {task_goal}\n"]

    # Ego memory (compact spatial summary) — rendered first so VLM reads spatial context early
    if memory_text:
        lines.append(memory_text)
        lines.append("")

    _append_task_context(lines, visible_objects, inventory_objects, task_criteria)
    lines.append(f"ERROR: {error_message}\n")

    if cascade_description:
        lines.append(f"Context: {cascade_description}\n")

    if visible_objects:
        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        if visible:
            lines.append("Objects currently in view — use the TYPE name. Direction shown relative to your facing:")
            for o in visible:
                extra = []
                if o.get("isPickedUp"): extra.append("held")
                if o.get("receptacle"): extra.append("receptacle")
                if o.get("openable"): extra.append("openable" if not o.get("isOpen") else "open")
                tag = f" ({', '.join(extra)})" if extra else ""
                dir_label = ""
                if agent_pos and o.get("position"):
                    dir_label = " <- " + _direction(agent_pos, agent_rot_y, o["position"])
                lines.append(f"  {o['objectType']}{tag}{dir_label}")
        else:
            lines.append("(No objects currently in view — you are likely facing a wall or obstacle. Rotate or MoveBack to find open space. Do NOT LookAround from here.)")
        lines.append("")

    if not memory_text:
        lines.append(build_eb_history_context(action_history))

    lines.append("\nDiagnose the failure and propose a recovery action. Be concise. Use first-person ('I').")
    lines.append("OUTPUT: valid JSON only. { first char, } last char. No markdown fences, no text outside JSON.")
    return "\n".join(lines)


# ------------------------------------------------------------------
# EB Agent 顶层
# ------------------------------------------------------------------

PLANNER_SYSTEM = """You are an embodied agent in a 3D household. The image is your FIRST-PERSON VIEW. Your job is HIGH-LEVEL PLANNING: you decide WHAT to do, not exactly HOW. Another part of you (Executor) will handle the detailed action steps.

Output a single high-level intent. Be specific about the target object. Use EXACT objectType names.

RULES:
- Look at the image, visible objects, task goal, hand status, and spatial memory.
- Decide the next logical sub-goal to make progress toward the task.
- If target is >0.5m away: intent is to APPROACH it first.
- If target is in hand and task requires putting it somewhere: intent is to PLACE it.
- If target is not visible: intent is to LOCATE it.

INTENTS (use these exact forms):
  approach <objectType>      — move toward the object until within 0.5m
  pickup <objectType>        — pick up the object (only if within 0.5m!)
  put <objectType> on <receptacleType> — place held object into receptacle
  open <objectType>          — open a container/cabinet/drawer
  close <objectType>         — close a container
  toggle on <objectType>     — turn on an appliance
  toggle off <objectType>    — turn off an appliance
  locate <objectType>        — search the room for a target not currently visible
  scan room                  — full 4-direction scan to understand surroundings
  wait                       — nothing to do, task in progress
  Done                       — task is complete

OUTPUT — valid JSON only. { first char, } last char. No markdown.

{
  "intent": "<intent phrase>",
  "target": "<objectType or empty>",
  "reasoning": "<1-3 sentences: why this intent now, first-person>"
}"""

PLANNER_REVIEW_SYSTEM = """You are an embodied agent reviewing your Executor's proposed action sequence. Default to APPROVE unless there is a CRITICAL error.

ONLY reject if:
- PickupObject/PutObject proposed when target is clearly >0.5m away
- MoveAhead proposed directly into a known blocked direction (from history)
- Actions would clearly move AWAY from the target

Do NOT reject for:
- Minor inefficiency (extra steps are fine)
- "Could be more direct" — the Executor sees the current view, trust it
- Slightly different approach than what you would do

OUTPUT — valid JSON only:

{
  "approved": true/false,
  "reason": "<1 sentence. if approved: 'ok'. if rejected: the critical error>",
  "corrected_actions": [{"action": "MoveAhead", "params": {}}, ...]
}"""


class EBAgent:
    def __init__(self, client: VLMClient):
        self.client = client

    # ------------------------------------------------------------------
    # Planner: high-level intent
    # ------------------------------------------------------------------
    def plan_intent(
        self,
        task_goal: str,
        image: np.ndarray,
        visible_objects: list[dict],
        action_history: list[dict],
        last_error: str | None,
        agent_pos: dict = None,
        agent_rot_y: float = 0.0,
        inventory_objects: list[dict] | None = None,
        hand_status: str = "",
        task_criteria: str = "",
        memory_text: str = "",
    ) -> dict:
        """Propose the next high-level intent."""
        lines = [f"Task goal: {task_goal}\n"]
        if memory_text:
            lines.append(memory_text)
            lines.append("")
        if hand_status:
            lines.append(f"HAND STATUS: {hand_status}")
        if task_criteria:
            lines.append(f"\nTASK COMPLETION CRITERIA:\n{task_criteria}")

        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        if visible:
            lines.append("\nObjects in view — direction relative to your facing:")
            for o in visible[:12]:
                extra = []
                if o.get("isPickedUp"): extra.append("held")
                if o.get("receptacle"): extra.append("receptacle")
                if o.get("openable"): extra.append("open")
                if o.get("toggleable"): extra.append("on" if o.get("isToggled") else "off")
                tag = f" ({','.join(extra)})" if extra else ""
                d = ""
                if agent_pos and o.get("position"):
                    d = " <- " + _direction(agent_pos, agent_rot_y, o["position"])
                lines.append(f"  {o['objectType']}{tag}{d}")
        else:
            lines.append("\n(No objects in view — you may be facing a wall. Rotate or MoveBack.)")

        if last_error:
            lines.append(f"\nLast error: {last_error}")

        lines.append("\nRecent history:")
        lines.append(build_eb_history_context(action_history[-5:] if len(action_history) > 5 else action_history))

        lines.append("\nPropose the next intent. Be specific. Output JSON only.")
        prompt = "\n".join(lines)

        return self.client.chat_with_image_json(
            system_prompt=PLANNER_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("intent", "target", "reasoning"),
        )

    # ------------------------------------------------------------------
    # Planner: review Executor's action sequence
    # ------------------------------------------------------------------
    def review_actions(
        self,
        intent: str,
        target: str,
        proposed_actions: list[dict],
        executor_reasoning: str,
        image: np.ndarray,
        visible_objects: list[dict],
        action_history: list[dict],
    ) -> dict:
        """Review Executor's action sequence. If rejected, provide corrected actions."""
        lines = [f"Your intent was: {intent}"]
        if target:
            lines.append(f"Target: {target}")
        lines.append("")
        lines.append("Executor proposed these actions:")
        for i, a in enumerate(proposed_actions, 1):
            act = a.get("action", "?")
            params = a.get("params", {})
            if params:
                lines.append(f"  {i}. {act}({params})")
            else:
                lines.append(f"  {i}. {act}")
        lines.append(f"\nExecutor reasoning: {executor_reasoning}")

        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        if visible:
            lines.append("\nObjects in view:")
            for o in visible[:8]:
                d = ""
                if o.get("receptacle"):
                    d = " (receptacle)"
                if o.get("isPickedUp"):
                    d += " (held)"
                lines.append(f"  {o['objectType']}{d}")

        lines.append("\nReview. If rejected, provide corrected_actions. Output JSON only.")
        prompt = "\n".join(lines)

        result = self.client.chat_with_image_json(
            system_prompt=PLANNER_REVIEW_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("approved", "reason", "corrected_actions"),
        )
        return result

    def propose_action(
        self,
        task_goal: str,
        image: np.ndarray,
        visible_objects: list[dict],
        action_history: list[dict],
        last_error: str | None,
        failed_object_ids: set = None,
        agent_pos: dict = None,
        agent_rot_y: float = 0.0,
        inventory_objects: list[dict] | None = None,
        task_criteria: str = "",
        memory_text: str = "",
    ) -> dict:
        """Phase 1: propose next action with spatial direction hints."""
        prompt = build_phase1_prompt(task_goal, visible_objects, action_history, last_error, failed_object_ids, agent_pos, agent_rot_y, inventory_objects, task_criteria, memory_text=memory_text)
        result = self.client.chat_with_image_json(
            system_prompt=PHASE1_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("action", "params", "reasoning"),
        )
        return result

    def propose_action_lookaround(
        self,
        task_goal: str,
        look_images: list,
        visible_objects: list[dict],
        action_history: list[dict],
        last_error: str | None,
        failed_object_ids: set = None,
        inventory_objects: list[dict] | None = None,
        task_criteria: str = "",
        memory_text: str = "",
    ) -> dict:
        """LookAround: 4 方向图片综合分析。"""
        NL = chr(10)
        lines = []
        lines.append(f"Task: {task_goal}")
        lines.append("")
        _append_task_context(lines, visible_objects, inventory_objects, task_criteria)
        lines.append("You just did a full-room scan. Below are 4 views: ahead, left, behind, right.")
        lines.append("Use the views to locate your target and decide your next move.")
        lines.append("")
        if memory_text:
            lines.append(memory_text)
        else:
            lines.append(build_eb_history_context(action_history))
        if last_error:
            lines.append(f"\nLast action error: {last_error}")
        if failed_object_ids:
            lines.append("\nObjectIds that failed before:")
            for fid in failed_object_ids:
                lines.append(f"  - {fid}")
        lines.append("")
        lines.append("Objects in scene:")
        for o in visible_objects:
            extra = []
            if o.get("isPickedUp"): extra.append("HELD")
            if o.get("receptacle"): extra.append("receptacle")
            if o.get("visibleBounds2D"): extra.append("VISIBLE")
            else: extra.append("hidden")
            tag = " [" + ", ".join(extra) + "]" if extra else ""
            lines.append(f"  {o['objectType']}{tag}")
        lines.append("")
        lines.append("CRITICAL: Use the EXACT objectType from the list above. The task description may say \"red cloth\" but the internal name is \"Cloth\". Copy it EXACTLY from this list. Do not invent your own name.")
        prompt = NL.join(lines)

        result = self.client.chat_with_images_json(
            system_prompt=PHASE1_SYSTEM,
            user_text=prompt,
            images=look_images,
            required_fields=("action", "params", "reasoning"),
        )
        return result

    def diagnose_failure(
        self,
        task_goal: str,
        error_message: str,
        image: np.ndarray,
        action_history: list[dict],
        cascade_level: int,
        visible_objects: list[dict] = None,
        agent_pos: dict = None,
        agent_rot_y: float = 0.0,
        inventory_objects: list[dict] | None = None,
        task_criteria: str = "",
        memory_text: str = "",
    ) -> dict:
        """Phase 3: 诊断失败并提议恢复。"""
        if cascade_level > 1:
            desc = f"This is your {cascade_level}th consecutive failure while recovering from a previous problem. Previous recovery attempts have not succeeded."
        elif cascade_level == 1:
            desc = "This failure occurred during normal task execution."
        else:
            desc = None

        prompt = build_phase3_prompt(task_goal, action_history, error_message, desc,
                                     visible_objects, agent_pos, agent_rot_y, inventory_objects, task_criteria,
                                     memory_text=memory_text)
        result = self.client.chat_with_image_json(
            system_prompt=PHASE3_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("diagnosis", "recovery_reasoning", "counterfactual", "proposed_recovery_action"),
        )
        return result