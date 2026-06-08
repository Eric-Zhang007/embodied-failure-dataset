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

PHASE1_SYSTEM = """You are an embodied agent inside a 3D household environment. The image you receive is your FIRST-PERSON VIEW — exactly what your eyes see right now. You inhabit a physical body in this space, and every action you take moves or rotates your body.

SPATIAL AWARENESS — always think about:
- Your current position and orientation in the room. You have a body that occupies space.
- The distance between you and each object. Objects far away cannot be reached — you must MoveAhead to get closer first.
- What is visible NOW in your field of view vs. what might be behind you or to your sides.
- The 3D layout: furniture placement, wall locations, object positions on surfaces.
- Height: objects on the floor, on tables, on shelves, or on the ceiling each require different camera angles (LookDown, level, LookUp).

VISUAL NAVIGATION — the image is your primary navigation tool:
- LOOK AT THE IMAGE to judge where objects are. If a target is on the LEFT side of the image, RotateLeft toward it. If on the RIGHT side, RotateRight. If CENTERED and close, MoveAhead.
- SIZE TELLS DISTANCE: a small object in the image is far away (need multiple MoveAhead steps). A large object filling your view is close (1 step or within reach).
- USE THE DIRECTION HINTS in the object list: each object shows its direction and distance relative to you (e.g., "ahead (2.0m)", "behind-left (3.5m)"). Rotate to face the object before moving.
- If you cannot see your target in the image, check the direction hints to know which way to turn.

NAVIGATION AND EXPLORATION:
- When MoveAhead succeeds, check your new view carefully. Has the target changed size? Are new objects visible? Is a receptacle now within reach? Stop moving when you are close enough to interact — do not walk past your target.
- When MoveAhead fails with "blocking": you hit furniture or a wall. Do NOT retry the same direction. Turn 90 degrees (RotateLeft or RotateRight) and try moving there. If blocked again, turn another 90 degrees. If blocked in ALL directions, use MoveBack to retreat, then rotate to find an open path.
- When you cannot find your target: it might be behind you. RotateLeft twice to turn around and check. Or rotate 90 degrees at a time while scanning. The room has 4 sides — explore all of them.
- You have 4 movement directions: forward, back, left, right. Use ALL of them. Sidestepping (MoveLeft/Right) helps you go around furniture without losing sight of your target.
- If you keep hitting the same obstacle from different angles, you are circling it. MoveBack to step away, then take a wider path around.

INTERACTION RANGE — you can only interact with objects within 0.5m:
- The visible objects list shows each object's direction AND distance from you (e.g., "ahead (1.2m)").
- Check the distance BEFORE proposing PickupObject, OpenObject, PutObject, or any interaction action.
- If distance > 0.5m: you are TOO FAR. You MUST MoveAhead (each step = 0.25m) to get closer first. At 1.0m, you need 2+ MoveAhead steps.
- If PickupObject fails with "not found" when the object IS visible in the list: you are almost certainly too far away. MoveAhead and retry.
- Never propose an interaction action on an object farther than 0.5m — it will fail.

CRITICAL PICKUP RULE: A visible object may be out of reach (too far, blocked, or inside a closed container). Check the distance first — if > 0.5m, MoveAhead. If it fails at close range, look around for another instance or check if a container needs opening first.

AVAILABLE ACTIONS (use EXACTLY these names, do NOT invent new ones):

NAVIGATION:
- MoveAhead: move forward 0.25m. If BLOCKED, do NOT retry — Rotate to find a clear path.
- MoveBack: move backward 0.25m.
- MoveLeft: strafe left 0.25m.
- MoveRight: strafe right 0.25m.
- RotateLeft: rotate 90 degrees left.
- RotateRight: rotate 90 degrees right.
- LookUp: tilt camera up.
- LookDown: tilt camera down.
- LookAround: full-room scan — 4 directional views, returns you to original facing. Use when target is lost.

OBJECT INTERACTION — use objectType (the plain type name from the list, no coordinates needed):
- PickupObject(objectType): pick up an object. Use the TYPE name exactly as shown in the list (e.g., "AlarmClock"). You do NOT need to copy any coordinates.
- PutObject(objectType, receptacleType): place held object INTO a receptacle. objectType=the TYPE you are HOLDING. receptacleType MUST match the TASK GOAL target (read the task description to know WHERE to put the object). Floor is NOT a desk.
- OpenObject(objectType), CloseObject(objectType), ToggleObjectOn(objectType), ToggleObjectOff(objectType)
- SliceObject(objectType), BreakObject(objectType)
- FillObjectWithLiquid(objectType), EmptyLiquidFromObject(objectType)
- DropHandObject: drop the object you are holding.

CRITICAL — OBJECT NAMES: Use ONLY the objectType printed in the visible objects list. The task description may use descriptive words (e.g., "red cloth", "alarm clock") — these are NOT the internal names. The internal name is what you see in the list (e.g., "Cloth", "AlarmClock"). COPY IT EXACTLY from the list. Do NOT invent your own name like "RedCloth". The names are case-sensitive. The system resolves the type to the correct object automatically.

TASK CONTROL:
- Done: only when the task is FULLY achieved by checking environment state.

RESPONSE RULES:
- reasoning MUST use FOUR-part STRUCTURED format: "scene: ... | goal: ... | plan: ... | reflection: ..."
- scene: describe what you SEE in the image right now — objects, locations, distances.
- goal: state the current sub-goal and your progress toward the task.
- plan: explain why THIS specific action was chosen to advance toward the goal.
- reflection: check your assumptions. Where did you think the target was? Were you right? Did you assume something that turned out to be wrong? What did the last error teach you (if any)? If you're repeating the same action that just failed, STOP and reconsider.
- Each part 1-3 sentences.

EXAMPLE — PickupObject:
{"action": "PickupObject", "params": {"objectType": "AlarmClock"}, "reasoning": "scene: I see an alarm clock on the desk ahead, about 1.0m away, clearly visible and within reach. | goal: I need to pick up the alarm clock. I have located it and am close enough to interact. | plan: The clock is directly ahead within reach; picking it up now advances the task. | reflection: The clock is exactly where the visible objects list says it is. I have confirmed its position visually. No errors so far — this action should succeed."}

EXAMPLE — PutObject:
{"action": "PutObject", "params": {"objectType": "AlarmClock", "receptacleType": "Desk"}, "reasoning": "scene: I am facing a wooden desk directly ahead at 0.5m. | goal: I need to place the alarm clock onto the desk. I am holding the clock and the desk is the correct receptacle. | plan: The desk is within reach and matches the task target; placing the clock now completes the task. | reflection: The desk is the correct target per the task goal. I am confident the clock will be placed correctly."}

RESPONSE FORMAT — YOU MUST OUTPUT VALID JSON ONLY:
- Your ENTIRE response must be a single JSON object: { must be the FIRST character, } must be the LAST character.
- NO text outside the braces. NO markdown fences (```). NO prefixes like "Here is my response:".
- The closing } is REQUIRED. If your JSON is truncated or missing }, your action will FAIL silently.
- Every field (action, params, reasoning) is REQUIRED. Missing fields will cause a FAILURE.

Respond with:
{
  "action": "<exact action name from the list above>",
  "params": {},
  "reasoning": "scene: <what you see> | goal: <task progress + sub-goal> | plan: <why this action> | reflection: <check assumptions, learn from errors>"
}"""



def _direction(agent_pos, agent_rot_y, obj_pos):
    dx = obj_pos["x"] - agent_pos["x"]
    dz = obj_pos["z"] - agent_pos["z"]
    rad = math.radians(agent_rot_y)
    fx, fz = math.sin(rad), math.cos(rad)
    forward = dx * fx + dz * fz
    right = dx * fz - dz * fx
    angle = math.degrees(math.atan2(right, forward))
    d = math.sqrt(dx*dx + dz*dz)
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

    visible = [o for o in visible_objects if o.get("visible")]
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
        lines.append("(No objects currently in view)")

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

    lines.append("\nCRITICAL — OBJECT NAMES: The task description may use descriptive words (e.g., \"red cloth\").")
    lines.append("These DO NOT match the internal objectType from the visible list above.")
    lines.append("You MUST copy the objectType EXACTLY from the visible objects list — character by character.")
    lines.append("Example: if the list shows \"Cloth\", you MUST write \"Cloth\", NOT \"RedCloth\" or \"red cloth\".")
    lines.append("If the list shows \"AlarmClock\", write \"AlarmClock\", not \"clock\" or \"alarm\".")
    lines.append("The names are case-sensitive and must match precisely. Copy-paste accuracy is mandatory.")
    if failed_object_ids:
        lines.append("IMPORTANT: Objects marked as previously failed may work if you get closer to them first.")
    lines.append("DISTANCE CHECK: Check the distance label next to each object above. If your target is > 0.5m away, you MUST MoveAhead first (0.25m per step). Interaction only works within 0.5m.")
    lines.append("OUTPUT: valid JSON only. { must be first char, } must be last char. No markdown, no text outside JSON.")
    return "\n".join(lines)


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
        visible = [o for o in visible_objects if o.get("visible")]
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
            lines.append("(No objects currently in view)")
        lines.append("")

    if not memory_text:
        lines.append(build_eb_history_context(action_history))

    lines.append("\nDiagnose the failure and propose a recovery action. Be concise. Use first-person ('I').")
    lines.append("OUTPUT: valid JSON only. { first char, } last char. No markdown fences, no text outside JSON.")
    return "\n".join(lines)


# ------------------------------------------------------------------
# EB Agent 顶层
# ------------------------------------------------------------------

class EBAgent:
    def __init__(self, client: VLMClient):
        self.client = client

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
            if o.get("visible"): extra.append("VISIBLE")
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
