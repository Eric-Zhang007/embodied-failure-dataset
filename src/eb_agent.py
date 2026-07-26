"""
EB Agent（具身 Agent）。

负责：
- Phase 1: 根据任务目标和当前状态提出下一步动作
- Phase 3: 遇到失败后进行诊断、恢复推理、可选的的反事实推理
"""

import json
import math
import re
import numpy as np

from src.vlm_client import VLMClient
from src.context_builder import build_eb_history_context, render_intent_tree
from src.curiosity_scorer import (
    build_curiosity_table_text,
    check_proposed_target,
    CuriosityStats,
)



# ------------------------------------------------------------------
# Phase 1: 动作提议
# ------------------------------------------------------------------

PHASE1_SYSTEM = """I am an embodied agent in a 3D household. The image is my FIRST-PERSON VIEW. I occupy a physical body; every action moves or rotates me.

RULES:
- Interaction range is 0.5m. Check distance BEFORE PickupObject/PutObject/etc. If >0.5m, MoveAhead (0.125m/step) first. Never interact beyond 0.5m.
- When MoveAhead BLOCKED: do NOT retry same direction and do NOT LookAround. Instead, Rotate 90deg and try there. If blocked in all 4 directions, MoveBack to escape the tight spot. LookAround is useless when you are boxed in — you already know you're stuck, you need to MOVE.
- When target not visible AND you have open space around you: rotate to scan. Use direction hints in the object list.
- If the object list below is empty: you are facing a wall or obstacle. DO NOT LookAround — Rotate or MoveBack to find open space, then find your target.
- Use all 4 movement directions. Sidestep (MoveLeft/Right) to go around obstacles.
- If hitting same obstacle repeatedly: MoveBack then wider path.

AVAILABLE ACTIONS — use EXACT names and syntax:

Navigation:
  MoveAhead          — forward 0.125m
  MoveBack           — backward 0.125m
  MoveLeft           — strafe left 0.125m
  MoveRight          — strafe right 0.125m
  RotateLeft         — turn 90deg left
  RotateRight        — turn 90deg right
  LookUp             — tilt camera up (+30° max from horizon)
  LookDown           — tilt camera down (-60° max from horizon)
  LookAround         — 4-direction scan. Use when target is lost.

Object interaction — use objectType from the visible objects list. Must be within 0.5m reach:
  PickupObject(objectType)
  PutObject(objectType, receptacleType)   — objectType=what you hold, receptacleType=TASK TARGET (not Floor!)
  OpenObject(objectType) / CloseObject(objectType)
  ToggleObjectOn(objectType) / ToggleObjectOff(objectType)
  SliceObject(objectType) / BreakObject(objectType)
  FillObjectWithLiquid(objectType, fillLiquid="water") / EmptyLiquidFromObject(objectType)
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


def _surface_position(agent_pos, obj):
    """Return (x, z) of nearest point on object's AABB to agent.

    Uses axisAlignedBoundingBox cornerPoints to compute distance to the
    object surface rather than its center. Falls back to obj['position']
    if no AABB data is available.
    """
    pos = obj.get("position")
    bbox = obj.get("axisAlignedBoundingBox")
    if bbox and bbox.get("cornerPoints"):
        corners = bbox["cornerPoints"]
        # cornerPoints format: [[x,y,z], [x,y,z], ...]
        xs = [p[0] for p in corners]
        zs = [p[2] for p in corners]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        # Clamp agent position to AABB extent
        cx = max(min_x, min(max_x, agent_pos["x"]))
        cz = max(min_z, min(max_z, agent_pos["z"]))
        return {"x": cx, "z": cz}
    # Fallback: use object center
    if pos:
        return {"x": pos["x"], "z": pos["z"]}
    return {"x": 0.0, "z": 0.0}


def _direction_for_obj(agent_pos, agent_rot_y, obj):
    """Like _direction but computes distance to object SURFACE (AABB) not center."""
    surf = _surface_position(agent_pos, obj)
    return _direction(agent_pos, agent_rot_y, surf)


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
                dir_label = " <- " + _direction_for_obj(agent_pos, agent_rot_y, o)
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
                dir_label = " <- " + _direction_for_obj(agent_pos, agent_rot_y, o)
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
# Phase 3: 失败诊断与恢复推理
# ------------------------------------------------------------------

PHASE3_SYSTEM = """You are an embodied agent that has just encountered a failure while performing a household task. Remember: you are in a 3D first-person environment — the image is what your eyes see, and you must reason about spatial relationships.

Your job is to:
0. FIRST: look at the image and your CURRENT INTENT (shown below). What were you trying to do? Why did it fail? The error message tells you what the environment rejected — believe it.
1. Diagnose WHY the failure happened — connect your intent, the error, and what you see. If you were trying to approach a target but hit an obstacle, say so. If you couldn't find the object, say so.
2. Propose a recovery action to get back on track — this should address the SPECIFIC failure cause.
3. Optionally, provide a counterfactual: "If I had done X instead of Y earlier, this failure would not have occurred."

If the user prompt contains an ENVIRONMENT-CONFIRMED RUNTIME TRAP with status "triggered", its recovery_action has been verified against the current scene. Use that exact recovery action before retrying the failed task action.

AVAILABLE ACTIONS (use EXACTLY these names, do NOT invent new ones):

NAVIGATION:
- MoveAhead: move forward 0.125m. If BLOCKED, Rotate to find a clear path.
- MoveBack: move backward 0.125m.
- MoveLeft: strafe left 0.125m.
- MoveRight: strafe right 0.125m.
- RotateLeft: rotate 90 degrees left.
- RotateRight: rotate 90 degrees right.
- LookUp: tilt camera up (+30° max from horizon).
- LookDown: tilt camera down (-60° max from horizon).

OBJECT INTERACTION — use objectType (plain type name, no coordinates):
- PickupObject(objectType), OpenObject(objectType), CloseObject(objectType), ToggleObjectOn(objectType), ToggleObjectOff(objectType), SliceObject(objectType), BreakObject(objectType), FillObjectWithLiquid(objectType, fillLiquid="water"), EmptyLiquidFromObject(objectType)
- PutObject(objectType, receptacleType): place held object INTO a receptacle. objectType=held type, receptacleType=target receptacle type (look for "(receptacle)" in the list). BOTH required.
- DropHandObject: drop held object.

TASK CONTROL:
- MoveSequence(steps): chain multiple movements. Steps: [{"action": "MoveAhead", "repeat": 5}, ...]. Stops on first failure. Use to close distance to a known target without re-scanning.

CRITICAL — COLLISION ENTITIES ARE NOT INTERACTABLE OBJECTS:
The error message may mention internal collision geometry names (Cube.001, OVENDOOR.001, Cube.527, etc.) — these are invisible physics boundaries, NOT objects you can interact with. Do NOT propose OpenObject/CloseObject for these names. They are not in the visible-objects list and cannot be opened or closed. Instead, use NAVIGATION (MoveBack, Rotate, MoveLeft/Right) to go around the obstacle. If the error says "Cube.001 is blocking", the recovery is to MoveBack and find a different path — NOT to "close the oven" or "open the cube."

Important rules for the counterfactual:
- counterfactual is REQUIRED — you MUST always provide one. Every failure has a root cause that traces back to an earlier decision.
- You must reference a specific past step index and specify what alternative action at that step would have prevented this failure.
- Even if you are uncertain, give your best analysis. An imperfect counterfactual is much more valuable than none.
- Do NOT set counterfactual to null under any circumstances.

OUTPUT FORMAT — YOU MUST OUTPUT VALID JSON ONLY:
- { must be the FIRST character of your response, } must be the LAST character.
- NO text outside the braces. NO markdown fences.
- The closing } is REQUIRED. Incomplete JSON = failed action.
- ALL text fields MUST use first-person: say "I tried..." not "The agent tried...". You ARE the agent.

{
  "diagnosis": "<1-2 sentences, be direct and concise. Use first-person: 'I ...'>",
  "recovery_reasoning": "<1-3 sentences, be direct and concise. Use first-person: 'I ...'>",
  "counterfactual": {
    "target_step": <step_index_number>,
    "alternative_action": {"action": "<action>", "params": {}},
    "reasoning": "<1 sentence: why this alternative would have prevented the failure>"
  },
  "proposed_recovery_action": {"action": "<action>", "params": {}}
}

proposed_recovery_action must be a physical movement or object interaction (MoveAhead, RotateLeft, PickupObject, etc.). Do NOT propose Done or LookAround as recovery actions.
For counterfactual.target_step: the step number (integer) where you should have done something differently. For counterfactual.alternative_action: the exact action you should have taken at that step instead — must be a valid executable action (not Done or LookAround)."""


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
    current_intent: str = "",
    trap_state: list[dict] | None = None,
) -> str:
    lines = [f"Task goal: {task_goal}\n"]

    # Ego memory (compact spatial summary) — rendered first so VLM reads spatial context early
    if memory_text:
        lines.append(memory_text)
        lines.append("")

    _append_task_context(lines, visible_objects, inventory_objects, task_criteria)
    lines.append(f"ERROR: {error_message}\n")

    if current_intent:
        lines.append(f"Your current intent was: {current_intent}")
        lines.append("(You failed while trying to carry out this intent. Diagnose why.)\n")

    if cascade_description:
        lines.append(f"Context: {cascade_description}\n")

    triggered_traps = [
        {
            "expected_failure": trap.get("expected_failure"),
            "recovery_action": trap.get("recovery_action"),
        }
        for trap in (trap_state or [])
        if trap.get("status") == "triggered"
    ]
    if triggered_traps:
        lines.append(
            "ENVIRONMENT-CONFIRMED RUNTIME TRAP — recover with its exact "
            "recovery_action before retrying:\n"
            + json.dumps(triggered_traps, ensure_ascii=False, sort_keys=True)
            + "\n"
        )

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
                    dir_label = " <- " + _direction_for_obj(agent_pos, agent_rot_y, o)
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

PLANNER_SYSTEM = """I am an embodied agent in a 3D household. The image is my FIRST-PERSON VIEW. My job is HIGH-LEVEL PLANNING: I decide WHAT to do, not exactly HOW. Another part of me (Executor) will handle the detailed action steps.

Output a single high-level intent. Be specific about the target object. Use EXACT objectType names.

RULES:
- Look at the image, visible objects, task goal, hand status, and spatial memory.
- MEMORY OVER VISUAL GUESSING: If the spatial memory contains a "WHERE MY TARGET IS" section, BELIEVE IT. The memory tracks each object by its unique identity — it knows the REAL location of your target even when a visually similar object (same color/shape) is in view. If memory says your target is inside Fridge, you must OPEN Fridge. Do NOT get distracted by a lookalike on CounterTop.
- CONTAINER-FIRST: If memory says the target was last seen INSIDE a specific container (Fridge, Cabinet, Microwave, etc.), and that container is visible or remembered, your intent MUST be to open that container. Searching other surfaces while memory pins the target to a container wastes steps and produces meaningless failures.
- DISAMBIGUATION: Objects marked with ⚠ in the visible list are NOT your target. Memory is authoritative — the ⚠ warning means the real target was seen elsewhere, and this is a different object that merely looks similar. Ignore lookalikes completely.
- RETRIEVAL FROM A CONTAINER: If the target is a heat/cool/clean-processed object (or anything I earlier placed inside a container such as Microwave/Fridge/Cabinet), it will NOT appear in the visible objects list while that container is closed — an object inside a closed container has visibleBounds2D=false. Its ABSENCE from the visible list is EXPECTED and is NOT evidence the object is gone or that I should search elsewhere. Memory still knows it is in that container. To retrieve it, decompose in this exact order: (1) "approach <container>" until the container is within 0.5m and in view — I may have rotated or stepped away after placing/heating it, so I must RE-APPROACH, not interact from where I now stand; (2) "open <container>"; (3) "pickup <target>". Do NOT abandon the container and wander to other receptacles just because the target is not currently visible.
- Decide the next logical sub-goal to make progress toward the task.
- If target is >0.5m away: intent is to APPROACH it first.
- If target is in hand and task requires putting it somewhere: intent is to PLACE it.
- If target is not visible: first check SPATIAL MEMORY. If the memory says where the target was last seen (e.g. "saw Apple on Counter"), output "approach <that location>".
- If the target has NEVER been seen: use "scan room" once to get a full-room overview. After scanning, DO NOT use "locate X" — the Executor has no spatial reasoning and will fail. Instead, pick a specific receptacle or area from the scan and use "approach <area>" to search there.

IF TARGET NEVER SEEN — EXPLORATION STRATEGY:
Instead of "locate X" (which gives the Executor no direction), create a concrete search plan:
1. Look at the AREA label and visible receptacles (Counter, Table, Desk, Shelf, etc.)
2. Pick ONE specific receptacle or visible object to approach
3. Output "approach <receptacle>" — the Executor can execute this. If the target isn't there, you'll try the next location on the following turn.
Example: if looking for an Apple in a kitchen, say "approach CounterTop" (not "locate Apple"). If it's not there, next turn say "approach DiningTable". This gives the Executor concrete targets it can reach.

INTENTS (use these exact forms):
  approach <objectType>      — move toward the object until within 0.5m
  pickup <objectType>        — pick up the object (only if within 0.5m!)
  put <objectType> on <receptacleType> — place held object into receptacle
  open <objectType>          — open a container/cabinet/drawer
  close <objectType>         — close a container
  toggle on <objectType>     — turn on an appliance
  toggle off <objectType>    — turn off an appliance
  scan room                  — full 4-direction scan (use ONCE, then pick specific targets from the scan)
  unmark <objectType>        — undo a SEARCHED mark (use when you suspect the target is still inside)
  wait                       — nothing to do, task in progress
  Done                       — task is complete (see below)

THREE SPECIAL INTENTS — how the system handles them:

1. "scan room" — when your current view is insufficient and you need a full-room overview.
   What happens: the system captures 4 directional views (ahead/left/behind/right) and shows
   them to you in the next call. Your job THEN is to look at all 4 views, decide which direction
   to face, and output a normal intent (e.g. "approach Desk"). The Executor then carries it out.
   Use "scan room" sparingly — only when the target has NEVER been seen and there is no
   spatial memory of it. If you scanned recently, you already have the layout; use it.

2. "Done" — when you believe ALL task completion criteria are met.
   What happens: the system runs a hard verification against the task criteria. If ALL pass,
   the episode ends successfully. If ANY criterion is missing, you will get a specific rejection
   message telling you exactly what is not yet satisfied — fix those conditions first.

3. "unmark <objectType>" — when you suspect a previously-searched receptacle might still
   contain the target (e.g. the target was occluded behind another object when you checked).
   What happens: the system clears the SEARCHED flag so the Executor can re-open and
   re-check that receptacle. Use this when your diagnosis of a recent failure suggests the
   target was "missed", "occluded", or "hidden" inside an already-searched container.

OUTPUT — valid JSON only. { first char, } last char. No markdown.

{
  "intent": "<intent phrase>",
  "target": "<objectType or empty>",
  "reasoning": "<1-3 sentences: why this intent now, first-person>"
}"""

PLANNER_REVIEW_SYSTEM = """I am an embodied agent reviewing my Executor's proposed action sequence. I should APPROVE unless there is a CRITICAL error.

ONLY reject if:
- PickupObject/PutObject proposed when target is clearly >0.5m away
- MoveAhead proposed directly into a known blocked direction (check recent history!)
- Actions would clearly move AWAY from the target

STUCK DETECTION: if the last 2+ attempts in recent history all failed at their FIRST action, I am trapped at a navigable-area edge. In this case, MoveBack IS the correct action — I should approve it, or if the Executor didn't propose it, add it to corrected_actions.

I should NOT reject for:
- Minor inefficiency (extra steps are fine)
- "Could be more direct" — the Executor sees the current view, I should trust it
- Slightly different approach than what I would do

OUTPUT — valid JSON only:

{
  "approved": true/false,
  "reason": "<1 sentence. if approved: 'ok'. if rejected: the critical error>",
  "corrected_actions": [{"action": "MoveAhead", "params": {}}, ...]
}"""


# ------------------------------------------------------------------
# Contrastive Planner (Spike 004): explore-bias system prompt
# ------------------------------------------------------------------

EXPLORE_BIAS_SYSTEM = """I am an embodied agent in EXPLORATION MODE. My job is HIGH-LEVEL PLANNING with an exploration bias.

CRITICAL — EXPLORATION DIRECTIVE:
- You MUST choose a location / receptacle you have NOT yet visited or searched.
- Avoid repeating locations that appear in your recent intent history.
- Look at the SPATIAL MEMORY for remembered-but-unvisited receptacles (not marked SEARCHED).
- If the target has NEVER been seen, pick a receptacle you have NOT approached yet.
- Prefer novel locations over familiar ones. Diversity is the goal.
- MEMORY OVER VISUAL GUESSING: If spatial memory says the target is inside a specific container, prioritize opening that container over novel exploration. The exploration bias does NOT override memory — memory of the target's actual location is always the highest priority.

Output a single high-level intent. Be specific about the target object. Use EXACT objectType names.

RULES:
- Look at the image, visible objects, task goal, hand status, and spatial memory.
- MEMORY OVER VISUAL GUESSING: If the spatial memory contains a "WHERE MY TARGET IS" section, BELIEVE IT. The memory tracks each object by its unique identity — it knows the REAL location of your target even when a visually similar object (same color/shape) is in view. If memory says your target is inside Fridge, you must OPEN Fridge. Do NOT get distracted by a lookalike on CounterTop.
- CONTAINER-FIRST: If memory says the target was last seen INSIDE a specific container (Fridge, Cabinet, Microwave, etc.), and that container is visible or remembered, your intent MUST be to open that container. Even when exploring, the known target location takes priority over unexplored areas.
- DISAMBIGUATION: Objects marked with ⚠ in the visible list are NOT your target. Memory is authoritative — the ⚠ warning means the real target was seen elsewhere, and this is a different object that merely looks similar. Ignore lookalikes completely.
- RETRIEVAL FROM A CONTAINER OVERRIDES EXPLORATION: If the target was earlier placed inside a container (Microwave/Fridge/Cabinet — e.g. after heating/cooling/cleaning), it will NOT appear in the visible list while that container is closed (visibleBounds2D=false). Its absence is EXPECTED and is NOT a reason to explore a new receptacle. Re-approach that SAME container to within 0.5m (I may have stepped/rotated away after placing it), then open it, then pickup the target. Never wander to a novel receptacle when memory pins the target inside a known container.
- Decide the next logical sub-goal to make progress toward the task.
- If target is >0.5m away: intent is to APPROACH it first.
- If target is in hand and task requires putting it somewhere: intent is to PLACE it.
- If target is not visible: first check SPATIAL MEMORY. If the memory says where the target was last seen (e.g. "saw Apple on Counter"), output "approach <that location>".
- If the target has NEVER been seen: use "scan room" once to get a full-room overview. After scanning, DO NOT use "locate X" — the Executor has no spatial reasoning and will fail. Instead, pick a specific receptacle or area from the scan and use "approach <area>" to search there.

IF TARGET NEVER SEEN — EXPLORATION STRATEGY:
Instead of "locate X" (which gives the Executor no direction), create a concrete search plan:
1. Look at the AREA label and visible receptacles (Counter, Table, Desk, Shelf, etc.)
2. Pick ONE specific receptacle or visible object to approach
3. Output "approach <receptacle>" — the Executor can execute this. If the target isn't there, you'll try the next location on the following turn.
Example: if looking for an Apple in a kitchen, say "approach CounterTop" (not "locate Apple"). If it's not there, next turn say "approach DiningTable". This gives the Executor concrete targets it can reach.

INTENTS (use these exact forms):
  approach <objectType>      — move toward the object until within 0.5m
  pickup <objectType>        — pick up the object (only if within 0.5m!)
  put <objectType> on <receptacleType> — place held object into receptacle
  open <objectType>          — open a container/cabinet/drawer
  close <objectType>         — close a container
  toggle on <objectType>     — turn on an appliance
  toggle off <objectType>    — turn off an appliance
  scan room                  — full 4-direction scan (use ONCE, then pick specific targets from the scan)
  unmark <objectType>        — undo a SEARCHED mark (use when you suspect the target is still inside)
  wait                       — nothing to do, task in progress
  Done                       — task is complete (see below)

OUTPUT — valid JSON only. { first char, } last char. No markdown.

{
  "intent": "<intent phrase>",
  "target": "<objectType or empty>",
  "reasoning": "<1-3 sentences: why this intent now, first-person>"
}"""


# ---------------------------------------------------------------------------
# Spike 003: Curiosity Scoreboard — prompt-bridge helpers
# ---------------------------------------------------------------------------

# Known AI2-THOR object types for extracting the target from task_goal
_CURIOSITY_KNOWN_TYPES: set[str] = {
    "AlarmClock", "Apple", "BaseballBat", "BasketBall", "Book", "Bowl", "Box",
    "Bread", "BreadSliced", "ButterKnife", "Candle", "CD", "CellPhone",
    "Cloth", "CoffeeMachine", "CreditCard", "Cup", "DishSponge", "Dumbbell",
    "Egg", "Fork", "HandTowel", "KeyChain", "Knife", "Ladle", "Laptop",
    "Lettuce", "Mug", "Newspaper", "Pan", "Pen", "Pencil", "PepperShaker",
    "Pillow", "Plate", "Plunger", "Pot", "Potato", "RemoteControl",
    "SaltShaker", "ScrubBrush", "SoapBar", "SoapBottle", "Spatula",
    "SprayBottle", "Statue", "TeddyBear", "TennisRacket", "TissueBox",
    "ToiletPaper", "Tomato", "Towel", "Vase", "Watch", "WateringCan",
    "WineBottle",
}


def _extract_target_object(task_goal: str) -> str:
    """Extract the most likely target object type from a task goal string.

    Uses a simple heuristic: find known AI2-THOR object types in the goal text.
    Returns the first match, or empty string if none found.
    """
    goal_lower = task_goal.lower()
    # Sort by length descending so "AlarmClock" matches before "Clock"
    for otype in sorted(_CURIOSITY_KNOWN_TYPES, key=len, reverse=True):
        if otype.lower() in goal_lower:
            return otype
    return ""


def _build_curiosity_for_prompt(task_goal: str, memory) -> str:
    """Build the curiosity scoreboard text for injection into the Planner prompt.

    Extracts the target object from task_goal, gathers receptacle entries from
    memory, and delegates to curiosity_scorer.build_curiosity_table_text().
    """
    target_object = _extract_target_object(task_goal)
    if not target_object:
        return ""

    entries = memory.get_receptacle_entries_for_curiosity()
    if not entries:
        return ""

    self_ref = memory  # for clarity in the call below
    return build_curiosity_table_text(
        target_object=target_object,
        memory_entries=entries,
        visit_counts=self_ref.receptacle_visit_counts,
        open_counts=self_ref.receptacle_open_counts,
        objects_found_counts=self_ref.objects_found_by_receptacle,
    )


def _check_curiosity_posthoc(
    task_goal: str,
    proposed_target: str,
    memory,
    curiosity_stats: CuriosityStats,
) -> dict:
    """Post-hoc check: is the proposed target reasonable per curiosity scores?

    Returns dict with: blocked, warning, fallback, score.
    """
    target_object = _extract_target_object(task_goal)
    if not target_object or not proposed_target:
        return {"blocked": False, "warning": None, "fallback": None, "score": 1.0}

    result = check_proposed_target(
        target_object=target_object,
        proposed_target=proposed_target,
        visit_counts=memory.receptacle_visit_counts,
        open_counts=memory.receptacle_open_counts,
        objects_found_counts=memory.objects_found_by_receptacle,
    )

    curiosity_stats.record_proposal(proposed_target, result["score"])

    if result.get("blocked"):
        fallback = result.get("best_alternative", "scan room")
        fallback_intent = f"approach {fallback}" if fallback and fallback != "scan room" else "scan room"
        curiosity_stats.record_block(proposed_target, result["score"], fallback_intent)
        result["fallback"] = fallback_intent
    elif result.get("warning"):
        curiosity_stats.soft_warnings += 1

    return result
# ------------------------------------------------------------------
# Phase-specific guardrails (Spike 005: progress-gating)
# ------------------------------------------------------------------

PHASE_RULES: dict[str, str] = {
    "exploration": (
        "\n"
        "PHASE: EXPLORATION\n"
        "The target has NOT been seen yet. You are surveying the environment.\n"
        "- Visit each candidate location ONCE. Do not revisit.\n"
        "- Start with the nearest visible receptacle.\n"
        "- After opening a receptacle, mentally mark it as CHECKED.\n"
        "- If all visible receptacles are checked, rotate to scan new areas.\n"
        "- Do NOT wander aimlessly — move with purpose toward unchecked areas."
    ),
    "navigation": (
        "\n"
        "PHASE: NAVIGATION\n"
        "Target was seen but is not currently visible. Navigate to its last known location.\n"
        "- Move toward the known target location from spatial memory.\n"
        "- Check your memory for the last known direction and distance.\n"
        "- If blocked en route, go around — do NOT abandon the destination.\n"
        "- Rotate periodically to verify you haven't passed the target."
    ),
    "approach": (
        "\n"
        "PHASE: APPROACH\n"
        "Target is visible but > 0.5m away. Close distance for interaction.\n"
        "- Close distance to within 0.5m for interaction.\n"
        "- Use MoveSequence for efficient multi-step movement.\n"
        "- If blocked, sidestep or find an alternative angle.\n"
        "- Do NOT attempt PickupObject/OpenObject until within 0.5m."
    ),
    "interaction": (
        "\n"
        "PHASE: INTERACTION\n"
        "Target is within reach. Manipulate it to progress the task.\n"
        "- Pick up the target, open it, or manipulate it as needed.\n"
        "- After interacting, re-assess: is the task complete?\n"
        "- If the interaction fails, diagnose why before retrying."
    ),
    "recovery": (
        "\n"
        "PHASE: RECOVERY\n"
        "The last action FAILED. You must diagnose and try an alternative.\n"
        "- Do NOT repeat the same action or approach the same target.\n"
        "- If a receptacle was opened and found empty, mark it as SEARCHED.\n"
        "- Rotate 90 and look for alternative paths or unexplored receptacles.\n"
        "- If stuck for 3+ steps, propose \"scan room\" to re-assess.\n"
        "- Identify the root cause: was it a collision, a reach error, or a missing object?"
    ),
}



class EBAgent:
    def __init__(self, client: VLMClient):
        self.client = client
        self._pending_dedup_constraint: str | None = None
        self._pending_trail_warning: str | None = None
        # Dedup statistics (public, reset per episode)
        self.dedup_stats: dict = {
            "warnings": 0,
            "forced": 0,
            "fallbacks": [],  # list of {"original_intent": str, "original_target": str, "fallback_intent": str}
            "blocked_intents": [],  # list of blocked (intent, target) pairs with counts
        }
        # Contrastive planner statistics (Spike 004, public, reset per episode)
        self.contrastive_stats: dict = {
            "activations": 0,          # how many times dual-planner was triggered
            "a_selected": 0,           # exploit (standard) chosen
            "b_selected": 0,           # explore chosen
            "selections": [],          # list of {"step": int, "chosen": "A"/"B", "a_intent": str, "b_intent": str, "reason": str}
        }
        # ── Spike 003: Curiosity Scoreboard statistics ──
        self.curiosity_stats = CuriosityStats()
        self._pending_curiosity_warning: str | None = None
        # ── Spike 005: Progress-gating phase tracking ──
        self._current_phase: str | None = None
        self._phase_transitions: list[dict] = []  # [{from, to, step_index}]
        # ── Ablation flags (set by BranchRunner before run; all default True) ──
        self.enable_intent_dedup: bool = True
        self.enable_curiosity_scoreboard: bool = True
        self.enable_progress_gating: bool = True

    def reset_dedup_stats(self):
        """Reset dedup, contrastive & curiosity statistics for a new episode."""
        self._pending_dedup_constraint = None
        self._pending_trail_warning = None
        self.dedup_stats = {
            "warnings": 0,
            "forced": 0,
            "fallbacks": [],
            "blocked_intents": [],
        }
        self.contrastive_stats = {
            "activations": 0,
            "a_selected": 0,
            "b_selected": 0,
            "selections": [],
        }
        # Spike 003: reset curiosity state
        self.curiosity_stats = CuriosityStats()
        self._pending_curiosity_warning = None
        # ── Spike 005: reset phase tracking ──
        self._current_phase = None
        self._phase_transitions = []

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
        intent_history: list[dict] | None = None,
        memory=None,
        critic_feedback: str | None = None,
        trail_text: str = "",
    ) -> dict:
        """Propose the next high-level intent.

        If memory (EgocentricMemory) is provided, intent deduplication is active:
        - 2 prior INCOMPLETE occurrences of the same (intent, target) -> injects a
          hard constraint into the NEXT prompt.
        - 3+ prior INCOMPLETE occurrences -> FORCE overrides the intent with a
          fallback exploration strategy using spatial memory.

        If critic_feedback is provided (from CriticGuard rejection), it is shown
        prominently as a hard constraint to guide re-generation.
        """
        lines = [f"Task goal: {task_goal}\n"]
        # ── Phase detection and guardrail injection (Spike 005: progress-gating) ──
        if self.enable_progress_gating:
            try:
                phase = self._detect_phase(
                    task_goal, visible_objects, memory, last_error,
                    intent_history, agent_pos, agent_rot_y,
                )
                if phase != self._current_phase:
                    self._phase_transitions.append({
                        "from": self._current_phase,
                        "to": phase,
                        "step_index": len(action_history),
                    })
                    self._current_phase = phase
                phase_block = self._get_phase_block(phase)
                if phase_block:
                    lines.append(phase_block)
            except Exception:
                pass  # Phase detection must never break the main flow

        # ── Critic feedback (002b): hard constraint from pre-execution Critic ──
        if critic_feedback:
            lines.append("=" * 50)
            lines.append("CRITIC REJECTED YOUR PREVIOUS INTENT:")
            lines.append(critic_feedback)
            lines.append("You MUST propose a DIFFERENT intent. Do NOT repeat the rejected one.")
            lines.append("=" * 50)
            lines.append("")
        if memory_text:
            lines.append(memory_text)
            lines.append("")

        # ── Spike 003: Curiosity Scoreboard ──
        # Compute a multi-dimensional exploration priority table for known receptacles.
        # Injected proactively — the Planner sees scores BEFORE forming its intent.
        curiosity_table = ""
        if self.enable_curiosity_scoreboard and memory is not None:
            curiosity_table = _build_curiosity_for_prompt(task_goal, memory)
        if curiosity_table:
            lines.append(curiosity_table)
            self.curiosity_stats.scoreboard_shown += 1

        if hand_status:
            lines.append(f"HAND STATUS: {hand_status}")
        if task_criteria:
            lines.append(f"\nTASK COMPLETION CRITERIA:\n{task_criteria}")

        # ── Inject pending dedup constraint from a previous 2-occurrence warning ──
        if self._pending_dedup_constraint:
            lines.append(f"\nHARD CONSTRAINT: {self._pending_dedup_constraint}")
            self._pending_dedup_constraint = None

        # ── Inject pending trail warning (Spike 006: area visited 3+ times) ──
        if self._pending_trail_warning:
            lines.append(f"\nTRAIL NOTE: {self._pending_trail_warning}")
            self._pending_trail_warning = None

        # ── Inject pending curiosity warning (Spike 003: low-score target blocked) ──
        if self._pending_curiosity_warning:
            lines.append(f"\nCURIOSITY WARNING: {self._pending_curiosity_warning}")
            self._pending_curiosity_warning = None

        # ── Trail summary (Spike 006: search-trail-cost) ──
        if trail_text:
            lines.append("")
            lines.append(trail_text)
            lines.append("")

        # Unified intent tree — replaces both "Intent history" and "Full EB history"
        if intent_history:
            tree_text = render_intent_tree(intent_history, action_history)
            if tree_text:
                lines.append(tree_text)
                lines.append("")
        elif action_history:
            # Fallback: show recent raw actions if no intent history yet
            recent_raw = action_history[-3:] if len(action_history) > 3 else action_history
            if recent_raw:
                lines.append("\nRecent actions:")
                lines.append(build_eb_history_context(recent_raw))

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
                    d = " <- " + _direction_for_obj(agent_pos, agent_rot_y, o)
                lines.append(f"  {o['objectType']}{tag}{d}")
        else:
            lines.append("\n(No objects in view — you may be facing a wall. Rotate or MoveBack.)")

        if last_error:
            lines.append(f"\nLast error: {last_error}")

        lines.append("\nPropose the next intent. Be specific. Output JSON only.")
        prompt = "\n".join(lines)

        result = self.client.chat_with_image_json(
            system_prompt=PLANNER_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("intent", "target", "reasoning"),
        )

        # ── Intent deduplication — check the proposed intent against history ──
        if self.enable_intent_dedup and intent_history and memory is not None:
            intent = result.get("intent", "")
            target = result.get("target", "")
            blocked, warning, fallback_intent = self._check_intent_dedup(
                intent, target, intent_history, memory, task_goal,
            )
            if blocked:
                result["intent"] = fallback_intent
                result["target"] = ""
                original_reasoning = result.get("reasoning", "")
                result["reasoning"] = (
                    f"[DEDUP OVERRIDE] Original intent '{intent} {target}' blocked "
                    f"after repeated failures. Fallback: {fallback_intent}. "
                    f"{original_reasoning}"
                )
                result["dedup_blocked"] = True
                result["dedup_original_intent"] = intent
                result["dedup_original_target"] = target
            elif warning:
                # Store for next prompt — the current proposal passes but next
                # time the Planner will see a hard constraint.
                self._pending_dedup_constraint = warning
                result["dedup_warning"] = warning

        # ── Spike 003: Curiosity Scoreboard post-hoc check ──
        # After the VLM returns, check whether the proposed target has a reasonable
        # curiosity score. Unlike dedup (which only blocks exact repeats), this
        # checks ALL locations against the multi-dimensional score.
        if self.enable_curiosity_scoreboard and memory is not None and not result.get("dedup_blocked"):
            intent = result.get("intent", "")
            target = result.get("target", "")
            # Only check approach/open/check intents that target a receptacle
            if target and any(kw in intent.lower() for kw in ("approach", "open", "check", "search")):
                curiosity_check = _check_curiosity_posthoc(
                    task_goal, target, memory, self.curiosity_stats
                )
                if curiosity_check.get("blocked"):
                    # Hard guard: score too low, force fallback
                    fallback = curiosity_check.get("fallback", "scan room")
                    result["intent"] = fallback
                    result["target"] = ""
                    original_reasoning = result.get("reasoning", "")
                    result["reasoning"] = (
                        f"[CURIOSITY OVERRIDE] '{target}' curiosity score "
                        f"({curiosity_check.get('score', 0):.2f}) too low. "
                        f"Fallback: {fallback}. {original_reasoning}"
                    )
                    result["curiosity_blocked"] = True
                    result["curiosity_original_target"] = target
                elif curiosity_check.get("warning"):
                    self._pending_curiosity_warning = curiosity_check["warning"]
                    result["curiosity_warning"] = curiosity_check["warning"]

        result["phase"] = getattr(self, "_current_phase", None)
        return result

    # ------------------------------------------------------------------
    # Intent deduplication (hard mechanism, not prompt-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_intent_verb(intent: str) -> str:
        """Normalize intent verb to canonical form for dedup matching.

        LLM variants like "go to X", "move to X", "walk to X" all mean
        "approach X". Normalize them so dedup sees them as the same intent.
        """
        if not intent:
            return ""
        lower = intent.strip().lower()
        # Common approach synonyms
        for prefix in ("go to ", "move to ", "walk to ", "head to ",
                       "navigate to ", "proceed to "):
            if lower.startswith(prefix):
                return "approach"
        # For other intents, use the first word as the verb
        return lower.split()[0] if lower.split() else lower

    @staticmethod
    def _normalize_target(target: str, memory) -> str:
        """Normalize an LLM-generated target string to its canonical objectType.

        Checks against all known object types in spatial memory using
        case-insensitive exact match and substring containment. Returns
        the canonical casing if a match is found, otherwise the original.
        """
        if not target or memory is None:
            return target.strip() if target else ""
        target_clean = target.strip()
        target_lower = target_clean.lower()

        all_types = memory.get_all_object_types()

        # 1. Case-insensitive exact match
        for otype in all_types:
            if otype.lower() == target_lower:
                return otype

        # 2. Substring containment (e.g. "Counter" -> "CounterTop",
        #    "desk" -> "DeskLamp")
        for otype in all_types:
            otype_lower = otype.lower()
            if target_lower in otype_lower or otype_lower in target_lower:
                return otype

        return target_clean

    def _check_intent_dedup(
        self,
        intent: str,
        target: str,
        intent_history: list[dict],
        memory=None,
        task_goal: str = "",
    ) -> tuple[bool, str | None, str | None]:
        """Check whether the proposed (intent, target) pair has been tried too many
        times without success.

        Uses semantic normalization: intent verbs are normalized ("go to" ->
        "approach") and targets are normalized to canonical objectTypes from
        spatial memory. This prevents LLM synonym variants from bypassing dedup.

        Returns (blocked, warning_message, fallback_intent):
        - blocked=False, warning=str:  2 prior INCOMPLETE occurrences -> inject
          hard constraint into the NEXT prompt.
        - blocked=True, warning=None:  3+ prior INCOMPLETE occurrences -> force
          override with fallback_intent.
        - blocked=False, warning=None: normal, no dedup action.
        """
        if not intent_history:
            return (False, None, None)

        # ── Semantic normalization (C5 fix) ──
        norm_verb = self._normalize_intent_verb(intent)
        norm_target = self._normalize_target(target, memory)
        key = (norm_verb, norm_target)

        # Only look at the most recent 5 intent history entries
        recent = intent_history[-5:] if len(intent_history) > 5 else intent_history

        # Match against history using normalized keys
        matches: list[dict] = []
        for ih in recent:
            ih_verb = self._normalize_intent_verb(ih.get("intent", ""))
            ih_target = self._normalize_target(ih.get("target", ""), memory)
            if (ih_verb, ih_target) == key:
                matches.append(ih)

        if len(matches) == 0:
            return (False, None, None)

        # Check if ALL prior matches were INCOMPLETE
        all_incomplete = all(not m.get("completed", False) for m in matches)

        if not all_incomplete:
            # If any prior attempt completed, dedup is not triggered
            return (False, None, None)

        count = len(matches)

        # ── Level 2: 3+ prior occurrences -> FORCE override ──
        if count >= 3:
            fallback_intent = self._generate_fallback_intent(
                intent, target, memory, intent_history, task_goal,
            )
            self.dedup_stats["forced"] += 1
            self.dedup_stats["fallbacks"].append({
                "original_intent": intent,
                "original_target": target,
                "fallback_intent": fallback_intent,
                "prior_attempts": count,
            })
            self.dedup_stats["blocked_intents"].append({
                "intent": intent,
                "target": target,
                "prior_attempts": count,
            })
            return (True, None, fallback_intent)

        # ── Level 1: 2 prior occurrences -> warning -> hard constraint next call ──
        if count >= 2:
            self.dedup_stats["warnings"] += 1
            desc = intent if target and target in intent else f"{intent} {target}".strip()
            warning = (
                f"You have already tried '{desc}' {count} times "
                f"without finding the target. Choose a DIFFERENT approach. "
                f"Do NOT propose '{desc}' again."
            )
            return (False, warning, None)

        return (False, None, None)

    # ------------------------------------------------------------------
    # Task-aware fallback helpers (C3 fix)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pick_two_task(task_goal: str) -> bool:
        """Detect whether task_goal describes a pick-two / multi-object task."""
        if not task_goal:
            return False
        lower = task_goal.lower()
        return ("pick two" in lower or "pick 2" in lower
                or " both " in lower or "and the " in lower)

    @staticmethod
    def _extract_parent_target(task_goal: str) -> str:
        """Extract the task's target receptacle from a task_goal string.

        Examples:
          "Pick up the Apple and put it on the Table."   -> "Table"
          "Pick up the Apple and put it in the Fridge."   -> "Fridge"
          "Heat the Apple and put it in the Microwave."   -> "Microwave"
        """
        if not task_goal:
            return ""
        m = re.search(r'(?:on|in|into)\s+the\s+([A-Z][a-zA-Z]+)', task_goal)
        if m:
            return m.group(1)
        return ""

    @staticmethod
    def _extract_object_target(task_goal: str) -> str:
        """Extract the primary object to pick from a task_goal string.

        Examples:
          "Pick up the Apple and put it on the Table."    -> "Apple"
          "Clean the Potato and put it in the Fridge."    -> "Potato"
        """
        if not task_goal:
            return ""
        m = re.search(r'(?:Pick up|Clean|Heat|Cool|Slice)\s+the\s+([A-Z][a-zA-Z]+)', task_goal)
        if m:
            return m.group(1)
        return ""

    def _generate_fallback_intent(
        self,
        intent: str,
        target: str,
        memory=None,
        intent_history: list[dict] | None = None,
        task_goal: str = "",
    ) -> str:
        """Generate a fallback exploration intent when the Planner is stuck in a
        repeated-intent loop. Uses spatial memory to find alternative targets.

        Task-aware (C3 fix):
        - NEVER blocks or deprioritises the task's TARGET receptacle (parent_target).
        - For pick_two tasks: the same source receptacle may need to be visited
          twice, so previously-approached receptacles are not penalised.

        Priority:
        1. Task target receptacle (parent_target) if visible or remembered
        2. Unvisited receptacles from spatial memory (excluding approached,
           but never excluding the task's parent_target)
        3. Visible receptacles not yet approached
        4. "scan room" -- ultimate fallback
        """
        # ── Parse task info (C3) ──
        parent_target = self._extract_parent_target(task_goal)
        object_target = self._extract_object_target(task_goal)
        is_pick_two = self._is_pick_two_task(task_goal)

        # Collect object types that have been repeatedly targeted (approached),
        # but NEVER include the task's parent_target.
        approached_types: set[str] = set()
        if intent_history:
            for ih in intent_history:
                ih_intent = ih.get("intent", "")
                ih_target = ih.get("target", "")
                norm_verb = self._normalize_intent_verb(ih_intent)
                if norm_verb == "approach" and ih_target:
                    norm_t = self._normalize_target(ih_target, memory)
                    # Guard: never block the task's target receptacle
                    if parent_target and norm_t.lower() == parent_target.lower():
                        continue
                    approached_types.add(norm_t)

        if memory is not None:
            # ── Priority 1: task target receptacle ──
            if parent_target:
                # Check if parent_target is visible right now
                visible_rec = memory.get_visible_receptacles()
                for rec in visible_rec:
                    if rec.lower() == parent_target.lower():
                        return f"approach {rec}"
                # Check if parent_target is remembered
                remembered = memory.get_remembered_receptacles()
                for rec in remembered:
                    if rec.lower() == parent_target.lower():
                        return f"approach {rec}"

            # ── Priority 2: unvisited receptacles (never block parent_target) ──
            unvisited = memory.get_unvisited_receptacles(approached_types)
            if unvisited:
                return f"approach {unvisited[0]}"

            # ── Priority 3: visible receptacles ──
            visible_rec = memory.get_visible_receptacles()
            if visible_rec:
                for rec in visible_rec:
                    # For pick_two, allow previously-approached receptacles
                    # (the second object may be on the same surface)
                    if is_pick_two:
                        return f"approach {rec}"
                    if rec not in approached_types:
                        return f"approach {rec}"
                # All visible receptacles approached -- still try the nearest
                return f"approach {visible_rec[0]}"

        # ── Ultimate fallback ──
        return "scan room"

    # ------------------------------------------------------------------
    # Phase detection for progress-gating (Spike 005)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_targets(task_goal: str) -> list[str]:
        """Extract object type names from a task goal string.

        Task goals follow ALFRED conventions like:
          "Pick up the Apple and put it on the CounterTop."
          "Clean the Apple and put it in the Fridge."
        We extract CamelCase words that follow "the ".
        """
        import re
        targets = re.findall(r'\bthe\s+([A-Z][a-zA-Z]+)', task_goal)
        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in targets:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def _detect_phase(
        self,
        task_goal: str,
        visible_objects: list[dict],
        memory,
        last_error: str | None,
        intent_history: list[dict] | None,
        agent_pos: dict | None = None,
        agent_rot_y: float = 0.0,
    ) -> str:
        """Detect the current task phase from state heuristics.

        Priority order (first match wins):
          1. recovery  — last action failed
          2. exploration — target NEVER seen in spatial memory
          3. navigation — target seen but not currently visible
          4. approach — target visible, distance > 0.5m
          5. interaction — target visible, distance <= 0.5m
        Falls back to exploration if no target can be extracted.
        """
        targets = self._extract_targets(task_goal)
        primary = targets[0] if targets else None

        # Check for recent failure (recovery)
        recent_failure = False
        if last_error:
            recent_failure = True
        elif intent_history:
            recent = intent_history[-3:] if len(intent_history) > 3 else intent_history
            for ih in recent:
                if not ih.get("completed", False):
                    recent_failure = True
                    break

        # Query spatial memory for target status
        target_seen = False
        target_visible = False
        target_distance: float | None = None

        if primary and memory is not None:
            target_seen = memory.has_type(primary)
            target_visible = memory.is_type_visible(primary)
            if target_visible:
                target_distance = memory.get_type_distance(primary)
            elif target_seen:
                target_distance = memory.get_remembered_type_distance(primary)

        # Fallback: check visible_objects when memory unavailable
        if primary and not target_seen and memory is None:
            for obj in visible_objects:
                if obj.get("visibleBounds2D") and obj.get("objectType") == primary:
                    target_visible = True
                    target_seen = True
                    surf = _surface_position(agent_pos, obj) if agent_pos else None
                    if surf and agent_pos:
                        dx = surf["x"] - agent_pos["x"]
                        dz = surf["z"] - agent_pos["z"]
                        target_distance = math.sqrt(dx * dx + dz * dz)
                    break

        # Also check visible_objects even when memory exists (may not be updated yet)
        if primary and not target_visible and memory is not None:
            for obj in visible_objects:
                if obj.get("visibleBounds2D") and obj.get("objectType") == primary:
                    target_visible = True
                    if target_distance is None:
                        surf = _surface_position(agent_pos, obj) if agent_pos else None
                        if surf and agent_pos:
                            dx = surf["x"] - agent_pos["x"]
                            dz = surf["z"] - agent_pos["z"]
                            target_distance = math.sqrt(dx * dx + dz * dz)
                    break

        # Phase decision
        if recent_failure:
            return "recovery"

        if primary is None:
            return "exploration"

        if not target_seen:
            return "exploration"

        if target_visible and target_distance is not None and target_distance <= 0.5:
            return "interaction"

        if target_visible and target_distance is not None and target_distance > 0.5:
            return "approach"

        if target_seen and not target_visible:
            return "navigation"

        return "exploration"

    def _get_phase_block(self, phase: str) -> str:
        """Return the phase-specific guardrail text to inject into the prompt."""
        return PHASE_RULES.get(phase, "")

    # ------------------------------------------------------------------
    # Contrastive Planner (Spike 004): explore-biased intent proposal
    # ------------------------------------------------------------------

    @staticmethod
    def _should_activate_contrastive(intent_history: list[dict] | None) -> bool:
        """Gating condition for contrastive dual-planner.

        Only activate when the Planner appears stuck in a fixation loop:
        the last 3 intent history entries all targeted the same location type.

        This avoids doubling API calls on every step — based on the Dynamic
        Self-Consistency (RASC, 2024) insight that extra samples are only
        needed when the model is uncertain/stuck.
        """
        if not intent_history or len(intent_history) < 3:
            return False
        recent = intent_history[-3:]
        targets = [ih.get("target", "") for ih in recent]
        # All 3 must have a non-empty target AND all be the same
        if not all(targets):
            return False
        if len(set(targets)) != 1:
            return False
        # All 3 must be incomplete (failed or abandoned)
        if all(ih.get("completed", False) for ih in recent):
            return False
        return True

    def plan_intent_explore(
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
        intent_history: list[dict] | None = None,
        memory=None,
    ) -> dict:
        """Propose an explore-biased intent. Same context as plan_intent() but:
        - Uses EXPLORE_BIAS_SYSTEM instead of PLANNER_SYSTEM
        - Injects explicit "avoid these recently targeted locations" warning
        - Prioritizes unvisited receptacles from spatial memory

        Based on the DiscussNav (ICRA 2024) principle: diverse expert
        perspectives produce better navigation decisions than a single view.
        """
        lines = [f"Task goal: {task_goal}\n"]

        if memory_text:
            lines.append(memory_text)
            lines.append("")
        if hand_status:
            lines.append(f"HAND STATUS: {hand_status}")
        if task_criteria:
            lines.append(f"\nTASK COMPLETION CRITERIA:\n{task_criteria}")

        # ── Exploration bias: list recently targeted locations to AVOID ──
        if intent_history:
            # Collect unique targets from recent incomplete intents
            recent_targets: list[str] = []
            for ih in reversed(intent_history[-10:]):
                t = ih.get("target", "")
                if t and t not in recent_targets:
                    recent_targets.append(t)
                if len(recent_targets) >= 5:
                    break

            if recent_targets:
                lines.append("\n" + "=" * 50)
                lines.append("EXPLORATION MODE — AVOID THESE RECENTLY TARGETED LOCATIONS:")
                for t in recent_targets:
                    lines.append(f"  - {t}")
                lines.append("Choose a DIFFERENT location. Prioritize ones you have NOT yet visited.")
                lines.append("=" * 50)

            # Unified intent tree — replaces both "Intent history" and "Full EB history"
            tree_text = render_intent_tree(intent_history, action_history)
            if tree_text:
                lines.append(tree_text)
                lines.append("")

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
                    d = " <- " + _direction_for_obj(agent_pos, agent_rot_y, o)
                lines.append(f"  {o['objectType']}{tag}{d}")
        else:
            lines.append("\n(No objects in view — you may be facing a wall. Rotate or MoveBack.)")

        if last_error:
            lines.append(f"\nLast error: {last_error}")

        lines.append("\nPropose the next intent. EXPLORATION MODE: prioritize unvisited locations.")
        lines.append("Be specific. Output JSON only.")
        prompt = "\n".join(lines)

        result = self.client.chat_with_image_json(
            system_prompt=EXPLORE_BIAS_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("intent", "target", "reasoning"),
        )

        # ── Still apply dedup check to explore intents ──
        if self.enable_intent_dedup and intent_history and memory is not None:
            intent = result.get("intent", "")
            target = result.get("target", "")
            blocked, warning, fallback_intent = self._check_intent_dedup(
                intent, target, intent_history, memory, task_goal,
            )
            if blocked:
                result["intent"] = fallback_intent
                result["target"] = ""
                original_reasoning = result.get("reasoning", "")
                result["reasoning"] = (
                    f"[DEDUP OVERRIDE] Original explore intent '{intent} {target}' "
                    f"blocked after repeated failures. Fallback: {fallback_intent}. "
                    f"{original_reasoning}"
                )
                result["dedup_blocked"] = True
                result["dedup_original_intent"] = intent
                result["dedup_original_target"] = target
            elif warning:
                self._pending_dedup_constraint = warning
                result["dedup_warning"] = warning

        return result

    # ==================================================================
    # Contrastive Selector (Spike 004): pick between exploit (A) and explore (B)
    # ==================================================================

    @staticmethod
    def select_intent(
        intent_a: dict,
        intent_b: dict,
        intent_history: list[dict] | None = None,
    ) -> tuple[dict, str, str]:
        """Select between two Planner proposals: A (exploit, standard) and B (explore).

        Selection heuristics (in priority order):
        1. If A matches a previously-tried-and-failed intent AND B does not -> pick B
        2. If A's (intent, target) pair appears 2+ times in recent history -> pick B
        3. If B's intent is identical to A's -> pick A (B added no diversity)
        4. Default: pick A (exploit is more efficient when not stuck)

        Returns (selected_intent_dict, chosen_label, reason_string).

        Grounded in:
        - Self-Consistency (Wang et al. 2022): majority-vote among diverse samples.
          Here, with only 2 samples, we use heuristic selection instead of voting.
        - DiscussNav (ICRA 2024): the "Decision Testing Expert" evaluates competing
          proposals and picks the best one using task-specific criteria.
        """
        intent_history = intent_history or []

        a_intent = intent_a.get("intent", "")
        a_target = intent_a.get("target", "")
        b_intent = intent_b.get("intent", "")
        b_target = intent_b.get("target", "")

        # ── Heuristic 1: A repeats a known-failed intent ──
        a_failed_before = False
        b_failed_before = False
        for ih in intent_history:
            if not ih.get("completed", False):
                if (ih.get("intent"), ih.get("target")) == (a_intent, a_target):
                    a_failed_before = True
                if (ih.get("intent"), ih.get("target")) == (b_intent, b_target):
                    b_failed_before = True
        if a_failed_before and not b_failed_before:
            return intent_b, "B", "A_intent_matches_known_failure"

        # ── Heuristic 2: A has been tried 2+ times recently ──
        recent = intent_history[-5:] if len(intent_history) > 5 else intent_history
        a_count = sum(
            1 for ih in recent
            if (ih.get("intent"), ih.get("target")) == (a_intent, a_target)
        )
        if a_count >= 2:
            return intent_b, "B", f"A_intent_tried_{a_count}_times"

        # ── Heuristic 3: B added no diversity (same as A) ──
        if (a_intent, a_target) == (b_intent, b_target):
            return intent_a, "A", "intents_identical"

        # ── Heuristic 4: B targets a SEARCHED receptacle → pick A ──
        # (B is wasting time revisiting a known-empty location)
        b_has_searched = "[SEARCHED]" in intent_b.get("reasoning", "")
        if b_has_searched:
            return intent_a, "A", "B_targets_searched_receptacle"

        # ── Default: prefer A (exploit is more efficient) ──
        return intent_a, "A", "default_prefer_exploit"

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
        last_error: str | None = None,
        memory_text: str = "",
        agent_pos: dict = None,
        agent_rot_y: float = 0.0,
    ) -> dict:
        """Review Executor's action sequence. If rejected, provide corrected actions."""
        lines = [f"Your intent was: {intent}"]
        if target:
            lines.append(f"Target: {target}")
        lines.append("")

        # Same context as Planner/Executor — spatial memory, objects in view, recent history, last error
        if memory_text:
            lines.append(memory_text)
            lines.append("")

        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        if visible:
            lines.append("Objects in view:")
            for o in visible[:10]:
                extra = []
                if o.get("isPickedUp"): extra.append("held")
                if o.get("receptacle"): extra.append("receptacle")
                if o.get("openable"): extra.append("open")
                tag = f" ({','.join(extra)})" if extra else ""
                d = ""
                if agent_pos and o.get("position"):
                    d = " <- " + _direction_for_obj(agent_pos, agent_rot_y, o)
                lines.append(f"  {o['objectType']}{tag}{d}")
        else:
            lines.append("(No objects in view)")
        lines.append("")

        recent = action_history[-5:] if len(action_history) > 5 else action_history
        if recent:
            lines.append("Recent actions:")
            lines.append(build_eb_history_context(recent))
            lines.append("")

        if last_error:
            lines.append(f"Last error: {last_error}")
            lines.append("If the last 2+ actions all failed on the first step, the agent is STUCK. MoveBack IS correct.\n")

        lines.append("Executor proposed these actions:")
        for i, a in enumerate(proposed_actions, 1):
            act = a.get("action", "?")
            params = a.get("params", {})
            if params:
                lines.append(f"  {i}. {act}({params})")
            else:
                lines.append(f"  {i}. {act}")
        lines.append(f"\nExecutor reasoning: {executor_reasoning}")

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
        lines.append("Use the views to find your target and decide your next move.")
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

    # ------------------------------------------------------------------
    # Planner: analyze full-room scan → direction + intent for Executor
    # ------------------------------------------------------------------
    SCAN_ANALYSIS_SYSTEM = """You are a task planner analyzing a full-room scan. You see 4 directional views: VIEW 1: ahead, VIEW 2: left, VIEW 3: behind, VIEW 4: right. You MUST output a direction to face AND a regular intent for the Executor to carry out.

RULES:
- Look at ALL 4 views carefully. Locate the task-relevant objects.
- "face_direction": which way to turn — "ahead", "left", "behind", or "right". Pick the direction that gives the best view of the target or the area to explore.
- "intent": a regular intent (approach X, pickup X, open X, etc.) from the Planner's standard intent list. NEVER use "locate X". If unsure of the target's location, pick a specific receptacle to approach and search.
- "target": the objectType to target, or empty string if the intent has no target.

OUTPUT — valid JSON only:

{
  "face_direction": "left",
  "intent": "approach Counter",
  "target": "Counter",
  "reasoning": "<what you see in each direction, why this direction and intent>"
}"""

    def analyze_scan_room(
        self,
        task_goal: str,
        look_images: list,
        visible_objects: list[dict],
        action_history: list[dict],
        last_error: str | None = None,
        inventory_objects: list[dict] | None = None,
        task_criteria: str = "",
        memory_text: str = "",
    ) -> dict:
        """After 4-view scan, analyze room and produce direction + intent for Executor."""
        NL = chr(10)
        lines = []
        lines.append(f"Task: {task_goal}")
        lines.append("")
        _append_task_context(lines, visible_objects, inventory_objects, task_criteria)
        lines.append("You just did a full-room scan. Below are 4 views: ahead, left, behind, right.")
        lines.append("Decide which direction to face and what intent to pursue.")
        lines.append("")
        if memory_text:
            lines.append(memory_text)
        else:
            lines.append(build_eb_history_context(action_history))
        if last_error:
            lines.append(f"\nLast error: {last_error}")
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
        lines.append("CRITICAL: Use EXACT objectType from the list above. Output direction + intent + target.")
        prompt = NL.join(lines)

        result = self.client.chat_with_images_json(
            system_prompt=self.SCAN_ANALYSIS_SYSTEM,
            user_text=prompt,
            images=look_images,
            required_fields=("face_direction", "intent", "target", "reasoning"),
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
        current_intent: str = "",
        trap_state: list[dict] | None = None,
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
                                     memory_text=memory_text, current_intent=current_intent,
                                     trap_state=trap_state)
        result = self.client.chat_with_image_json(
            system_prompt=PHASE3_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("diagnosis", "recovery_reasoning", "counterfactual", "proposed_recovery_action"),
        )
        if not result.get("counterfactual") or not isinstance(result["counterfactual"], dict):
            import logging
            logging.warning("Phase 3: counterfactual is null — retrying with explicit reminder")
            retry_prompt = prompt + (
                "\n\nCRITICAL REMINDER: Your previous response did not include a counterfactual. "
                "You MUST output a counterfactual. Every failure traces back to an earlier decision. "
                "Look at the action history, find a step where a different action would have avoided this error, "
                "and provide the target_step, alternative_action, and reasoning. "
                "Do NOT output null for counterfactual."
            )
            result = self.client.chat_with_image_json(
                system_prompt=PHASE3_SYSTEM,
                user_text=retry_prompt,
                image=image,
                required_fields=("diagnosis", "recovery_reasoning", "counterfactual", "proposed_recovery_action"),
            )
        return result
