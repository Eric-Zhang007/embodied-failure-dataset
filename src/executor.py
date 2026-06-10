"""
Executor — takes a Planner intent + full agent context → outputs concrete action chunk.
Same agent identity as Planner (first-person). Uses 8B model.

Rich context: spatial memory, objects in view with directions, hand status,
task criteria, recent history, last error — same information the Planner sees.
"""
from __future__ import annotations
import numpy as np
from src.vlm_client import VLMClient
from src.context_builder import build_eb_history_context

EXECUTOR_SYSTEM = """You are an embodied agent in a 3D household. The image is your FIRST-PERSON VIEW. You receive a high-level intent and must output 1-5 concrete actions to carry it out.

RULES:
- Interaction range is 0.5m. Move to within 0.5m BEFORE PickupObject/PutObject/etc.
- When MoveAhead BLOCKED: do NOT retry same direction and do NOT LookAround. Rotate 90deg and try there. If blocked in all directions, MoveBack.
- When target not visible: rotate to scan. Use direction hints in the object list.
- MoveAhead moves 0.25m. Count steps: 1.0m = 4 steps of MoveAhead, 2.0m = 8 steps.
- Copy objectType EXACTLY from the visible objects list. "Clock" is wrong; "AlarmClock" is correct.
- If the intent target is not visible yet, use Rotate/Move to find it.
- If the object list is empty: you are facing a wall or obstacle. DO NOT LookAround — Rotate or MoveBack.

ACTIONS:
  MoveAhead / MoveBack / MoveLeft / MoveRight (0.25m each)
  RotateLeft / RotateRight (90deg)
  LookUp / LookDown
  PickupObject(objectType)
  PutObject(objectType, receptacleType) — receptacleType from visible list, matching TASK TARGET
  OpenObject(objectType) / CloseObject(objectType)
  ToggleObjectOn(objectType) / ToggleObjectOff(objectType)
  SliceObject(objectType) / BreakObject(objectType)
  FillObjectWithLiquid(objectType) / EmptyLiquidFromObject(objectType)
  DropHandObject
  Done — only when ALL task completion criteria are met

OUTPUT — valid JSON only. { first char, } last char. No markdown. All text fields use first-person.

{
  "actions": [
    {"action": "MoveAhead", "params": {}},
    {"action": "PickupObject", "params": {"objectType": "AlarmClock"}}
  ],
  "status": "done" | "partial" | "failed",
  "reasoning": "scene: <what you see> | plan: <why these actions> | reflection: <verify assumptions>",
  "status_reason": "<1 sentence>"
}
- "done": intent fully achieved with these actions
- "partial": made progress, need another call with same intent
- "failed": intent cannot be achieved from current position"""


class ExecutorAgent:
    """Low-level action executor using 8B model."""

    def __init__(self, client: VLMClient):
        self.client = client

    def execute_intent(
        self,
        intent: str,
        target: str,
        image: np.ndarray,
        visible_objects: list[dict],
        action_history: list[dict],
        last_error: str | None,
        agent_pos: dict | None = None,
        agent_rot_y: float = 0.0,
        hand_status: str = "",
        task_criteria: str = "",
        memory_text: str = "",
        failed_object_ids: set = None,
        planner_feedback: str | None = None,
    ) -> dict:
        """Given an intent and full agent context, output action chunk."""
        from src.eb_agent import _direction, _append_task_context

        lines = [f"Your intent: {intent}"]
        if target:
            lines.append(f"Target object: {target}")
        lines.append("")

        # Spatial memory FIRST
        if memory_text:
            lines.append(memory_text)
            lines.append("")

        # Hand status + task criteria
        _append_task_context(lines, visible_objects, None, task_criteria)
        if hand_status:
            lines.insert(-2, f"HAND STATUS: {hand_status}")

        if planner_feedback:
            lines.append(f"\nPLANNER FEEDBACK: {planner_feedback}")

        if failed_object_ids is None:
            failed_object_ids = set()

        # Objects in view with direction labels
        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        receptacles = [o for o in visible if o.get("receptacle")]
        if receptacles:
            lines.append("Receptacles in view (for PutObject):")
            for o in receptacles:
                d = ""
                if agent_pos and o.get("position"):
                    d = " <- " + _direction(agent_pos, agent_rot_y, o["position"])
                lines.append(f"  {o['objectType']}{d}")
        if visible:
            lines.append("\nObjects in view — direction relative to your facing:")
            for o in visible[:12]:
                extra = []
                if o.get("isPickedUp"): extra.append("held")
                if o.get("receptacle"): extra.append("receptacle")
                if o.get("openable"): extra.append("openable" if not o.get("isOpen") else "open")
                if o.get("toggleable"): extra.append("on" if o.get("isToggled") else "off")
                tag = f" ({','.join(extra)})" if extra else ""
                prev = " [failed before]" if o.get("objectId") in failed_object_ids else ""
                d = ""
                if agent_pos and o.get("position"):
                    d = " <- " + _direction(agent_pos, agent_rot_y, o["position"])
                lines.append(f"  {o['objectType']}{tag}{prev}{d}")
        else:
            lines.append("(No objects in view — you may be facing a wall. Rotate or MoveBack. Do NOT LookAround.)")

        if failed_object_ids:
            lines.append("\nWARNING: these objectIds failed before. Do NOT propose them again:")
            for fid in list(failed_object_ids)[:5]:
                lines.append(f"  - {fid}")

        # Recent history for context
        recent = action_history[-5:] if len(action_history) > 5 else action_history
        if recent:
            lines.append("\nRecent actions:")
            lines.append(build_eb_history_context(recent))

        if last_error:
            lines.append(f"\nLast error: {last_error}")

        lines.append("\nOutput your action sequence. Be precise about distances.")
        prompt = "\n".join(lines)

        return self.client.chat_with_image_json(
            system_prompt=EXECUTOR_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("actions", "status", "reasoning", "status_reason"),
        )
