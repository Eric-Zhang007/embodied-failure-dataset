"""
Executor — takes a Planner intent + current view → outputs concrete action chunk.
Same agent identity as Planner (first-person). Uses 8B model for fast per-step execution.

Review loop: Executor proposes actions → Planner reviews → approve or reject with feedback.
"""
from __future__ import annotations
import numpy as np
from src.vlm_client import VLMClient

EXECUTOR_SYSTEM = """You are an embodied agent in a 3D household. You receive a high-level intent and must output 1-5 concrete actions to carry it out.

RULES:
- Interaction range is 0.5m. Move to within 0.5m BEFORE PickupObject/PutObject/etc.
- When MoveAhead BLOCKED: try MoveLeft/Right, then Rotate, then MoveBack. Do NOT retry same blocked direction.
- MoveAhead moves 0.25m. Count steps: 1.0m = 4 steps of MoveAhead.
- Copy objectType EXACTLY from the visible objects list.
- If the intent target is not visible yet, use Rotate/Move to find it.

ACTIONS:
  MoveAhead / MoveBack / MoveLeft / MoveRight (0.25m each)
  RotateLeft / RotateRight (90deg)
  LookUp / LookDown
  PickupObject(objectType)
  PutObject(objectType, receptacleType) — receptacleType from visible list
  OpenObject(objectType) / CloseObject(objectType)
  ToggleObjectOn(objectType) / ToggleObjectOff(objectType)
  SliceObject(objectType) / BreakObject(objectType)
  FillObjectWithLiquid(objectType) / EmptyLiquidFromObject(objectType)
  DropHandObject
  Done

OUTPUT — valid JSON only:
- { must be the FIRST character, } must be the LAST character.
- NO text outside braces. NO markdown.

{
  "actions": [
    {"action": "MoveAhead", "params": {}},
    {"action": "PickupObject", "params": {"objectType": "AlarmClock"}}
  ],
  "status": "done" | "partial" | "failed",
  "reasoning": "<first-person reasoning: what you see and why these actions>",
  "status_reason": "<1 sentence: why done/partial/failed>"
}
- "done": intent fully achieved with these actions
- "partial": made progress, need another call with same intent
- "failed": intent cannot be achieved from current position (Planner will replan)"""


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
        agent_pos: dict | None = None,
        agent_rot_y: float = 0.0,
        hand_status: str = "",
        planner_feedback: str | None = None,
        max_retries: int = 2,
    ) -> dict:
        """Given an intent and current view, output action chunk.

        planner_feedback: if Planner rejected a previous attempt, this is the reason.
        """
        from src.eb_agent import _direction

        lines = [f"Your intent: {intent}"]
        if target:
            lines.append(f"Target: {target}")
        if hand_status:
            lines.append(f"HAND STATUS: {hand_status}")
        lines.append("")

        if planner_feedback:
            lines.append(f"PLANNER REJECTED YOUR PREVIOUS ACTIONS: {planner_feedback}")
            lines.append("Rewrite your actions based on this feedback.\n")

        # Visible objects
        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        if visible:
            lines.append("Objects in view — direction relative to your facing:")
            for o in visible[:12]:
                extra = []
                if o.get("isPickedUp"):
                    extra.append("held")
                if o.get("receptacle"):
                    extra.append("receptacle")
                if o.get("openable"):
                    extra.append("openable" if not o.get("isOpen") else "open")
                tag = f" ({','.join(extra)})" if extra else ""
                d = ""
                if agent_pos and o.get("position"):
                    d = " <- " + _direction(agent_pos, agent_rot_y, o["position"])
                lines.append(f"  {o['objectType']}{tag}{d}")
        else:
            lines.append("(No objects in view — you may be facing a wall. Rotate or MoveBack.)")

        lines.append("\nOutput your action sequence. Be concise.")
        prompt = "\n".join(lines)

        return self.client.chat_with_image_json(
            system_prompt=EXECUTOR_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("actions", "status", "reasoning", "status_reason"),
            max_retries=max_retries,
        )
