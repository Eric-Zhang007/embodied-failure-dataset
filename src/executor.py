"""
Executor Agent — translates a Planner intent into a short action chunk.

Operates in a tight loop: receives an intent, outputs 1-5 concrete actions,
executes them, reports status back to Planner. Does NOT know the task goal —
only the current intent.
"""

from __future__ import annotations

import numpy as np

from src.vlm_client import VLMClient

EXECUTOR_SYSTEM = """You are an action executor. Given a first-person view and a single intent, output 1-5 concrete actions to achieve it.

RULES:
- Interaction range is 0.5m. Check distance before PickupObject/PutObject/etc.
- When MoveAhead is BLOCKED: do NOT retry same direction. Rotate or MoveBack.
- Never interact with objects beyond 0.5m.
- Copy objectType EXACTLY from the visible objects list.

ACTIONS:
  MoveAhead / MoveBack / MoveLeft / MoveRight  (0.25m each)
  RotateLeft / RotateRight  (90°)
  LookUp / LookDown
  PickupObject(objectType)
  PutObject(objectType, receptacleType)  — receptacleType from intent
  OpenObject(objectType) / CloseObject(objectType)
  ToggleObjectOn(objectType) / ToggleObjectOff(objectType)
  DropHandObject
  Done

OUTPUT — valid JSON only. { first char, } last char.
{
  "actions": [
    {"action": "MoveAhead", "params": {}},
    {"action": "PickupObject", "params": {"objectType": "AlarmClock"}}
  ],
  "status": "done" | "partial" | "failed",
  "status_reason": "<1 sentence>"
}
- "done": intent fully achieved
- "partial": made progress but need more actions (will be called again with same intent)
- "failed": intent cannot be achieved from current position (Planner will replan)"""


class ExecutorAgent:
    """Translates intents into short action chunks."""

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
    ) -> dict:
        """Given an intent and current view, output an action chunk.

        Returns: {"actions": [...], "status": "done"|"partial"|"failed", "status_reason": "..."}
        """
        from src.eb_agent import _direction

        # Build compact user prompt
        lines = [f"Intent: {intent}"]
        if target:
            lines.append(f"Target object: {target}")
        lines.append("")

        # Visible receptacles
        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        receptacles = [o for o in visible if o.get("receptacle")]
        if receptacles:
            lines.append("Receptacles in view:")
            for o in receptacles:
                d = ""
                if agent_pos and o.get("position"):
                    d = " <- " + _direction(agent_pos, agent_rot_y, o["position"])
                lines.append(f"  {o['objectType']}{d}")

        # All visible objects
        if visible:
            lines.append("\nVisible objects:")
            for o in visible[:15]:
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

        prompt = "\n".join(lines)

        result = self.client.chat_with_image_json(
            system_prompt=EXECUTOR_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("actions", "status", "status_reason"),
        )
        return result
