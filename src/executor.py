"""
Executor — takes a Planner intent + full agent context → outputs concrete action chunk.
Same agent identity as Planner (first-person). Uses 8B model.

Rich context: spatial memory, objects in view with directions, hand status,
task criteria, recent history, last error — same information the Planner sees.
"""
from __future__ import annotations
import numpy as np
from src.vlm_client import VLMClient
from src.context_builder import render_current_intent_steps

EXECUTOR_SYSTEM = """I am an embodied agent in a 3D household. The image is my FIRST-PERSON VIEW. I receive a high-level intent and must output 1-5 concrete actions to carry it out.

RULES:
- Output at most 12 actions. Use "repeat" to batch same-direction movement. If you output more than 12, the entire sequence will be rejected.
- Interaction range is 0.5m. Move to within 0.5m BEFORE PickupObject/PutObject/etc.
- Each movement step = 0.125m. BATCH same-direction moves with "repeat": {"action": "MoveAhead", "repeat": 8}. 0.6m = repeat 5. 1.0m = repeat 8. 1.8m = repeat 15. distance / 0.125, round up.
- IMPORTANT: For MoveRight/MoveLeft (strafing), use repeat ≤ 4. Sidesteps are riskier than forward movement — you can strafe into furniture. MoveRight×8 in a tight kitchen WILL fail. For lateral movement, move 2-4 steps at most, then reassess.
- Do NOT output individual MoveAhead entries repeatedly. Use "repeat" instead.
- SOLID OBJECTS: large furniture (Table, Desk, Bed, Cabinet, Dresser, Shelf, Sofa, ArmChair, Fridge, Counter, SideTable, CoffeeTable) CANNOT be walked through. You must go AROUND them. If a solid object is between you and your target, use MoveLeft/MoveRight to sidestep, or Rotate and find a clear path. Moving directly toward a solid object WILL fail.
- When MoveAhead BLOCKED: try MoveLeft, then MoveRight, then RotateLeft+MoveAhead, then RotateRight+MoveAhead. If ALL blocked, you are boxed in. MoveBack repeatedly (at LEAST 8 steps = 1.0m) to escape into open space, THEN rotate and find your target. Do NOT go back toward the obstacle.
- STUCK ESCAPE (CRITICAL): if the last 2+ attempts all hit obstacles, you are trapped. Output a PURE escape sequence: MoveBack×8 (1.0m minimum — 0.5m is NOT enough to clear furniture). After MoveBack, RotateLeft or RotateRight. That is ALL — do NOT add MoveAhead or any approach action after rotating. The next call will handle the approach. Do NOT sneak re-approach into the escape sequence.
- When target not visible: RotateLeft or RotateRight to find it. Use direction hints in the object list.
- Copy objectType EXACTLY from the visible objects list. "Clock" is wrong; "AlarmClock" is correct.
- If the intent target is not visible yet, use Rotate/Move to find it.
- If the object list is empty: you are facing a wall or obstacle. MoveBack to find open space, then re-orient.

ACTIONS:
  MoveAhead / MoveBack / MoveLeft / MoveRight (0.125m each, use "repeat": N to batch)
  RotateLeft / RotateRight (90deg)
  LookUp / LookDown (tilt camera by 30°; cameraHorizon is positive down: valid range -30° up to +60° down)
  PickupObject(objectType)
  PutObject(objectType, receptacleType) — receptacleType from visible list, matching TASK TARGET
  OpenObject(objectType) / CloseObject(objectType)
  ToggleObjectOn(objectType) / ToggleObjectOff(objectType)
  SliceObject(objectType) / BreakObject(objectType)
  FillObjectWithLiquid(objectType) / EmptyLiquidFromObject(objectType)
  DropHandObject

OUTPUT — valid JSON only. { first char, } last char. No markdown. All text fields use first-person.

{
  "actions": [
    {"action": "MoveAhead", "repeat": 8},
    {"action": "PickupObject", "params": {"objectType": "Knife"}}
  ],
  "status": "done" | "partial" | "failed",
  "reasoning": "scene: <what you see> | plan: <why these actions> | reflection: <verify assumptions>",
  "status_reason": "<1 sentence>"
}
- "done": intent fully achieved with these actions
- "partial": made progress, need another call with same intent
- "failed": intent cannot be achieved from current position
- "repeat" batches same-direction movement steps (MoveAhead/MoveBack/MoveLeft/MoveRight). Defaults to 1."""


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
        camera_horizon: float = 0.0,
        trail_text: str = "",
        planner_reasoning: str = "",
    ) -> dict:
        """Given an intent and full agent context, output action chunk."""
        from src.eb_agent import _direction, _direction_for_obj, _append_task_context

        # ── Current intent block (replaces old intent header + recent actions) ──
        current_intent_block = render_current_intent_steps(
            intent=intent,
            target=target,
            planner_reasoning=planner_reasoning,
            steps=action_history,
        )
        lines = [current_intent_block, ""]

        # Spatial memory FIRST
        if memory_text:
            lines.append(memory_text)
            lines.append("")

        # Trail summary (Spike 006: search-trail-cost)
        if trail_text:
            lines.append(trail_text)
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
                    d = " <- " + _direction_for_obj(agent_pos, agent_rot_y, o)
                lines.append(f"  {o['objectType']}{d}")
        if visible:
            lines.append("\nObjects in view — direction relative to your facing:")
            # Highlight solid obstacles
            _SOLID_TYPES = {"Desk", "DiningTable", "SideTable", "CoffeeTable", "Bed", "Cabinet",
                            "Dresser", "Shelf", "Sofa", "ArmChair", "Fridge", "Counter", "CounterTop",
                            "Stool", "Chair", "Ottoman", "Bench"}
            for o in visible[:12]:
                extra = []
                if o.get("isPickedUp"): extra.append("held")
                if o.get("receptacle"): extra.append("receptacle")
                if o.get("openable"): extra.append("openable" if not o.get("isOpen") else "open")
                if o.get("toggleable"): extra.append("on" if o.get("isToggled") else "off")
                if o.get("objectType") in _SOLID_TYPES and not o.get("isPickedUp"):
                    extra.append("SOLID - do NOT walk through")
                tag = f" ({','.join(extra)})" if extra else ""
                prev = " [failed before]" if o.get("objectId") in failed_object_ids else ""
                d = ""
                if agent_pos and o.get("position"):
                    d = " <- " + _direction_for_obj(agent_pos, agent_rot_y, o)
                lines.append(f"  {o['objectType']}{tag}{prev}{d}")
        else:
            lines.append("(No objects in view — you may be facing a wall. Rotate or MoveBack.)")

        if failed_object_ids:
            lines.append("\nWARNING: these objectIds failed before. Do NOT propose them again:")
            for fid in list(failed_object_ids)[:5]:
                lines.append(f"  - {fid}")

        # (Step history is already shown via render_current_intent_steps above)

        if last_error:
            lines.append(f"\nLast error: {last_error}")

        normalized_horizon = camera_horizon - 360 if camera_horizon > 180 else camera_horizon
        lines.append(
            f"\nCAMERA: cameraHorizon = {normalized_horizon:.0f}° "
            f"(0°=level; positive=down; valid up=-30°, down=+60°; "
            f"remaining LookUp={normalized_horizon + 30:.0f}°, "
            f"LookDown={60 - normalized_horizon:.0f}°). "
            "Do not output LookUp or LookDown beyond these bounds."
        )

        lines.append(f"\nGRID: 1 step = 0.125m. Use \\\"repeat\\\" to batch: distance / 0.125 = repeat count. E.g. 1.8m away → \\\"action\\\": \\\"MoveAhead\\\", \\\"repeat\\\": 15.")
        lines.append("Output your action sequence. USE REPEAT. Do NOT output individual steps.")
        prompt = "\n".join(lines)

        result = self.client.chat_with_image_json(
            system_prompt=EXECUTOR_SYSTEM,
            user_text=prompt,
            image=image,
            required_fields=("actions", "status", "reasoning", "status_reason"),
            max_tokens=2048,
        )
        return result
