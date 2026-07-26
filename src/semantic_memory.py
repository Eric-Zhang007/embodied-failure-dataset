"""
Semantic Egocentric Memory.

Receptacle-grouped, freshness-aware memory with task progress tracking.
Renders first-person natural language ("I see...", "I remember...").

Distinguishes itself from GeometricMemory by:
  - Grouping objects by their parent receptacle (where they are)
  - Tracking freshness (how recently confirmed)
  - Maintaining a task progress checklist
  - Disambiguating similar-looking visible objects from remembered targets
  - Using first-person "I" perspective consistently
"""
from __future__ import annotations

import math, re
from copy import deepcopy
from collections import OrderedDict
from dataclasses import asdict

from src.memory_interface import MemoryInterface
from src.geometric_memory import (
    StuckTracker,
    _egocentric_direction,
    _ObjectEntry,
    _ObstacleEntry,
)


class SemanticMemory(MemoryInterface):
    AGING_THRESHOLD: int = 40  # longer than geometric — semantics need more history
    BLOCK_THRESHOLD: int = 2
    FRESH_THRESHOLD: int = 5   # steps within which an observation is "fresh"

    def __init__(self):
        self._objects: dict[str, _ObjectEntry] = {}
        self._obstacles: list[_ObstacleEntry] = []
        self._step_counter: int = 0
        self._last_error: str | None = None
        self._agent_x: float = 0.0
        self._agent_z: float = 0.0
        self._agent_rot_y: float = 0.0
        self._area_label: str = ""
        self._stuck_tracker = StuckTracker()
        self._task_type: str = ""
        self._task_target: str = ""
        self._task_receptacle: str = ""

        # Visitation stats
        self.receptacle_visit_counts: dict[str, int] = {}
        self.receptacle_open_counts: dict[str, int] = {}
        self.objects_found_by_receptacle: dict[str, int] = {}

        # State tracking per objectId
        self._object_state: dict[str, dict] = {}  # objectId → {is_open, is_toggled, temperature, ...}

        # Searched markers (type → count)
        self._searched_types: dict[str, int] = {}
        self._state_change_callback = None

    # ── core update ──────────────────────────────────────────────────

    def export_state(self) -> dict:
        """Return the complete semantic state in a JSON-safe form."""
        tracker = self._stuck_tracker
        return {
            "objects": {oid: asdict(entry) for oid, entry in self._objects.items()},
            "obstacles": [asdict(obstacle) for obstacle in self._obstacles],
            "step_counter": self._step_counter,
            "last_error": self._last_error,
            "agent": {"x": self._agent_x, "z": self._agent_z, "rot_y": self._agent_rot_y},
            "area_label": self._area_label,
            "task": {
                "type": self._task_type,
                "target": self._task_target,
                "receptacle": self._task_receptacle,
            },
            "receptacle_visit_counts": dict(self.receptacle_visit_counts),
            "receptacle_open_counts": dict(self.receptacle_open_counts),
            "objects_found_by_receptacle": dict(self.objects_found_by_receptacle),
            "object_state": deepcopy(self._object_state),
            "searched_types": dict(self._searched_types),
            "stuck_tracker": {
                "intent_failure": self._pack_tuple_map(tracker._intent_failure),
                "intent_last_step": self._pack_tuple_map(tracker._intent_last_step),
                "consecutive_failures": tracker._consecutive_failures,
                "blocked_directions": self._pack_tuple_map(tracker._blocked_directions),
                "block_dir_last_step": self._pack_tuple_map(tracker._block_dir_last_step),
                "obstacle_intent_blocks": self._pack_tuple_map(tracker._obstacle_intent_blocks),
                "obs_last_step": self._pack_tuple_map(tracker._obs_last_step),
            },
        }

    @classmethod
    def from_state(cls, state: dict) -> "SemanticMemory":
        """Restore a memory instance previously produced by export_state()."""
        memory = cls()
        memory._objects = {
            oid: _ObjectEntry(**entry)
            for oid, entry in state.get("objects", {}).items()
        }
        memory._obstacles = [
            _ObstacleEntry(**obstacle)
            for obstacle in state.get("obstacles", [])
        ]
        memory._step_counter = state.get("step_counter", 0)
        memory._last_error = state.get("last_error")
        agent = state.get("agent", {})
        memory._agent_x = agent.get("x", 0.0)
        memory._agent_z = agent.get("z", 0.0)
        memory._agent_rot_y = agent.get("rot_y", 0.0)
        memory._area_label = state.get("area_label", "")
        task = state.get("task", {})
        memory._task_type = task.get("type", "")
        memory._task_target = task.get("target", "")
        memory._task_receptacle = task.get("receptacle", "")
        memory.receptacle_visit_counts = dict(state.get("receptacle_visit_counts", {}))
        memory.receptacle_open_counts = dict(state.get("receptacle_open_counts", {}))
        memory.objects_found_by_receptacle = dict(state.get("objects_found_by_receptacle", {}))
        memory._object_state = deepcopy(state.get("object_state", {}))
        memory._searched_types = dict(state.get("searched_types", {}))

        tracker_state = state.get("stuck_tracker", {})
        tracker = memory._stuck_tracker
        tracker._intent_failure = cls._unpack_tuple_map(tracker_state.get("intent_failure", []))
        tracker._intent_last_step = cls._unpack_tuple_map(tracker_state.get("intent_last_step", []))
        tracker._consecutive_failures = tracker_state.get("consecutive_failures", 0)
        tracker._blocked_directions = cls._unpack_tuple_map(tracker_state.get("blocked_directions", []))
        tracker._block_dir_last_step = cls._unpack_tuple_map(tracker_state.get("block_dir_last_step", []))
        tracker._obstacle_intent_blocks = cls._unpack_tuple_map(tracker_state.get("obstacle_intent_blocks", []))
        tracker._obs_last_step = cls._unpack_tuple_map(tracker_state.get("obs_last_step", []))
        return memory

    def set_state_change_callback(self, callback) -> None:
        self._state_change_callback = callback

    def _state_changed(self) -> None:
        if self._state_change_callback is not None:
            self._state_change_callback(self.export_state())

    @staticmethod
    def _pack_tuple_map(values: dict[tuple, int]) -> list[dict]:
        return [{"key": list(key), "value": value} for key, value in values.items()]

    @staticmethod
    def _unpack_tuple_map(values: list[dict]) -> dict[tuple, int]:
        return {tuple(item["key"]): item["value"] for item in values}

    def update(
        self,
        metadata: dict,
        visible_objects: list[dict],
        action: str,
        success: bool,
        error_message: str | None,
        task_criteria: str = "",
        intent: str = "",
    ) -> None:
        self._step_counter += 1

        agent = metadata.get("agent", {})
        self._agent_x = agent.get("position", {}).get("x", 0.0)
        self._agent_z = agent.get("position", {}).get("z", 0.0)
        self._agent_rot_y = agent.get("rotation", {}).get("y", 0.0)

        self._extract_task_info(task_criteria)
        inventory = metadata.get("inventoryObjects") or []
        held_ids = {o.get("objectId") for o in inventory if o.get("objectId")}
        # AI2-THOR inventory entries include objectId. Retain a type-based
        # fallback only for incomplete metadata from older recordings.
        held_types_without_id = {
            o.get("objectType", "")
            for o in inventory
            if o.get("objectType") and not o.get("objectId")
        }

        held_before_action = (
            {oid for oid, entry in self._objects.items() if entry.status == "held"}
            if action == "PutObject" and success else set()
        )
        current_ids: set[str] = set()
        for obj in visible_objects:
            oid = obj.get("objectId", "")
            if not oid:
                continue
            otype = obj.get("objectType", "")
            if not otype:
                continue
            if not obj.get("visibleBounds2D"):
                continue

            current_ids.add(oid)
            pos = obj.get("position", {})
            bbox = obj.get("axisAlignedBoundingBox")
            if bbox and bbox.get("cornerPoints"):
                corners = bbox["cornerPoints"]
                xs = [p[0] for p in corners]
                zs = [p[2] for p in corners]
                ox = max(min(xs), min(max(xs), self._agent_x))
                oz = max(min(zs), min(max(zs), self._agent_z))
            else:
                ox, oz = pos.get("x", 0.0), pos.get("z", 0.0)
            direction, dist = _egocentric_direction(
                self._agent_x, self._agent_z, self._agent_rot_y, ox, oz,
            )
            is_recep = obj.get("receptacle", False)
            is_task_recep = is_recep and otype == self._task_receptacle
            is_held = oid in held_ids or otype in held_types_without_id
            parent_id = (obj.get("parentReceptacles") or [None])[0]

            status = "held" if is_held else "visible"

            if oid in self._objects:
                e = self._objects[oid]
                e.egocentric_dir = direction
                e.egocentric_dist = dist
                e.status = status
                e.last_seen_step = self._step_counter
                e.seen_count += 1
                e.is_receptacle = is_recep
                e.is_task_receptacle = is_task_recep
                e.parent_receptacle_id = parent_id
            else:
                self._objects[oid] = _ObjectEntry(
                    object_id=oid, object_type=otype,
                    is_receptacle=is_recep, is_task_receptacle=is_task_recep,
                    status=status, egocentric_dir=direction, egocentric_dist=dist,
                    last_seen_step=self._step_counter, seen_count=1,
                    parent_receptacle_id=parent_id,
                )
                self._record_discovery(oid, obj, visible_objects)

            # Track object state
            self._track_object_state(oid, otype, obj)

        for oid, entry in self._objects.items():
            if oid not in current_ids and entry.status == "visible":
                entry.status = "remembered"

        if action == "PutObject" and success:
            for oid in held_before_action:
                entry = self._objects.get(oid)
                if entry:
                    entry.status = "placed"
                    entry.last_seen_step = self._step_counter
        else:
            for held_id in held_ids:
                entry = self._objects.get(held_id)
                if entry:
                    entry.status = "held"
                    entry.last_seen_step = self._step_counter
            for held_type in held_types_without_id:
                for entry in self._objects.values():
                    if entry.object_type == held_type:
                        entry.status = "held"
                        entry.last_seen_step = self._step_counter
            for oid, entry in self._objects.items():
                if entry.status == "held" and oid not in held_ids and oid not in current_ids:
                    entry.status = "remembered"

        blocker = None
        blocker_id = None
        blocker_dir = None
        if not success and error_message:
            blocker = self._parse_blocker(error_message)
            blocker_id = self._parse_blocker_id(error_message)
            if blocker:
                obs = self._find_or_create_obstacle(blocker)
                obs.block_count += 1
                obs.last_block_step = self._step_counter
                blocker_dir = obs.direction
            self._last_error = self._compress_error(action, error_message)
        elif success:
            self._last_error = None

        intent_parts = intent.split(maxsplit=1)
        intent_verb = intent_parts[0] if intent_parts else ""
        intent_target = intent_parts[1] if len(intent_parts) > 1 else ""
        self._stuck_tracker.update(
            step=self._step_counter, intent=intent_verb, target=intent_target,
            success=success, blocked_by=blocker_id or blocker, blocked_dir=blocker_dir,
            action=action,
        )

        self._age_entries()
        if self._step_counter % 5 == 0 or not self._area_label:
            self._update_area_label(metadata)
        self._state_changed()

    # ── render ───────────────────────────────────────────────────────

    def render(self) -> str:
        sections: list[str] = []

        # Section 0: Situation summary (stuck warnings)
        situation = self._stuck_tracker.render_summary(self._step_counter)
        if situation:
            sections.append(situation)

        # Section 1: Task state
        sections.append(self._render_task_state())

        # Section 2: Visible objects — grouped by receptacle
        sections.append(self._render_visible_grouped())

        # Section 3: Remembered objects — where my target is
        sections.append(self._render_remembered_target_info())

        # Section 4: Remembered objects — grouped by receptacle
        sections.append(self._render_remembered_grouped())

        # Section 5: What I've already checked
        sections.append(self._render_checked())

        # Section 6: Obstacles
        sections.append(self._render_obstacles())

        # Section 7: Last error
        if self._last_error:
            sections.append(f"LAST ERROR: {self._last_error}")

        result = "\n".join(s for s in sections if s)
        if len(result) > 2000:
            result = result[:1997] + "..."
        return result

    def _render_task_state(self) -> str:
        lines = ["=== TASK ==="]
        hand_items = [e for e in self._objects.values() if e.status == "held"]
        hand_str = ", ".join(e.object_type for e in hand_items) if hand_items else "nothing"
        lines.append(f"I am holding {hand_str}.")

        if self._task_target:
            phases = self._detect_phase()
            lines.append(f"Goal: {self._task_type or 'complete'} the {self._task_target} → {self._task_receptacle}")
            lines.append(f"Phase: {phases['phase']}")

            # Progress checklist
            target = self._task_target
            checks = []
            checks.append(("Find " + target, self.has_type(target)))
            checks.append(("Pick up", any(e.object_type == target and e.status == "held"
                                          for e in self._objects.values())))
            checks.append(("Heat/Cool/Clean" if self._task_type else "Process",
                          False))  # tracked via object state
            checks.append(("Place in " + self._task_receptacle,
                          any(e.object_type == target and e.status == "placed"
                              for e in self._objects.values())))
            parts = []
            for label, done in checks:
                parts.append(f"{'✓' if done else '□'} {label}")
            lines.append("Progress: " + "  ".join(parts))

        return "\n".join(lines) + "\n"

    def _render_visible_grouped(self) -> str:
        visible = [e for e in self._objects.values() if e.status == "visible"]
        if not visible:
            return "=== WHAT I SEE ===\n  (nothing visible right now — I may be facing a wall)\n"

        # Group by parent receptacle
        groups = self._group_by_receptacle(visible)
        lines = ["=== WHAT I SEE ==="]
        for parent_label, entries in groups.items():
            lines.append(f"\nOn {parent_label}:")
            for e in entries:
                line = self._render_entry(e)
                # Disambiguation: if target is remembered and a similar type is visible
                if (self._task_target and e.object_type != self._task_target
                        and self._looks_similar(e.object_type, self._task_target)
                        and self.has_type(self._task_target)
                        and not self.is_type_visible(self._task_target)):
                    line += f"\n     ⚠ This is NOT the {self._task_target} — I last saw it elsewhere!"
                lines.append(line)
        return "\n".join(lines) + "\n"

    def _render_remembered_target_info(self) -> str:
        if not self._task_target:
            return ""
        target_entries = [e for e in self._objects.values()
                          if e.object_type == self._task_target
                          and e.status == "remembered"]
        if not target_entries:
            if self.has_type(self._task_target):
                return ""  # target is visible — no need for remembered info
            return ""

        lines = ["=== WHERE MY TARGET IS ==="]
        for e in target_entries:
            age = self._step_counter - e.last_seen_step
            freshness = "fresh" if age <= self.FRESH_THRESHOLD else (
                f"{age} steps ago" if age < 30 else "a long time ago"
            )
            parent = self._lookup_parent_label(e.parent_receptacle_id)
            loc = f"at {parent}" if parent else f"{e.egocentric_dir}, ~{e.egocentric_dist:.1f}m"
            lines.append(f"  ▸ {e.object_type} — last seen {freshness} {loc}.")
            if parent and not self._is_receptacle_visible(parent):
                lines.append(f"    I should re-locate it around {parent} before interacting.")

        # ── Failure saturation: if I've failed to interact with the target at
        #     its last known location repeatedly, append a confidence-lowering hint.
        failures_at_location = 0
        for key, count in self._stuck_tracker._intent_failure.items():
            intent_verb, intent_target = key
            if intent_target == self._task_target and intent_verb in ("pickup", "approach", "open"):
                failures_at_location = max(failures_at_location, count)
        if failures_at_location >= 3:
            lines.append(f"\n  ⚡ WARNING: I've failed to reach the {self._task_target} here "
                         f"{failures_at_location} times. I may be stuck at a bad angle — "
                         f"try approaching from a different direction or checking other "
                         f"locations where the {self._task_target} might also be.")
        return "\n".join(lines) + "\n"

    def _render_remembered_grouped(self) -> str:
        remembered = [e for e in self._objects.values()
                      if e.status == "remembered"
                      and e.object_type != self._task_target]  # target shown above
        if not remembered:
            return ""

        groups = self._group_by_receptacle(remembered)
        lines = ["=== WHAT I REMEMBER ==="]
        for parent_label, entries in groups.items():
            lines.append(f"\nIn/on {parent_label}:")
            for e in entries:
                lines.append(self._render_entry(e))
        return "\n".join(lines) + "\n"

    def _render_checked(self) -> str:
        lines = ["=== I HAVE ALREADY CHECKED ==="]
        has_any = False

        # Show searched receptacles (from markers)
        for rectype, _ in sorted(self._searched_types.items()):
            has_any = True
            # Check if the target was found here
            found_here = any(
                e.object_type == self._task_target and
                e.parent_receptacle_id and
                self._lookup_parent_type(e.parent_receptacle_id) == rectype
                for e in self._objects.values()
            )
            if found_here:
                lines.append(f"  ✓ {rectype} — my target was here!")
            else:
                lines.append(f"  ✗ {rectype} — no target found here")

        # Show receptacles that were visited but not marked searched
        for rectype, count in self.receptacle_visit_counts.items():
            if rectype not in self._searched_types and count >= 2:
                has_any = True
                lines.append(f"  ↻ {rectype} — visited {count}×, should check more thoroughly")

        if not has_any:
            lines.append("  (nothing checked yet — I should start exploring)")
        return "\n".join(lines) + "\n"

    def _render_obstacles(self) -> str:
        active = [o for o in self._obstacles
                  if o.block_count >= self.BLOCK_THRESHOLD
                  and (self._step_counter - o.last_block_step) <= 5]
        if not active:
            return ""
        lines = ["=== OBSTACLES ==="]
        for o in active:
            suggestion = self._obstacle_suggestion(o)
            lines.append(f"  {o.object_type} {o.direction} blocked me {o.block_count}×. {suggestion}")
        return "\n".join(lines) + "\n"

    # ── entry rendering ──────────────────────────────────────────────

    def _render_entry(self, e: _ObjectEntry) -> str:
        if e.status == "held":
            return f"  ▸ {e.object_type} — I am holding this"
        elif e.status == "placed":
            return f"  ▸ {e.object_type} — I placed this (hand free)"
        elif e.status == "visible":
            tags = []
            if e.is_task_receptacle:
                tags.append("TASK RECEPTACLE")
            elif e.is_receptacle:
                tags.append("receptacle")
            if e.object_type == self._task_target:
                tags.append("★ MY TARGET")
            if e.searched:
                tags.append("searched")
            tag_str = f" ({', '.join(tags)})" if tags else ""
            return f"  ▸ {e.object_type} — {e.egocentric_dir}, {e.egocentric_dist:.1f}m{tag_str}"
        else:
            age = self._step_counter - e.last_seen_step
            freshness = "recently" if age <= self.FRESH_THRESHOLD else f"{age} steps ago"
            tags = []
            if e.searched:
                tags.append("searched")
            if e.object_type == self._task_target:
                tags.append("★ MY TARGET")
            tag_str = f" ({', '.join(tags)})" if tags else ""
            return f"  ▸ {e.object_type} — saw {e.egocentric_dir}, ~{e.egocentric_dist:.1f}m, {freshness}{tag_str}"

    # ── receptacle grouping ──────────────────────────────────────────

    def _group_by_receptacle(self, entries: list[_ObjectEntry]) -> OrderedDict[str, list[_ObjectEntry]]:
        """Group entries by parent receptacle label, keeping insertion order."""
        groups: OrderedDict[str, list[_ObjectEntry]] = OrderedDict()
        unparented: list[_ObjectEntry] = []

        for e in entries:
            parent_label = self._lookup_parent_label(e.parent_receptacle_id)
            if parent_label:
                groups.setdefault(parent_label, []).append(e)
            else:
                unparented.append(e)

        # Add unparented at the end, grouped as "Nearby"
        if unparented:
            groups["nearby (no parent)"] = unparented

        return groups

    def _lookup_parent_label(self, parent_id: str | None) -> str | None:
        if not parent_id:
            return None
        entry = self._objects.get(parent_id)
        if entry:
            return entry.object_type
        return None

    def _lookup_parent_type(self, parent_id: str | None) -> str | None:
        if not parent_id:
            return None
        entry = self._objects.get(parent_id)
        return entry.object_type if entry else None

    def _is_receptacle_visible(self, rectype: str) -> bool:
        return any(e.object_type == rectype and e.status == "visible" and e.is_receptacle
                   for e in self._objects.values())

    def _looks_similar(self, type_a: str, type_b: str) -> bool:
        """Heuristic: could these two types be visually confused?"""
        similar_pairs = {
            ("Tomato", "Apple"), ("Apple", "Tomato"),
            ("Potato", "Apple"), ("Apple", "Potato"),
            ("Egg", "Potato"), ("Potato", "Egg"),
            ("Bread", "Potato"), ("Potato", "Bread"),
            ("Lettuce", "Apple"), ("Apple", "Lettuce"),
        }
        return (type_a, type_b) in similar_pairs

    # ── state tracking ───────────────────────────────────────────────

    def _track_object_state(self, oid: str, otype: str, obj: dict):
        state = self._object_state.setdefault(oid, {})
        if "isOpen" in obj:
            state["is_open"] = obj["isOpen"]
        if "isToggled" in obj:
            state["is_toggled"] = obj["isToggled"]
        if "isBroken" in obj:
            state["is_broken"] = obj["isBroken"]
        if "temperature" in obj:
            state["temperature"] = obj["temperature"]
        state["pickupable"] = obj.get("pickupable", False)

    def _record_discovery(self, oid: str, obj: dict, visible_objects: list[dict]):
        parent_ids = obj.get("parentReceptacles") or []
        if not parent_ids:
            return
        parent_oid = parent_ids[0]
        parent_type = None
        entry = self._objects.get(parent_oid)
        if entry and entry.is_receptacle:
            parent_type = entry.object_type
        else:
            for vo in visible_objects:
                if vo.get("objectId") == parent_oid and vo.get("receptacle"):
                    parent_type = vo.get("objectType")
                    break
        if parent_type:
            self.objects_found_by_receptacle[parent_type] = \
                self.objects_found_by_receptacle.get(parent_type, 0) + 1

    # ── task info extraction ─────────────────────────────────────────

    def _extract_task_info(self, criteria: str):
        if not criteria or self._task_target:
            return
        # Parse patterns like:
        # "Pick up the Apple and put it inside Fridge → CounterTop" (heat/cool)
        # "Pick up the SoapBottle and put it on the Toilet."
        # "Heat the Apple and put it inside CounterTop."
        task_type_map = {
            "heat": "Heat", "cool": "Cool", "clean": "Clean",
            "pick_and_place": "", "pick": "",
        }
        lower = criteria.lower()
        for key, val in task_type_map.items():
            if key in lower:
                self._task_type = val
                break

        # Extract target object: "Heat the Apple" → "Apple"
        m = re.search(r'(?:Heat|Cool|Clean|Pick up)\s+(?:the\s+)?(\w+)', criteria, re.IGNORECASE)
        if m:
            self._task_target = m.group(1)
        else:
            # BranchRunner passes completion criteria such as
            # "  - Apple must be inside Cabinet", not natural-language goals.
            m = re.search(
                r'(?:^|\n)\s*(?:[-*]\s*)?(?:Two\s+)?(\w+)\s+must be\b',
                criteria,
                re.IGNORECASE,
            )
            if m:
                self._task_target = m.group(1)

        # Extract receptacle
        inside_targets = re.findall(r'\binside\s+(?:the\s+)?(\w+)', criteria, re.IGNORECASE)
        on_targets = re.findall(r'\bon\s+(?:the\s+)?(\w+)', criteria, re.IGNORECASE)
        if inside_targets:
            self._task_receptacle = inside_targets[-1]
        elif on_targets and "put it" in lower:
            self._task_receptacle = on_targets[-1]

        if self._task_type and not self._task_target:
            # Bare task: "heat the apple"
            pass

    # ── phase detection ──────────────────────────────────────────────

    def _detect_phase(self) -> dict:
        target = self._task_target
        if not target:
            return {"phase": "EXPLORATION", "reason": "no target identified"}

        held = any(e.object_type == target and e.status == "held"
                   for e in self._objects.values())

        if held:
            return {"phase": "DELIVERY",
                    "reason": f"I am holding the {target}; I need to process and deliver it"}

        # ── Failure-aware phase override: if I've tried to interact with the target
        #     at its current location too many times, force EXPLORATION so I don't
        #     keep retrying a dead-end approach.
        failures_at_target = 0
        for key, count in self._stuck_tracker._intent_failure.items():
            verb, tgt = key
            if tgt == target and verb in ("pickup", "approach", "open", "put"):
                failures_at_target = max(failures_at_target, count)
        if failures_at_target >= 4:
            return {"phase": "EXPLORATION",
                    "reason": (f"I've failed to reach the {target} {failures_at_target} times "
                               "at its last known location — I should explore alternatives.")}

        if self.has_type(target):
            if self.is_type_visible(target):
                dist = self.get_type_distance(target)
                if dist is not None and dist <= 0.5:
                    return {"phase": "INTERACTION", "reason": f"{target} is within reach"}
                return {"phase": "APPROACH", "reason": f"{target} is visible but >0.5m away"}
            return {"phase": "SEARCH",
                    "reason": f"I have seen {target} before but it is not visible now"}

        return {"phase": "EXPLORATION", "reason": f"I have never seen the {target}"}

    # ── queries (shared API) ─────────────────────────────────────────

    def has_type(self, object_type: str) -> bool:
        return any(e.object_type == object_type for e in self._objects.values())

    def is_type_visible(self, object_type: str) -> bool:
        return any(e.object_type == object_type and e.status == "visible"
                   for e in self._objects.values())

    def get_type_distance(self, object_type: str) -> float | None:
        best = None
        for e in self._objects.values():
            if e.object_type == object_type and e.status == "visible":
                if best is None or e.egocentric_dist < best:
                    best = e.egocentric_dist
        return best

    def get_all_object_types(self) -> set[str]:
        return {e.object_type for e in self._objects.values()}

    def get_visible_receptacles(self) -> list[str]:
        candidates: list[tuple[str, float]] = []
        for e in self._objects.values():
            if e.is_receptacle and e.status == "visible":
                candidates.append((e.object_type, e.egocentric_dist))
        candidates.sort(key=lambda x: x[1])
        seen = set()
        return [ot for ot, _ in candidates if not (ot in seen or seen.add(ot))]

    def get_remembered_receptacles(self) -> list[str]:
        candidates: list[tuple[str, int]] = []
        for e in self._objects.values():
            if e.is_receptacle and e.status == "remembered":
                candidates.append((e.object_type, e.last_seen_step))
        candidates.sort(key=lambda x: -x[1])
        seen = set()
        return [ot for ot, _ in candidates if not (ot in seen or seen.add(ot))]

    def get_unvisited_receptacles(self, approached_types: set[str] | None = None) -> list[str]:
        approached = approached_types or set()
        remembered: list[tuple[str, int]] = []
        visible: list[tuple[str, float]] = []
        for e in self._objects.values():
            if not e.is_receptacle or e.object_type in approached:
                continue
            if e.status == "remembered":
                remembered.append((e.object_type, e.last_seen_step))
            elif e.status == "visible":
                visible.append((e.object_type, e.egocentric_dist))
        remembered.sort(key=lambda x: -x[1])
        visible.sort(key=lambda x: x[1])
        seen = set()
        result = []
        for ot, _ in remembered:
            if ot not in seen:
                seen.add(ot)
                result.append(ot)
        for ot, _ in visible:
            if ot not in seen:
                seen.add(ot)
                result.append(ot)
        return result

    def get_searched_receptacle_types(self) -> set[str]:
        return set(self._searched_types.keys())

    def get_type_distance_any(self, object_type: str) -> float | None:
        best = None
        best_step = -1
        for e in self._objects.values():
            if e.object_type == object_type and e.last_seen_step > best_step:
                best_step = e.last_seen_step
                best = e.egocentric_dist
        return best

    def get_remembered_type_distance(self, object_type: str) -> float | None:
        best = None
        best_step = -1
        for e in self._objects.values():
            if e.object_type == object_type and e.last_seen_step > best_step:
                best_step = e.last_seen_step
                best = e.egocentric_dist
        return best

    # ── searched markers ─────────────────────────────────────────────

    def mark_searched(self, object_type: str | None = None, object_id: str | None = None) -> int:
        count = 0
        for e in self._objects.values():
            if object_id is not None:
                if e.object_id == object_id:
                    e.searched = True
                    count += 1
            elif object_type is not None:
                if e.object_type == object_type:
                    e.searched = True
                    count += 1
        if object_type:
            self._searched_types[object_type] = self._searched_types.get(object_type, 0) + 1
        self._state_changed()
        return count

    def unmark_searched(self, object_type: str) -> int:
        had_type_marker = object_type in self._searched_types
        count = 0
        for e in self._objects.values():
            if e.object_type == object_type and e.searched:
                e.searched = False
                count += 1
        self._searched_types.pop(object_type, None)
        if count or had_type_marker:
            self._state_changed()
        return count

    def unmark_searched_by_id(self, object_id: str) -> int:
        e = self._objects.get(object_id)
        if e and e.searched:
            e.searched = False
            self._state_changed()
            return 1
        return 0

    # ── visitation tracking ──────────────────────────────────────────

    def record_receptacle_visit(self, receptacle_type: str) -> None:
        self.receptacle_visit_counts[receptacle_type] = \
            self.receptacle_visit_counts.get(receptacle_type, 0) + 1
        self._state_changed()

    def record_receptacle_open(self, receptacle_type: str) -> None:
        self.receptacle_open_counts[receptacle_type] = \
            self.receptacle_open_counts.get(receptacle_type, 0) + 1
        self.receptacle_visit_counts[receptacle_type] = \
            self.receptacle_visit_counts.get(receptacle_type, 0) + 1
        self._state_changed()

    def record_object_discovered_in(self, receptacle_type: str) -> None:
        self.objects_found_by_receptacle[receptacle_type] = \
            self.objects_found_by_receptacle.get(receptacle_type, 0) + 1
        self._state_changed()

    def record_hidden_object(self, object_id: str, container_id: str) -> bool:
        """Record a successful hide-object mutation without retaining its old location."""
        entry = self._objects.get(object_id)
        if entry is None:
            return False
        entry.parent_receptacle_id = container_id
        entry.status = "remembered"
        entry.last_seen_step = self._step_counter
        entry.searched = False
        self._searched_types.pop(entry.object_type, None)
        self._state_changed()
        return True

    def get_receptacle_entries_for_curiosity(self) -> list[dict]:
        entries = []
        seen_types = set()
        for e in self._objects.values():
            if not e.is_receptacle or e.object_type in seen_types:
                continue
            seen_types.add(e.object_type)
            entries.append({
                "object_type": e.object_type,
                "status": e.status,
                "egocentric_dir": e.egocentric_dir,
                "egocentric_dist": e.egocentric_dist,
                "last_seen_step": e.last_seen_step,
                "age_steps": self._step_counter - e.last_seen_step,
            })
        return entries

    # ── internal helpers ─────────────────────────────────────────────

    # (reuse from GeometricMemory via import or copy)
    _ENTITY_ALIASES: dict[str, str] = {
        "standardwallsize": "a wall", "wall": "a wall",
    }

    @staticmethod
    def _normalize_entity(raw: str) -> str:
        cleaned = raw
        if ":" in cleaned and cleaned.split(":")[0].startswith("FP"):
            cleaned = cleaned.split(":", 1)[1]
        cleaned = re.sub(r"[._]\w+$", "", cleaned)
        lower = cleaned.lower()
        if lower in SemanticMemory._ENTITY_ALIASES:
            return SemanticMemory._ENTITY_ALIASES[lower]
        words = re.findall(r"[A-Z][a-z]*|\d+", cleaned)
        if words:
            readable = " ".join(w.lower() for w in words)
            return f"an {readable}" if readable[0] in "aeiou" else f"a {readable}"
        return f"a {cleaned.lower()}"

    def _parse_blocker(self, error: str) -> str | None:
        blocker_id = self._parse_blocker_id(error)
        if blocker_id:
            return self._normalize_entity(blocker_id)
        if "blocking" in error.lower():
            parts = error.split(" blocking")
            if parts:
                before = parts[0].strip().split()
                if before:
                    return self._normalize_entity(before[-1].rstrip("."))
        return None

    @staticmethod
    def _parse_blocker_id(error: str) -> str | None:
        """Extract the simulator's obstacle ID without MoveSequence prose."""
        if " is blocking " not in error:
            return None
        raw = error.split(" is blocking ", 1)[0].strip()
        return raw.rsplit(": ", 1)[-1].strip() or None

    def _compress_error(self, action: str, error: str) -> str:
        blocker = self._parse_blocker(error)
        if blocker and action == "MoveAhead":
            return f"{action} blocked by {blocker}. I should MoveLeft or MoveBack."
        if "not holding anything" in error.lower():
            return "PutObject failed: my hand is empty."
        if "not found" in error.lower():
            return f"{action} failed: target not in reach."
        short = error.split(".")[0].strip()
        return short[:120]

    def _find_or_create_obstacle(self, obj_type: str) -> _ObstacleEntry:
        for o in self._obstacles:
            if o.object_type == obj_type:
                return o
        obs = _ObstacleEntry(object_type=obj_type, direction="ahead",
                             block_count=0, last_block_step=self._step_counter)
        self._obstacles.append(obs)
        return obs

    def _obstacle_suggestion(self, obs: _ObstacleEntry) -> str:
        if obs.direction in ("ahead", "ahead-left", "ahead-right"):
            return "I should MoveLeft or MoveBack — do NOT retry forward."
        elif obs.direction in ("left", "right"):
            return "I should MoveBack then Rotate."
        return "I should MoveBack to create space."

    def _age_entries(self):
        threshold = self._step_counter - self.AGING_THRESHOLD
        to_remove = [oid for oid, e in self._objects.items()
                     if e.last_seen_step <= threshold
                     and e.status in {"remembered", "placed"}
                     and e.object_type not in {self._task_target, self._task_receptacle}]
        for oid in to_remove:
            del self._objects[oid]
            self._object_state.pop(oid, None)

    def _update_area_label(self, metadata: dict):
        scene = metadata.get("sceneName", "")
        if scene:
            self._area_label = self._scene_to_area(scene)
            return
        if not self._objects:
            self._area_label = "unknown room"
            return
        all_types = {e.object_type for e in self._objects.values()}
        bedroom = {"Bed", "Pillow", "Nightstand", "Dresser", "AlarmClock"}
        kitchen = {"Fridge", "Microwave", "StoveBurner", "Sink", "CounterTop"}
        living = {"Sofa", "CoffeeTable", "Television", "Book"}
        office = {"Desk", "Laptop", "Chair", "DeskLamp", "SideTable"}
        scores = {"bedroom": len(all_types & bedroom), "kitchen": len(all_types & kitchen),
                  "living room": len(all_types & living), "office": len(all_types & office)}
        best = max(scores, key=scores.get)
        self._area_label = best if scores[best] > 0 else "room"

    @staticmethod
    def _scene_to_area(scene: str) -> str:
        match = re.search(r"FloorPlan(\d+)", scene)
        if match:
            mapping = {1: "kitchen", 2: "living room", 3: "bathroom", 4: "bedroom",
                       5: "living room", 6: "living room", 7: "kitchen",
                       8: "living room", 9: "living room", 10: "kitchen"}
            return mapping.get(int(match.group(1)), "room")
        return "room"
