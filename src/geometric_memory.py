"""
Geometric Egocentric Memory.

Flat-list spatial memory: accumulates all objects the agent has EVER seen
(visibleBounds2D=True). Objects stay in memory even when they leave view.
Stored by objectId internally; rendered by objectType with numbering for duplicates.
"""
from __future__ import annotations

import math, re
from dataclasses import dataclass, field
from collections import OrderedDict

from src.memory_interface import MemoryInterface

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _ObjectEntry:
    object_id: str
    object_type: str
    is_receptacle: bool
    is_task_receptacle: bool
    status: str  # "visible", "held", "placed", "remembered"
    egocentric_dir: str
    egocentric_dist: float
    last_seen_step: int
    seen_count: int
    parent_receptacle_id: str | None = None  # objectId of receptacle this object sits on/in
    searched: bool = False  # marked when receptacle has been opened and checked

@dataclass
class _ObstacleEntry:
    object_type: str
    direction: str
    block_count: int
    last_block_step: int


class StuckTracker:
    """Tracks repetition and blockage signals for situation summary.

    Uses cumulative per-(intent, target) dict so switching to another intent
    and back does NOT reset the counter — superior to sliding-window dedup.
    Blocked directions and obstacle-intent pairs are aged out by step expiry.
    """
    INTENT_REPEAT_THRESHOLD = 3
    OBSTACLE_REPEAT_THRESHOLD = 3
    INTENT_AGE_LIMIT = 20
    BLOCK_AGE_LIMIT = 15

    _NEUTRAL_ACTIONS = {
        "RotateLeft", "RotateRight", "LookUp", "LookDown",
        "Done", "LookAround",
    }

    def __init__(self):
        self._intent_failure: dict[tuple[str, str], int] = {}
        self._intent_last_step: dict[tuple[str, str], int] = {}
        self._consecutive_failures = 0
        self._blocked_directions: dict[tuple[str, str], int] = {}
        self._block_dir_last_step: dict[tuple[str, str], int] = {}
        self._obstacle_intent_blocks: dict[tuple[str, str, str], int] = {}
        self._obs_last_step: dict[tuple[str, str, str], int] = {}

    def update(self, step: int, intent: str, target: str,
               success: bool, blocked_by: str | None, blocked_dir: str | None,
               action: str = ""):
        key = (intent, target)
        if not success:
            self._intent_failure[key] = self._intent_failure.get(key, 0) + 1
        self._intent_last_step[key] = step
        self._age_dicts(self._intent_last_step, self.INTENT_AGE_LIMIT, step,
                        self._intent_failure)

        if action in self._NEUTRAL_ACTIONS:
            pass
        elif success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        if blocked_by and blocked_dir:
            dk = (blocked_dir, blocked_by)
            self._blocked_directions[dk] = self._blocked_directions.get(dk, 0) + 1
            self._block_dir_last_step[dk] = step
            self._age_dicts(self._block_dir_last_step, self.BLOCK_AGE_LIMIT, step,
                            self._blocked_directions)

        if blocked_by and intent:
            ok = (blocked_by, intent, target)
            self._obstacle_intent_blocks[ok] = self._obstacle_intent_blocks.get(ok, 0) + 1
            self._obs_last_step[ok] = step
            self._age_dicts(self._obs_last_step, self.BLOCK_AGE_LIMIT, step,
                            self._obstacle_intent_blocks)

    def render_summary(self, current_step: int) -> str:
        lines: list[str] = []
        for key in list(self._intent_last_step.keys()):
            intent, target = key
            fails = self._intent_failure.get(key, 0)
            if fails >= self.INTENT_REPEAT_THRESHOLD:
                lines.append(
                    f"WARNING: Intent '{intent} {target}' has failed "
                    f"{fails} times without recovery."
                )
        if self._consecutive_failures >= 5:
            lines.append(
                f"I have been stuck for {self._consecutive_failures} consecutive steps."
            )
        recent = {k: v for k, v in self._blocked_directions.items() if v >= 2}
        if recent:
            parts = [f"{d}({o}×{c})" for (d, o), c in sorted(recent.items())]
            lines.append(f"Blocked directions: {', '.join(parts)}.")
        for (obj, intent, tgt), cnt in self._obstacle_intent_blocks.items():
            if cnt >= self.OBSTACLE_REPEAT_THRESHOLD:
                lines.append(
                    f"The same obstacle ({obj}) has blocked intent "
                    f"'{intent} {tgt}' {cnt} times."
                )
        if not lines:
            return ""
        return "SITUATION SUMMARY:\n" + "\n".join(f"  {ln}" for ln in lines) + "\n"

    @staticmethod
    def _age_dicts(last_step: dict, age_limit: int, current_step: int, *dicts):
        stale = [k for k, s in last_step.items() if current_step - s > age_limit]
        for k in stale:
            last_step.pop(k, None)
            for d in dicts:
                d.pop(k, None)


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

def _egocentric_direction(
    agent_x: float, agent_z: float, agent_rot_y: float,
    obj_x: float, obj_z: float,
) -> tuple[str, float]:
    dx = obj_x - agent_x
    dz = obj_z - agent_z
    rad = math.radians(agent_rot_y)
    fx, fz = math.sin(rad), math.cos(rad)
    forward = dx * fx + dz * fz
    right = dx * fz - dz * fx
    angle = math.degrees(math.atan2(right, forward))
    d = math.sqrt(dx * dx + dz * dz)
    if -22.5 <= angle <= 22.5:        label = "ahead"
    elif 22.5 < angle <= 67.5:        label = "ahead-right"
    elif 67.5 < angle <= 112.5:       label = "right"
    elif 112.5 < angle <= 157.5:      label = "behind-right"
    elif angle > 157.5 or angle < -157.5: label = "behind"
    elif -157.5 <= angle < -112.5:    label = "behind-left"
    elif -112.5 <= angle < -67.5:     label = "left"
    else:                             label = "ahead-left"
    return label, d


# ---------------------------------------------------------------------------
# Spike 003: auto-discovery tracking helper
# ---------------------------------------------------------------------------

def _auto_record_discovery(
        memory: "GeometricMemory",
    new_obj_id: str,
    obj_data: dict,
    visible_objects: list[dict],
) -> None:
    """When a newly-discovered object enters memory, credit its parent receptacle.

    Finds the parent receptacle's objectType and increments
    memory.objects_found_by_receptacle for that type.
    """
    parent_ids = obj_data.get("parentReceptacles") or []
    if not parent_ids:
        return

    parent_oid = parent_ids[0]
    # Look up parent receptacle type: first in memory, then in visible_objects
    parent_type = None
    entry = memory._objects.get(parent_oid)
    if entry and entry.is_receptacle:
        parent_type = entry.object_type
    else:
        for vo in visible_objects:
            if vo.get("objectId") == parent_oid and vo.get("receptacle"):
                parent_type = vo.get("objectType")
                break

    if parent_type:
        memory.objects_found_by_receptacle[parent_type] = \
            memory.objects_found_by_receptacle.get(parent_type, 0) + 1


# ---------------------------------------------------------------------------
# Memory class
# ---------------------------------------------------------------------------

class GeometricMemory(MemoryInterface):
    AGING_THRESHOLD: int = 20
    BLOCK_THRESHOLD: int = 2

    def __init__(self):
        self._objects: dict[str, _ObjectEntry] = {}  # objectId -> entry
        self._obstacles: list[_ObstacleEntry] = []
        self._step_counter: int = 0
        self._last_error: str | None = None
        self._agent_path: list[tuple[float, float]] = []
        self._area_label: str = ""
        self._agent_rot_y: float = 0.0
        self._agent_x: float = 0.0
        self._agent_z: float = 0.0
        self._stuck_tracker = StuckTracker()

        # ── Spike 003: Curiosity Scoreboard visitation tracking ──
        # Track per-receptacle-type statistics for multi-dimensional scoring.
        # Keyed by objectType (e.g. "Fridge", "CounterTop", "Cabinet").
        self.receptacle_visit_counts: dict[str, int] = {}
        self.receptacle_open_counts: dict[str, int] = {}
        self.objects_found_by_receptacle: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        self._agent_path.append((self._agent_x, self._agent_z))
        if len(self._agent_path) > 20:
            self._agent_path = self._agent_path[-20:]

        task_receptacle = self._extract_task_receptacle(task_criteria)
        inventory = metadata.get("inventoryObjects") or []
        held_types = {o.get("objectType", "") for o in inventory}

        # Track which objectIds are in the current frame (visibleBounds2D)
        current_ids: set[str] = set()

        for obj in visible_objects:
            oid = obj.get("objectId", "")
            if not oid:
                continue
            otype = obj.get("objectType", "")
            if not otype:
                continue

            # Only add objects that are actually visible in the current frame
            if not obj.get("visibleBounds2D"):
                continue

            current_ids.add(oid)
            pos = obj.get("position", {})
            # Use AABB surface position when available for accurate distance
            bbox = obj.get("axisAlignedBoundingBox")
            if bbox and bbox.get("cornerPoints"):
                corners = bbox["cornerPoints"]
                # cornerPoints format: [[x,y,z], [x,y,z], ...]
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
            is_task_recep = is_recep and otype == task_receptacle
            is_held = otype in held_types

            if is_held:
                status = "held"
            else:
                status = "visible"

            if oid in self._objects:
                entry = self._objects[oid]
                entry.egocentric_dir = direction
                entry.egocentric_dist = dist
                entry.status = status
                entry.last_seen_step = self._step_counter
                entry.seen_count += 1
                entry.is_receptacle = is_recep
                entry.is_task_receptacle = is_task_recep
                entry.parent_receptacle_id = (obj.get("parentReceptacles") or [None])[0]
            else:
                self._objects[oid] = _ObjectEntry(
                    object_id=oid,
                    object_type=otype,
                    is_receptacle=is_recep,
                    is_task_receptacle=is_task_recep,
                    status=status,
                    egocentric_dir=direction,
                    egocentric_dist=dist,
                    last_seen_step=self._step_counter,
                    seen_count=1,
                    parent_receptacle_id=(obj.get("parentReceptacles") or [None])[0],
                )
                # ── Spike 003: track object discovery per receptacle ──
                _auto_record_discovery(self, oid, obj, visible_objects)

        # Mark previously-visible objects that are no longer in view as "remembered"
        for oid, entry in self._objects.items():
            if oid not in current_ids and entry.status in ("visible",):
                entry.status = "remembered"

        # Handle held objects that may NOT be in visible_objects
        for held_type in held_types:
            for entry in self._objects.values():
                if entry.object_type == held_type and entry.status != "held":
                    entry.status = "held"
                    entry.last_seen_step = self._step_counter

        # Handle PutObject success: mark held as "placed"
        if action == "PutObject" and success:
            for entry in self._objects.values():
                if entry.status == "held":
                    entry.status = "placed"
                    entry.last_seen_step = self._step_counter

        # Handle failures: track obstacles
        blocker = None
        blocker_dir = None
        if not success and error_message:
            blocker = self._parse_blocker(error_message)
            if blocker:
                obs = self._find_or_create_obstacle(blocker)
                obs.block_count += 1
                obs.last_block_step = self._step_counter
                blocker_dir = obs.direction
            self._last_error = self._compress_error(action, error_message)

        # Update stuck tracker (PR #2 — cumulative intent failure tracking)
        intent_parts = intent.split(maxsplit=1)
        intent_verb = intent_parts[0] if intent_parts else ""
        intent_target = intent_parts[1] if len(intent_parts) > 1 else ""
        self._stuck_tracker.update(
            step=self._step_counter,
            intent=intent_verb,
            target=intent_target,
            success=success,
            blocked_by=blocker,
            blocked_dir=blocker_dir,
            action=action,
        )

        self._age_entries()
        if self._step_counter % 5 == 0 or not self._area_label:
            self._update_area_label(metadata)

    def mark_searched(self, object_type: str | None = None, object_id: str | None = None) -> int:
        """Mark objects as searched. Returns count of objects marked.

        Args:
            object_type: Match entries with this objectType (case-sensitive).
            object_id: Match a specific entry by objectId. Takes precedence over object_type.

        Returns:
            Number of entries marked.
        """
        count = 0
        for entry in self._objects.values():
            if object_id is not None:
                if entry.object_id == object_id:
                    entry.searched = True
                    count += 1
            elif object_type is not None:
                if entry.object_type == object_type:
                    entry.searched = True
                    count += 1
        return count

    def unmark_searched(self, object_type: str) -> int:
        """Remove searched flag from all entries matching object_type. Returns count unmarked."""
        count = 0
        for entry in self._objects.values():
            if entry.object_type == object_type and entry.searched:
                entry.searched = False
                count += 1
        return count

    def unmark_searched_by_id(self, object_id: str) -> int:
        """Remove searched flag from a specific entry by objectId. Returns 1 if found, 0 otherwise."""
        entry = self._objects.get(object_id)
        if entry and entry.searched:
            entry.searched = False
            return 1
        return 0

    def render(self) -> str:
        lines = []
        situation = self._stuck_tracker.render_summary(self._step_counter)
        if situation:
            lines.append(situation)
        lines.append("SPATIAL MEMORY — what you remember seeing:\n")
        if self._area_label:
            lines.append(f"AREA: {self._area_label}")
        lines.append("OBJECTS:")
        obj_lines = self._render_objects()
        if obj_lines:
            lines.extend(obj_lines)

        active_obstacles = [
            o for o in self._obstacles
            if o.block_count >= self.BLOCK_THRESHOLD
            and (self._step_counter - o.last_block_step) <= 5
        ]
        if active_obstacles:
            lines.append("\nOBSTACLES:")
            for o in active_obstacles:
                suggestion = self._obstacle_suggestion(o)
                lines.append(
                    f"  {o.object_type} {o.direction} blocked MoveAhead "
                    f"{o.block_count} times. {suggestion}"
                )

        if self._last_error:
            lines.append(f"\nLAST ERROR: {self._last_error}")

        result = "\n".join(lines)
        if len(result) > 1500:
            result = result[:1497] + "..."
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_task_receptacle(self, criteria: str) -> str:
        if "inside " in criteria:
            return criteria.split("inside ")[-1].strip()
        return ""

    _ENTITY_ALIASES: dict[str, str] = {
        "standardwallsize": "a wall",
        "wall": "a wall",
    }

    @staticmethod
    def _normalize_entity(raw: str) -> str:
        cleaned = raw
        if ":" in cleaned and cleaned.split(":")[0].startswith("FP"):
            cleaned = cleaned.split(":", 1)[1]
        cleaned = re.sub(r"[._]\w+$", "", cleaned)
        lower = cleaned.lower()
        if lower in GeometricMemory._ENTITY_ALIASES:
            return GeometricMemory._ENTITY_ALIASES[lower]
        words = re.findall(r"[A-Z][a-z]*|\d+", cleaned)
        if words:
            readable = " ".join(w.lower() for w in words)
            if readable[0] in "aeiou":
                return f"an {readable}"
            return f"a {readable}"
        return f"a {cleaned.lower()}"

    def _parse_blocker(self, error: str) -> str | None:
        if " is blocking " in error:
            raw = error.split(" is blocking ")[0].strip()
            return self._normalize_entity(raw)
        if "blocking" in error.lower():
            parts = error.split(" blocking")
            if parts:
                before = parts[0].strip().split()
                if before:
                    return self._normalize_entity(before[-1].rstrip("."))
        return None

    def _compress_error(self, action: str, error: str) -> str:
        blocker = self._parse_blocker(error)
        if blocker and action == "MoveAhead":
            return f"{action} blocked by {blocker}. MoveLeft or MoveBack to go around."
        if "not holding anything" in error.lower():
            return "PutObject failed: hand empty. Object was already placed."
        if "not found" in error.lower():
            return f"{action} failed: target object not in reach."
        short = error.split(".")[0].strip()
        if len(short) > 120:
            short = short[:117] + "..."
        return short

    def _find_or_create_obstacle(self, obj_type: str) -> _ObstacleEntry:
        for o in self._obstacles:
            if o.object_type == obj_type:
                return o
        obs = _ObstacleEntry(
            object_type=obj_type, direction="ahead",
            block_count=0, last_block_step=self._step_counter,
        )
        self._obstacles.append(obs)
        return obs

    def _obstacle_suggestion(self, obs: _ObstacleEntry) -> str:
        if obs.direction in ("ahead", "ahead-left", "ahead-right"):
            return "MoveLeft or MoveBack — do NOT retry forward."
        elif obs.direction in ("left", "right"):
            return "MoveBack then Rotate to find clear path."
        return "MoveBack to create space."

    def _age_entries(self):
        threshold = self._step_counter - self.AGING_THRESHOLD
        to_remove = [
            oid for oid, e in self._objects.items()
            if e.last_seen_step <= threshold and e.status == "remembered"
        ]
        for oid in to_remove:
            del self._objects[oid]

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
        scores = {
            "bedroom": len(all_types & bedroom),
            "kitchen": len(all_types & kitchen),
            "living room": len(all_types & living),
            "office": len(all_types & office),
        }
        best = max(scores, key=scores.get)
        self._area_label = best if scores[best] > 0 else "room"

    def _scene_to_area(self, scene: str) -> str:
        match = re.search(r"FloorPlan(\d+)", scene)
        if match:
            num = int(match.group(1))
            mapping = {
                1: "kitchen", 2: "living room", 3: "bathroom", 4: "bedroom",
                5: "living room", 6: "living room", 7: "kitchen",
                8: "living room", 9: "living room", 10: "kitchen",
            }
            return mapping.get(num, "room")
        return "room"

    def _render_objects(self) -> list[str]:
        all_entries = list(self._objects.values())

        def _sort_key(e: _ObjectEntry) -> tuple:
            if e.status == "held":            prio = 0
            elif e.is_task_receptacle:        prio = 1
            elif e.is_receptacle:             prio = 2
            else:                             prio = 3
            return (prio, -e.last_seen_step)

        sorted_entries = sorted(all_entries, key=_sort_key)

        # Detect duplicate types and assign numbers
        type_counts: dict[str, int] = {}
        type_indices: dict[str, int] = {}
        for e in sorted_entries:
            t = e.object_type
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, count in type_counts.items():
            if count > 1:
                type_indices[t] = 0

        # Build labels and store for parent-reference resolution
        oid_to_label: dict[str, str] = {}
        lines: list[str] = []
        for e in sorted_entries:
            label = e.object_type
            if e.object_type in type_indices:
                type_indices[e.object_type] += 1
                label = f"{e.object_type}{type_indices[e.object_type]}"
            oid_to_label[e.object_id] = label
            line = self._render_object_line(e, label, oid_to_label)
            if line:
                lines.append(line)
        return lines

    # ------------------------------------------------------------------
    # Exploration / dedup helpers
    # ------------------------------------------------------------------

    def get_remembered_receptacles(self) -> list[str]:
        """Return receptacle object types that are remembered (not currently visible),
        sorted by most recently seen. Used for dedup fallback exploration."""
        candidates: list[tuple[str, int]] = []
        for entry in self._objects.values():
            if entry.is_receptacle and entry.status == "remembered":
                candidates.append((entry.object_type, entry.last_seen_step))
        candidates.sort(key=lambda x: -x[1])
        seen: set[str] = set()
        result: list[str] = []
        for otype, _ in candidates:
            if otype not in seen:
                seen.add(otype)
                result.append(otype)
        return result

    def get_visible_receptacles(self) -> list[str]:
        """Return receptacle object types currently visible, sorted by distance."""
        candidates: list[tuple[str, float]] = []
        for entry in self._objects.values():
            if entry.is_receptacle and entry.status == "visible":
                candidates.append((entry.object_type, entry.egocentric_dist))
        candidates.sort(key=lambda x: x[1])
        seen: set[str] = set()
        result: list[str] = []
        for otype, _ in candidates:
            if otype not in seen:
                seen.add(otype)
                result.append(otype)
        return result

    def get_all_object_types(self) -> set[str]:
        """Return all unique objectTypes ever stored in memory.

        Used by intent dedup to normalize LLM-generated target strings
        (e.g. "refrigerator" -> "Fridge") against known canonical names.
        """
        return {e.object_type for e in self._objects.values()}

    def get_unvisited_receptacles(self, approached_types: set[str] | None = None) -> list[str]:
        """Return receptacle types in memory that have NOT been approached yet.
        Prioritizes remembered over visible, then by recency."""
        approached = approached_types or set()
        remembered: list[tuple[str, int]] = []
        visible: list[tuple[str, float]] = []
        for entry in self._objects.values():
            if not entry.is_receptacle:
                continue
            if entry.object_type in approached:
                continue
            if entry.status == "remembered":
                remembered.append((entry.object_type, entry.last_seen_step))
            elif entry.status == "visible":
                visible.append((entry.object_type, entry.egocentric_dist))
        remembered.sort(key=lambda x: -x[1])
        visible.sort(key=lambda x: x[1])
        seen: set[str] = set()
        result: list[str] = []
        for otype, _ in remembered:
            if otype not in seen:
                seen.add(otype)
                result.append(otype)
        for otype, _ in visible:
            if otype not in seen:
                seen.add(otype)
                result.append(otype)
        return result

    # ------------------------------------------------------------------
    # Phase detection helpers (Spike 005: progress-gating)
    # ------------------------------------------------------------------

    def has_type(self, object_type: str) -> bool:
        """Check if any object of this type has ever been seen / stored in memory."""
        for entry in self._objects.values():
            if entry.object_type == object_type:
                return True
        return False

    def is_type_visible(self, object_type: str) -> bool:
        """Check if at least one instance of this type is currently in view."""
        for entry in self._objects.values():
            if entry.object_type == object_type and entry.status == "visible":
                return True
        return False

    def get_type_distance(self, object_type: str) -> float | None:
        """Return AABB surface distance to the closest visible instance of this type.
        Returns None if no visible instance is found."""
        best: float | None = None
        for entry in self._objects.values():
            if entry.object_type == object_type and entry.status == "visible":
                if best is None or entry.egocentric_dist < best:
                    best = entry.egocentric_dist
        return best

    def get_remembered_type_distance(self, object_type: str) -> float | None:
        """Return last known distance for this type (most recent seen, not necessarily
        currently visible). Returns None if never seen."""
        best: float | None = None
        best_step: int = -1
        for entry in self._objects.values():
            if entry.object_type == object_type:
                if entry.last_seen_step > best_step:
                    best_step = entry.last_seen_step
                    best = entry.egocentric_dist
        return best

    def get_searched_receptacle_types(self) -> set[str]:
        """Return the set of receptacle objectTypes that have been marked as searched."""
        result: set[str] = set()
        for entry in self._objects.values():
            if entry.searched and entry.is_receptacle:
                result.add(entry.object_type)
        return result

    # ------------------------------------------------------------------
    # Spike 003: Curiosity Scoreboard visitation tracking
    # ------------------------------------------------------------------

    def record_receptacle_visit(self, receptacle_type: str) -> None:
        """Record that the agent visited/approached this receptacle type."""
        self.receptacle_visit_counts[receptacle_type] = \
            self.receptacle_visit_counts.get(receptacle_type, 0) + 1

    def record_receptacle_open(self, receptacle_type: str) -> None:
        """Record that the agent opened/interacted with this receptacle type."""
        self.receptacle_open_counts[receptacle_type] = \
            self.receptacle_open_counts.get(receptacle_type, 0) + 1
        # Opening counts as a visit too (you can't open without approaching)
        self.receptacle_visit_counts[receptacle_type] = \
            self.receptacle_visit_counts.get(receptacle_type, 0) + 1

    def record_object_discovered_in(self, receptacle_type: str) -> None:
        """Record that a previously-unseen object was discovered when interacting
        with this receptacle. Used to compute discovery score."""
        self.objects_found_by_receptacle[receptacle_type] = \
            self.objects_found_by_receptacle.get(receptacle_type, 0) + 1

    def get_receptacle_entries_for_curiosity(self) -> list[dict]:
        """Return all known receptacle entries as a list of dicts suitable for
        consumption by curiosity_scorer.build_curiosity_table_text().

        Each dict has: object_type, status, egocentric_dir, egocentric_dist,
        last_seen_step, age_steps (steps since last seen).
        """
        entries: list[dict] = []
        seen_types: set[str] = set()
        for e in self._objects.values():
            if not e.is_receptacle:
                continue
            if e.object_type in seen_types:
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

    def _render_object_line(self, e: _ObjectEntry, label: str,
                            oid_to_label: dict[str, str] | None = None) -> str:
        if e.status == "held":
            status_text = "held in hand"
        elif e.status == "placed":
            status_text = "placed (hand empty now)"
        elif e.status == "visible":
            status_text = f"in view {e.egocentric_dir} {e.egocentric_dist:.1f}m"
        else:  # remembered
            age = self._step_counter - e.last_seen_step
            status_text = f"saw {e.egocentric_dir} ~{e.egocentric_dist:.1f}m ~{age} steps ago"

        tags = []
        if e.is_task_receptacle:
            tags.append("receptacle")
        elif e.is_receptacle:
            tags.append("receptacle?")
        if e.searched:
            tags.append("SEARCHED")

        line = f"  {label} — {status_text}"
        # Show parent relationship if known (object-to-object spatial relation)
        if e.parent_receptacle_id and oid_to_label:
            parent_label = oid_to_label.get(e.parent_receptacle_id)
            if parent_label:
                if e.status == "visible":
                    line += f" on {parent_label}"
                else:
                    line += f" (was on {parent_label})"
        if tags:
            line += f" ({', '.join(tags)})"
        return line


# ---------------------------------------------------------------------------
# SearchTrail: 2D grid tracking where the agent has physically been
# ---------------------------------------------------------------------------

class SearchTrail:
    """2D grid tracking physical agent positions for revisit detection.

    Maintains a coarse grid of visited positions with visit counts.
    Provides revisit scores and text summaries for Planner/Executor prompts.

    Resolution defaults to 0.5m -- cells are ~0.5m x 0.5m squares.
    """

    def __init__(self, resolution: float = 0.5):
        self._cells: dict[tuple[int, int], int] = {}  # (gx, gz) -> count
        self._resolution = resolution

    def record(self, agent_x: float, agent_z: float) -> int:
        gx = int(agent_x / self._resolution)
        gz = int(agent_z / self._resolution)
        self._cells[(gx, gz)] = self._cells.get((gx, gz), 0) + 1
        return self._cells[(gx, gz)]

    def revisit_score(self, target_x: float, target_z: float) -> float:
        """0.0 = never visited, higher = more visits."""
        gx = int(target_x / self._resolution)
        gz = int(target_z / self._resolution)
        return float(self._cells.get((gx, gz), 0))

    def total_cells_visited(self) -> int:
        return len(self._cells)

    def render_summary(self, receptacles: list[dict], agent_pos: dict | None = None) -> str:
        """Compact text showing which receptacle areas the agent has visited.

        Args:
            receptacles: list of dicts with objectType, position, receptacle=True.
            agent_pos: optional dict with x, z for the agent's own cell marker.

        Returns:
            Multi-line string like:
            TRAIL (visited areas):
              Fridge area x3 (HEAVILY visited), CounterTop x1, Cabinet x0 (unvisited)
              -> Prioritize UNVISITED or least-visited areas.
        """
        # Group receptacles by objectType, dedup, get visit count per type
        seen_types: set[str] = set()
        entries: list[tuple[str, float]] = []
        for obj in receptacles:
            otype = obj.get("objectType", "")
            if not otype or otype in seen_types:
                continue
            if not obj.get("receptacle"):
                continue
            seen_types.add(otype)
            pos = obj.get("position", {})
            score = self.revisit_score(pos.get("x", 0), pos.get("z", 0))
            entries.append((otype, score))

        if not entries:
            return ""

        # Build compact summary line
        parts: list[str] = []
        for otype, score in entries:
            if score >= 3:
                parts.append(f"{otype} area x{int(score)} (HEAVILY visited)")
            elif score >= 1:
                parts.append(f"{otype} area x{int(score)}")
            else:
                parts.append(f"{otype} x0 (unvisited)")

        lines = ["TRAIL (visited areas):"]
        lines.append("  " + ", ".join(parts))
        lines.append("  -> Prioritize UNVISITED or least-visited areas.")
        return "\n".join(lines)
