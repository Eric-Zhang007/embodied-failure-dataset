"""
Egocentric Compressed Memory.

Accumulates all objects the agent has EVER seen (visibleBounds2D=True).
Objects stay in memory even when they leave the current view.
Stored by objectId internally; rendered by objectType with numbering for duplicates.
"""
from __future__ import annotations

import math, re
from dataclasses import dataclass, field
from collections import OrderedDict

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

@dataclass
class _ObstacleEntry:
    object_type: str
    direction: str
    block_count: int
    last_block_step: int

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
# Memory class
# ---------------------------------------------------------------------------

class EgocentricMemory:
    AGING_THRESHOLD: int = 20
    MAX_OBJECTS: int = 15
    MAX_OBSTACLES: int = 3
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
                )

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
        if not success and error_message:
            blocker = self._parse_blocker(error_message)
            if blocker:
                obs = self._find_or_create_obstacle(blocker)
                obs.block_count += 1
                obs.last_block_step = self._step_counter
            self._last_error = self._compress_error(action, error_message)

        self._age_entries()
        if self._step_counter % 5 == 0 or not self._area_label:
            self._update_area_label(metadata)

    def render(self) -> str:
        lines = []
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
            for o in active_obstacles[:self.MAX_OBSTACLES]:
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
        if lower in EgocentricMemory._ENTITY_ALIASES:
            return EgocentricMemory._ENTITY_ALIASES[lower]
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

        lines: list[str] = []
        for e in sorted_entries[:self.MAX_OBJECTS]:
            label = e.object_type
            if e.object_type in type_indices:
                type_indices[e.object_type] += 1
                label = f"{e.object_type}{type_indices[e.object_type]}"
            line = self._render_object_line(e, label)
            if line:
                lines.append(line)
        return lines

    def _render_object_line(self, e: _ObjectEntry, label: str) -> str:
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

        line = f"  {label} — {status_text}"
        if tags:
            line += f" ({', '.join(tags)})"
        return line
