"""
Egocentric Compressed Memory.

Replaces the verbose JSON action history (~6000+ chars) with a compact spatial
memory (~400-800 chars) that tracks what the agent has seen and where, expressed
in first-person egocentric terms. No omniscient information. No world coordinates
exposed to the VLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _ObjectEntry:
    """Internal tracking for one observed object."""

    object_type: str
    is_receptacle: bool
    is_task_receptacle: bool  # matches TASK COMPLETION CRITERIA parent_target
    status: str  # "visible", "held", "remembered", "placed"
    egocentric_dir: str  # direction label at time last seen
    egocentric_dist: float  # distance in meters at time last seen
    world_x: float  # internal only — for dedup
    world_z: float  # internal only — for dedup
    last_seen_step: int
    seen_count: int


@dataclass
class _ObstacleEntry:
    """Persistent obstacle that has blocked movement."""

    object_type: str
    direction: str  # relative to agent at last block
    block_count: int
    last_block_step: int


# ---------------------------------------------------------------------------
# Direction helpers (mirrors eb_agent._direction logic)
# ---------------------------------------------------------------------------


def _egocentric_direction(
    agent_x: float, agent_z: float, agent_rot_y: float,
    obj_x: float, obj_z: float,
) -> tuple[str, float]:
    """Return (direction_label, distance_m) relative to agent."""
    dx = obj_x - agent_x
    dz = obj_z - agent_z
    rad = math.radians(agent_rot_y)
    fx, fz = math.sin(rad), math.cos(rad)
    forward = dx * fx + dz * fz
    right = dx * fz - dz * fx
    angle = math.degrees(math.atan2(right, forward))
    d = math.sqrt(dx * dx + dz * dz)

    if -22.5 <= angle <= 22.5:
        label = "ahead"
    elif 22.5 < angle <= 67.5:
        label = "ahead-right"
    elif 67.5 < angle <= 112.5:
        label = "right"
    elif 112.5 < angle <= 157.5:
        label = "behind-right"
    elif angle > 157.5 or angle < -157.5:
        label = "behind"
    elif -157.5 <= angle < -112.5:
        label = "behind-left"
    elif -112.5 <= angle < -67.5:
        label = "left"
    else:
        label = "ahead-left"
    return label, d


# ---------------------------------------------------------------------------
# Memory class
# ---------------------------------------------------------------------------


class EgocentricMemory:
    """Maintains a running egocentric spatial memory across steps."""

    AGING_THRESHOLD: int = 10  # steps before an unseen object is forgotten
    MAX_OBJECTS: int = 12  # max lines in OBJECTS block
    MAX_OBSTACLES: int = 3  # max lines in OBSTACLES block
    BLOCK_THRESHOLD: int = 2  # min blocks before reporting obstacle

    def __init__(self):
        self._objects: dict[str, list[_ObjectEntry]] = {}  # type -> list
        self._obstacles: list[_ObstacleEntry] = []
        self._step_counter: int = 0
        self._last_error: str | None = None
        # agent path for area tracking
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
        """Called after every action execution."""
        self._step_counter += 1

        # Track agent position / orientation internally
        agent = metadata.get("agent", {})
        self._agent_x = agent.get("position", {}).get("x", 0.0)
        self._agent_z = agent.get("position", {}).get("z", 0.0)
        self._agent_rot_y = agent.get("rotation", {}).get("y", 0.0)
        self._agent_path.append((self._agent_x, self._agent_z))
        if len(self._agent_path) > 20:
            self._agent_path = self._agent_path[-20:]

        # Determine task-relevant receptacle type
        task_receptacle = self._extract_task_receptacle(task_criteria)

        # Update visible objects
        inventory = metadata.get("inventoryObjects") or []
        held_types = {o.get("objectType", "") for o in inventory}

        for obj in visible_objects:
            otype = obj.get("objectType", "")
            if not otype:
                continue
            pos = obj.get("position", {})
            ox, oz = pos.get("x", 0.0), pos.get("z", 0.0)
            direction, dist = _egocentric_direction(
                self._agent_x, self._agent_z, self._agent_rot_y, ox, oz,
            )
            is_recep = obj.get("receptacle", False)
            is_task_recep = is_recep and otype == task_receptacle
            is_held = otype in held_types

            existing = self._find_entry(otype, ox, oz)

            if is_held:
                status = "held"
            else:
                status = "visible"

            if existing:
                existing.egocentric_dir = direction
                existing.egocentric_dist = dist
                existing.status = status
                existing.last_seen_step = self._step_counter
                existing.seen_count += 1
            else:
                entry = _ObjectEntry(
                    object_type=otype,
                    is_receptacle=is_recep,
                    is_task_receptacle=is_task_recep,
                    status=status,
                    egocentric_dir=direction,
                    egocentric_dist=dist,
                    world_x=ox,
                    world_z=oz,
                    last_seen_step=self._step_counter,
                    seen_count=1,
                )
                self._objects.setdefault(otype, []).append(entry)

        # Update held objects that may NOT be in visible_objects
        for held_type in held_types:
            entries = self._objects.get(held_type, [])
            for e in entries:
                if e.status != "held":
                    e.status = "held"
                    e.last_seen_step = self._step_counter

        # Handle PutObject success: mark held object as "placed"
        if action == "PutObject" and success:
            for entries in self._objects.values():
                for e in entries:
                    if e.status == "held":
                        e.status = "placed"
                        e.last_seen_step = self._step_counter

        # Handle failures: track obstacles
        if not success and error_message:
            blocker = self._parse_blocker(error_message)
            if blocker:
                obs = self._find_or_create_obstacle(blocker)
                obs.block_count += 1
                obs.last_block_step = self._step_counter
            self._last_error = self._compress_error(action, error_message)
        else:
            # On success, clear last error after a few steps
            pass

        # Age out old entries
        self._age_entries()

        # Update area label periodically
        if self._step_counter % 5 == 0 or not self._area_label:
            self._update_area_label(metadata)

    def render(self) -> str:
        """Generate the compact memory text for the Phase 1 prompt."""
        lines = []
        lines.append("SPATIAL MEMORY — what you remember seeing:\n")

        # AREA line
        if self._area_label:
            lines.append(f"AREA: {self._area_label}")

        # OBJECTS block
        lines.append("OBJECTS:")
        obj_lines = self._render_objects()
        if obj_lines:
            lines.extend(obj_lines)

        # OBSTACLES block
        active_obstacles = [
            o for o in self._obstacles
            if o.block_count >= self.BLOCK_THRESHOLD
            and (self._step_counter - o.last_block_step) <= 5
        ]
        if active_obstacles:
            lines.append("\nOBSTACLES:")
            for o in active_obstacles[: self.MAX_OBSTACLES]:
                suggestion = self._obstacle_suggestion(o)
                lines.append(
                    f"  {o.object_type} {o.direction} blocked MoveAhead "
                    f"{o.block_count} times. {suggestion}"
                )

        # LAST ERROR
        if self._last_error:
            lines.append(f"\nLAST ERROR: {self._last_error}")

        result = "\n".join(lines)

        # Enforce hard cap
        if len(result) > 1000:
            # Truncate from OBJECTS block
            header_end = result.find("OBJECTS:\n") + len("OBJECTS:\n")
            obj_start = header_end
            obj_end = result.find("\n\nOBSTACLES:", obj_start)
            if obj_end < 0:
                obj_end = result.find("\n\nLAST ERROR:", obj_start)
            if obj_end < 0:
                obj_end = len(result)
            # Keep first N object lines
            obj_section = result[obj_start:obj_end]
            obj_lines_list = obj_section.strip().split("\n")
            kept = obj_lines_list[:8]  # keep only 8 object lines
            truncated_objects = "\n".join(kept)
            if len(obj_lines_list) > 8:
                truncated_objects += f"\n  ... ({len(obj_lines_list) - 8} more objects)"
            result = (
                result[:obj_start]
                + truncated_objects
                + result[obj_end:]
            )

        # Final cap
        if len(result) > 1000:
            result = result[:997] + "..."

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_entry(self, otype: str, wx: float, wz: float) -> _ObjectEntry | None:
        """Find existing entry of same type near the given world position."""
        candidates = self._objects.get(otype, [])
        for e in candidates:
            dist = math.sqrt((e.world_x - wx) ** 2 + (e.world_z - wz) ** 2)
            if dist < 0.5:  # same instance if within 0.5m
                return e
        return None

    def _extract_task_receptacle(self, criteria: str) -> str:
        """Extract receptacle type from task criteria text like 'AlarmClock must be inside Desk'."""
        if "inside " in criteria:
            return criteria.split("inside ")[-1].strip()
        return ""

    # Known AI2-THOR internal names → human-readable
    _ENTITY_ALIASES: dict[str, str] = {
        "standardwallsize": "a wall",
        "wall": "a wall",
    }

    @staticmethod
    def _normalize_entity(raw: str) -> str:
        """Convert AI2-THOR internal entity name to human-readable label.
        
        Examples:
            FP304:StandardWallSize.001 → a wall
            Desk_9e51b54b            → a desk
            DiningTable_524ab915     → a dining table
            Chair_df9e062b           → a chair
        """
        # Strip FloorPlan prefix: FP304:StandardWallSize.001 → StandardWallSize.001
        cleaned = raw
        if ":" in cleaned and cleaned.split(":")[0].startswith("FP"):
            cleaned = cleaned.split(":", 1)[1]
        
        # Strip instance suffix: Desk_9e51b54b → Desk, StandardWallSize.001 → StandardWallSize
        import re
        cleaned = re.sub(r"[._]\w+$", "", cleaned)
        
        # Check aliases
        lower = cleaned.lower()
        if lower in EgocentricMemory._ENTITY_ALIASES:
            return EgocentricMemory._ENTITY_ALIASES[lower]
        
        # CamelCase → human-readable: DiningTable → dining table
        words = re.findall(r"[A-Z][a-z]*|\d+", cleaned)
        if words:
            readable = " ".join(w.lower() for w in words)
            # Add article
            if readable[0] in "aeiou":
                return f"an {readable}"
            return f"a {readable}"
        
        return f"a {cleaned.lower()}"

    def _parse_blocker(self, error: str) -> str | None:
        """Extract blocker object name from AI2-THOR error message."""
        # Pattern: "ObjectName is blocking Agent 0 from moving..."
        if " is blocking " in error:
            raw = error.split(" is blocking ")[0].strip()
            return self._normalize_entity(raw)
        # Pattern: "ObjectName is blocking..."
        if "blocking" in error.lower():
            parts = error.split(" blocking")
            if parts:
                before = parts[0].strip().split()
                if before:
                    raw = before[-1].rstrip(".")
                    return self._normalize_entity(raw)
        return None

    def _compress_error(self, action: str, error: str) -> str:
        """Compress error message to one line."""
        blocker = self._parse_blocker(error)
        if blocker and action == "MoveAhead":
            return f"{action} blocked by {blocker}. MoveLeft or MoveBack to go around."
        if "not holding anything" in error.lower():
            return "PutObject failed: hand empty. Object was already placed."
        if "not found" in error.lower():
            return f"{action} failed: target object not in reach."
        # Generic compression
        short = error.split(".")[0].strip()
        if len(short) > 120:
            short = short[:117] + "..."
        return short

    def _find_or_create_obstacle(self, obj_type: str) -> _ObstacleEntry:
        """Find existing obstacle or create new one."""
        for o in self._obstacles:
            if o.object_type == obj_type:
                return o
        # Determine direction relative to agent
        direction = "ahead"  # default
        obs = _ObstacleEntry(
            object_type=obj_type,
            direction=direction,
            block_count=0,
            last_block_step=self._step_counter,
        )
        self._obstacles.append(obs)
        return obs

    def _obstacle_suggestion(self, obs: _ObstacleEntry) -> str:
        """Generate a heuristic suggestion for avoiding obstacle."""
        if obs.direction in ("ahead", "ahead-left", "ahead-right"):
            return "MoveLeft or MoveBack — do NOT retry forward."
        elif obs.direction in ("left", "right"):
            return "MoveBack then Rotate to find clear path."
        return "MoveBack to create space."

    def _age_entries(self):
        """Remove entries not seen for AGING_THRESHOLD steps."""
        threshold = self._step_counter - self.AGING_THRESHOLD
        for otype in list(self._objects.keys()):
            self._objects[otype] = [
                e for e in self._objects[otype]
                if e.last_seen_step > threshold
            ]
            if not self._objects[otype]:
                del self._objects[otype]

    def _update_area_label(self, metadata: dict):
        """Derive area label from scene name or observed landmarks."""
        # AI2-THOR metadata may have sceneName; use it if available
        scene = metadata.get("sceneName", "")
        if scene:
            # sceneName is like "FloorPlan307" — map to simpler label
            self._area_label = self._scene_to_area(scene)
            return

        # Fallback: derive from what agent has seen
        if not self._objects:
            self._area_label = "unknown room"
            return

        # Simple heuristic: count furniture types
        all_types = list(self._objects.keys())
        bedroom_furniture = {"Bed", "Pillow", "Nightstand", "Dresser", "AlarmClock"}
        kitchen_furniture = {"Fridge", "Microwave", "StoveBurner", "Sink", "CounterTop"}
        living_furniture = {"Sofa", "CoffeeTable", "Television", "Book"}
        office_furniture = {"Desk", "Laptop", "Chair", "DeskLamp", "SideTable"}

        scores = {
            "bedroom": len(set(all_types) & bedroom_furniture),
            "kitchen": len(set(all_types) & kitchen_furniture),
            "living room": len(set(all_types) & living_furniture),
            "office": len(set(all_types) & office_furniture),
        }
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            self._area_label = best
        else:
            self._area_label = "room"

    def _scene_to_area(self, scene: str) -> str:
        """Map AI2-THOR scene name to a human-readable area label."""
        # Extract floor plan number for mapping
        import re
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
        """Render the OBJECTS block as a list of lines."""
        # Collect all entries, sorted by importance
        all_entries: list[_ObjectEntry] = []
        for entries in self._objects.values():
            all_entries.extend(entries)

        # Sort: held > task_receptacle > other_receptacle > regular > aged
        def _sort_key(e: _ObjectEntry) -> tuple:
            # Priority (lower = shown first)
            if e.status == "held":
                prio = 0
            elif e.is_task_receptacle:
                prio = 1
            elif e.is_receptacle:
                prio = 2
            else:
                prio = 3
            # Recency (higher = shown first among same priority)
            recency = -e.last_seen_step
            return (prio, recency)

        sorted_entries = sorted(all_entries, key=_sort_key)
        lines: list[str] = []

        for e in sorted_entries[: self.MAX_OBJECTS]:
            line = self._render_object_line(e)
            if line:
                lines.append(line)

        return lines

    def _render_object_line(self, e: _ObjectEntry) -> str:
        """Render one object entry line."""
        # Status text
        if e.status == "held":
            status_text = "held in hand"
        elif e.status == "placed":
            status_text = "placed (hand empty now)"
        elif e.status == "visible":
            d = e.egocentric_dist
            status_text = f"in view {e.egocentric_dir} {d:.1f}m"
        else:
            age = self._step_counter - e.last_seen_step
            d = e.egocentric_dist
            status_text = f"saw {e.egocentric_dir} ~{d:.1f}m ~{age} steps ago"

        # Tags
        tags = []
        if e.is_task_receptacle:
            tags.append("receptacle ✓")
        elif e.is_receptacle:
            tags.append("NOT task target")

        line = f"  {e.object_type} — {status_text}"
        if tags:
            line += f" ({', '.join(tags)})"
        return line
