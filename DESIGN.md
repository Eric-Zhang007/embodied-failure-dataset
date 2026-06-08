# DESIGN: Egocentric Compressed Memory

## 1. Data Structure

### Core: `EgoMemory` dataclass

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LandmarkEntry:
    """A single landmark the agent has observed."""
    object_type: str          # e.g. "Desk", "AlarmClock", "SideTable"
    last_direction: str       # egocentric direction when last seen, e.g. "ahead-right"
    last_distance: float      # distance in meters when last seen
    step_seen: int            # step index when last observed
    cumulative_rotation: float # agent's cumulative yaw rotation WHEN this was recorded
    is_target: bool = False   # True if this type matches task target object/receptacle
    receptacle: bool = False  # True if this object is a receptacle
    instances: int = 1        # how many distinct instances seen (capped at 2)

@dataclass 
class EgoMemory:
    """First-person egocentric memory. Only contains what the agent has observed."""
    # Landmarks: key objects seen, deduplicated by type (only receptacles + task target)
    landmarks: dict[str, LandmarkEntry] = field(default_factory=dict)  
    # "Desk" -> LandmarkEntry, "AlarmClock" -> LandmarkEntry, etc.
    
    # Task progress
    task_target_object: str = ""     # e.g. "AlarmClock"
    task_target_receptacle: str = "" # e.g. "Desk"
    picked_up: bool = False          # have we successfully picked up the target?
    placed: bool = False             # have we successfully placed the target?
    
    # Recent failure context (compact, last 2 failures only)
    last_failure_step: int = -1
    last_failure_diagnosis: str = ""     # 1 sentence
    last_failure_recovery: str = ""      # 1 sentence
    prev_failure_step: int = -1
    prev_failure_diagnosis: str = ""
    
    # Navigation state
    steps_since_last_rotation: int = 0     # reset on RotateLeft/Right
    consecutive_same_direction_moves: int = 0  # detect wall-humping
    cumulative_yaw: float = 0.0           # tracks net rotation from start
    current_step: int = 0
```

### Why this structure:
- **Landmarks dict**: Keyed by objectType, not objectId. This naturally deduplicates multiple instances (two AlarmClocks → one entry "AlarmClock: seen ahead-left ~0.5m"). We cap `instances` at 2 to signal multiplicity to the VLM.
- **Task progress**: Booleans extracted from environment state, not VLM guesswork. Authoritative.
- **Failure context**: Compact; only last 2 failures. Replaces verbose JSON history + RECENT FAILURE INSIGHT.
- **Navigation state**: Tracks cumulative rotation for direction correction, and detects when agent is bumping the same wall.

---

## 2. Update Algorithm

### `update_memory(memory: EgoMemory, step_data: dict) -> EgoMemory`

Called every step by `BranchRunner.run()` after Phase 1 action is recorded.

```python
def update_memory(memory: EgoMemory, step_data: dict) -> EgoMemory:
    """
    step_data keys used:
      - visible_objects: list[dict]
      - inventory_objects: list[dict]
      - agent_pos, agent_rot_y
      - action: str
      - success: bool
      - error_message: str | None
      - eb_diagnosis: str | None
      - eb_recovery_reasoning: str | None
      - task_criteria: str
      - pddl_params: dict
    """
```

#### Step 2a: Track cumulative rotation

```python
action = step_data["action"]
if action in ("RotateLeft",):
    memory.cumulative_yaw += 90
    memory.steps_since_last_rotation = 0
elif action in ("RotateRight",):
    memory.cumulative_yaw -= 90
    memory.steps_since_last_rotation = 0
else:
    memory.steps_since_last_rotation += 1

# Detect wall-humping: same Move action 3+ times without rotation
if action in ("MoveAhead","MoveBack","MoveLeft","MoveRight"):
    if step_data.get("last_action") == action:
        memory.consecutive_same_direction_moves += 1
    else:
        memory.consecutive_same_direction_moves = 0
else:
    memory.consecutive_same_direction_moves = 0
```

#### Step 2b: Update landmarks from visible objects

Only track objects that are: (a) receptacles, OR (b) the task target object type, OR (c) objects that have been interacted with (picked up, opened, etc.). This keeps the landmark list small.

```python
visible = [o for o in step_data["visible_objects"] if o.get("visible")]
agent_pos = step_data["agent_pos"]
agent_rot_y = step_data["agent_rot_y"]

for obj in visible:
    otype = obj["objectType"]
    is_receptacle = obj.get("receptacle", False)
    is_target = (otype == memory.task_target_object or otype == memory.task_target_receptacle)
    is_interacted = obj.get("isPickedUp", False)
    
    if not (is_receptacle or is_target or is_interacted):
        continue
    
    # Compute egocentric direction (uses existing _direction from eb_agent.py)
    direction = _direction(agent_pos, agent_rot_y, obj["position"])
    distance = _compute_distance(agent_pos, obj["position"])
    
    existing = memory.landmarks.get(otype)
    if existing is None:
        memory.landmarks[otype] = LandmarkEntry(
            object_type=otype,
            last_direction=direction,
            last_distance=round(distance, 1),
            step_seen=memory.current_step,
            cumulative_rotation=memory.cumulative_yaw,
            is_target=is_target,
            receptacle=is_receptacle,
            instances=1,
        )
    else:
        # Update if newly seen, and detect multiple instances
        existing.last_direction = direction
        existing.last_distance = round(distance, 1)
        existing.step_seen = memory.current_step
        existing.cumulative_rotation = memory.cumulative_yaw
        # If previously unseen for > 5 steps and distance differs significantly, count as new instance
        if (memory.current_step - existing.step_seen > 5 and 
            abs(distance - existing.last_distance) > 1.0):
            existing.instances = min(existing.instances + 1, 2)
```

#### Step 2c: Update task progress from environment

```python
inventory = step_data.get("inventory_objects", [])
if inventory and any(o.get("objectType") == memory.task_target_object for o in inventory):
    memory.picked_up = True

# Check placement by looking at the task checker result
if step_data.get("task_complete_check"):
    memory.placed = True
```

#### Step 2d: Update failure context

```python
if not step_data.get("success", True):
    # Shift old failure
    if memory.last_failure_step >= 0:
        memory.prev_failure_step = memory.last_failure_step
        memory.prev_failure_diagnosis = memory.last_failure_diagnosis
    memory.last_failure_step = memory.current_step
    memory.last_failure_diagnosis = (step_data.get("eb_diagnosis") or 
                                      step_data.get("error_message") or "")[:200]
    memory.last_failure_recovery = (step_data.get("eb_recovery_reasoning") or "")[:200]
```

#### Step 2e: Expire stale landmarks

Landmarks not re-observed for > 8 steps get a `(stale)` tag. Landmarks not seen for > 20 steps are dropped.

---

## 3. Prompt Integration

### New function: `build_ego_memory_text(memory: EgoMemory, current_rot_y: float) -> str`

Renders the memory to ~500 chars of text. Converts stored directions (which are relative to the agent's rotation at the time of observation) back to current egocentric frame using cumulative rotation.

```python
def build_ego_memory_text(memory: EgoMemory, current_rot_y: float) -> str:
    """Render egocentric memory as compact text for VLM prompt."""
    lines = ["--- EGO MEMORY ---"]
    
    # Progress bar
    progress_parts = []
    if memory.picked_up:
        progress_parts.append(f"PICKED {memory.task_target_object}")
    else:
        progress_parts.append(f"need {memory.task_target_object}")
    if memory.placed:
        progress_parts.append(f"PLACED in {memory.task_target_receptacle}")
    else:
        progress_parts.append(f"need {memory.task_target_receptacle}")
    lines.append("Progress: " + " | ".join(progress_parts))
    
    # Landmarks with direction correction
    if memory.landmarks:
        lines.append("Landmarks seen (direction from YOUR current facing):")
        for otype, entry in memory.landmarks.items():
            # Correct direction for agent's rotation since observation
            rotation_delta = (memory.cumulative_yaw - entry.cumulative_rotation) % 360
            corrected_dir = _adjust_direction(entry.last_direction, rotation_delta)
            
            age = memory.current_step - entry.step_seen
            stale = " (stale)" if age > 8 else ""
            multi = f" x{entry.instances}" if entry.instances > 1 else ""
            target_tag = " ★TARGET" if entry.is_target else ""
            lines.append(
                f"  {otype}{multi}{target_tag}: seen {corrected_dir} ~{entry.last_distance:.1f}m "
                f"({age} steps ago){stale}"
            )
    
    # Hand status (prominent, not buried)
    if memory.picked_up and not memory.placed:
        lines.append(f"HOLDING: {memory.task_target_object} — find {memory.task_target_receptacle} to place it")
    elif memory.placed:
        lines.append("HAND: empty (object placed) — call Done if criteria met")
    else:
        lines.append("HAND: empty — need to find and pick up target")
    
    # Compact failure context
    if memory.last_failure_step >= 0:
        lines.append(f"Last failure (step {memory.last_failure_step}): {memory.last_failure_diagnosis}")
        if memory.last_failure_recovery:
            lines.append(f"  Recovery hint: {memory.last_failure_recovery}")
    
    # Navigation warning
    if memory.consecutive_same_direction_moves >= 3:
        lines.append("⚠️ You've moved the same direction 3+ times. ROTATE to find a new path.")
    
    return "\n".join(lines)
```

### Helper: `_adjust_direction(direction: str, delta_deg: float) -> str`

Converts an egocentric direction string ("ahead-left (1.3m)") to account for agent rotation. Uses the quadrant system: ahead/left/behind/right + combinations. For 90° increments (which is what our RotateLeft/Right use), this is straightforward.

```python
_DIR_MAP = {
    "ahead": 0, "ahead-right": 45, "right": 90, "behind-right": 135,
    "behind": 180, "behind-left": 225, "left": 270, "ahead-left": 315,
}
_DIR_REVERSE = {v: k for k, v in _DIR_MAP.items()}

def _adjust_direction(direction: str, delta_deg: float) -> str:
    """Adjust egocentric direction by cumulative rotation delta."""
    # Extract base direction and distance
    import re
    match = re.match(r"(\S+(?:-\S+)?)\s*\(([^)]+)\)", direction)
    if not match:
        return direction
    base_dir, dist = match.group(1), match.group(2)
    
    orig_deg = _DIR_MAP.get(base_dir, 0)
    new_deg = (orig_deg - delta_deg) % 360
    
    # Find closest cardinal
    closest = min(_DIR_REVERSE.keys(), key=lambda k: abs((k - new_deg + 180) % 360 - 180))
    new_dir = _DIR_REVERSE[closest]
    return f"{new_dir} ({dist})"
```

### Modified `build_phase1_prompt()`

The key change: replace line 207's `build_eb_history_context(action_history)` call with `build_ego_memory_text(ego_memory, agent_rot_y)`. Also move HAND STATUS to be derived from memory rather than buried in the middle.

New `build_phase1_prompt` signature and body:

```python
def build_phase1_prompt(
    task_goal: str,
    visible_objects: list[dict],
    action_history: list[dict],      # KEPT for backward compat, but not rendered
    last_error: str | None,
    failed_object_ids: set = None,
    agent_pos: dict = None,
    agent_rot_y: float = 0.0,
    inventory_objects: list[dict] | None = None,
    task_criteria: str = "",
    ego_memory: EgoMemory | None = None,  # NEW parameter
) -> str:
    lines = [f"Task goal: {task_goal}\n"]
    
    # NEW: Memory goes FIRST, before visible objects
    # This ensures the VLM reads spatial context before raw object lists
    if ego_memory is not None:
        lines.append(build_ego_memory_text(ego_memory, agent_rot_y))
        lines.append("")
    
    # HAND STATUS + completion criteria (kept compact, after memory)
    _append_task_context(lines, visible_objects, inventory_objects, task_criteria)
    
    # ... rest of visible objects rendering (unchanged lines 169-204) ...
    
    # REMOVED: build_eb_history_context(action_history)  ← was line 207
    
    # KEEP: last_error, recent failure insight (but now memory already has failure context)
    if last_error:
        lines.append(f"\nLast action error: {last_error}")
    
    # ... CRITICAL OBJECT NAMES (keep) ...
    
    return "\n".join(lines)
```

### Estimated prompt size comparison:

| Section | Old (8B test step 9) | New |
|---------|---------------------|-----|
| System prompt | 7738 | 7738 |
| Task goal + hand status + criteria | ~400 | ~400 |
| Visible objects | ~500 | ~500 |
| **History / Memory** | **~4000** (10 JSON steps) | **~400** (ego memory) |
| Failure insight | ~200 | ~150 |
| Footer (obj names rules) | ~400 | ~400 |
| **Total user text** | **~5500** | **~1850** |

---

## 4. File Changes

### NEW: `src/ego_memory.py`
Contains `EgoMemory`, `LandmarkEntry` dataclasses, `update_memory()`, `build_ego_memory_text()`, `_adjust_direction()`, `_compute_distance()`.

```python
# src/ego_memory.py
from dataclasses import dataclass, field
from typing import Optional
import math

@dataclass
class LandmarkEntry: ...

@dataclass
class EgoMemory: ...

def init_ego_memory(task_goal: str, pddl_params: dict, task_criteria: str) -> EgoMemory:
    """Create initial memory from task definition."""
    ...

def update_memory(memory: EgoMemory, step_data: dict) -> EgoMemory:
    """Update memory after each step."""
    ...

def build_ego_memory_text(memory: EgoMemory, current_rot_y: float, max_chars: int = 600) -> str:
    """Render memory as compact text for VLM prompt. Truncates to max_chars."""
    ...
```

### MODIFY: `src/eb_agent.py`
1. Import `EgoMemory`, `build_ego_memory_text` from `src.ego_memory`
2. Add `ego_memory: EgoMemory | None = None` parameter to `build_phase1_prompt()` (line 152)
3. Add memory rendering call before `_append_task_context` (after line 163)
4. Remove or conditionalize `build_eb_history_context()` call on line 207: `if ego_memory is None: lines.append(build_eb_history_context(action_history))`
5. Same changes to `build_phase3_prompt()` (line 297) — add ego_memory param
6. Add `ego_memory` param to `EBAgent.propose_action()` (line 348), `propose_action_lookaround()` (line 371), `diagnose_failure()` (line 420)

### MODIFY: `src/branch_runner.py`
1. Import `init_ego_memory`, `update_memory`, `EgoMemory` from `src.ego_memory`
2. In `BranchRunner.run()` (line 71), after extracting `task_criteria` (line 117):
   ```python
   if step_index == 0:
       ego_memory = init_ego_memory(
           ep.data["task_goal"], 
           ep.data.get("pddl_params", {}), 
           task_criteria
       )
   ```
3. Pass `ego_memory` to `self.eb_agent.propose_action(...)` (line 133) and all other EB calls
4. After each step is recorded (around line 188+), call:
   ```python
   step_data_for_memory = {
       "visible_objects": state["metadata"].get("objects", []),
       "inventory_objects": inventory_objects,
       "agent_pos": agent_pose.get("position"),
       "agent_rot_y": agent_pose.get("rotation", {}).get("y", 0.0),
       "action": proposed_action,
       "success": result["success"],
       "error_message": result.get("error"),
       "eb_diagnosis": eb_phase3.get("diagnosis") if phase3_done else None,
       "eb_recovery_reasoning": eb_phase3.get("recovery_reasoning") if phase3_done else None,
       "pddl_params": ep.data.get("pddl_params", {}),
   }
   ego_memory = update_memory(ego_memory, step_data_for_memory)
   ```

### MODIFY: `src/context_builder.py`
No changes needed — `build_eb_history_context` remains for Oracle (which still needs full history). Only EB switches to compressed memory.

---

## 5. Edge Cases

### 5a: Lost object (agent drops or can't find target)
- **Detection**: `memory.picked_up` was True but `inventory_objects` is empty and target not visible
- **Memory annotation**: Add a "LOST TRACK" note: `"⚠️ Was holding AlarmClock but lost it. Last seen: ahead-left ~0.5m (3 steps ago). Look around."`
- **Implementation**: In `update_memory`, if `memory.picked_up and not inventory_objects and not any target visible`

### 5b: Multiple instances (two AlarmClocks)
- **Detection**: `landmarks["AlarmClock"].instances` increments when a new instance is seen at a significantly different location
- **Memory annotation**: `"AlarmClock x2: seen ahead ~0.5m (2 steps ago)"`
- **VLM guidance**: The "x2" suffix signals there may be more than one. The VLM should still try PickupObject on whichever is closest.

### 5c: Room transitions
- **Detection**: All landmarks become stale (not re-seen for 10+ steps)
- **Memory behavior**: Stale landmarks stay with `(stale)` tag for 20 steps then drop. New room's objects fill in fresh landmarks.
- **No special handling needed** — memory naturally refreshes as agent explores.

### 5d: Agent rotation making old directions wrong
- **Solution**: `_adjust_direction()` corrects stored directions using `cumulative_yaw`. Since RotateLeft/Right are always 90°, correction is exact.
- **Edge**: If agent rotates 360°, direction is back to original. Correctly handled by `% 360`.

### 5e: Wall-humping detection
- **Detection**: `consecutive_same_direction_moves >= 3`
- **Memory annotation**: `"⚠️ You've moved the same direction 3+ times. ROTATE to find a new path."`
- **Reset**: Any rotation or different action resets the counter.

### 5f: Failure recovery loop
- **Detection**: Two consecutive failures with the same diagnosis text
- **Memory annotation**: `"⚠️ Same failure twice. Try a DIFFERENT approach."`
- **Implementation**: Compare `last_failure_diagnosis` with `prev_failure_diagnosis` for substring match.

---

## 6. Example Walkthrough

Task: "move the alarm clock from one desk to another one"

### Step 0 (initial state)
```
--- EGO MEMORY ---
Progress: need AlarmClock | need Desk
Landmarks seen (direction from YOUR current facing):
  AlarmClock ★TARGET: seen ahead-left ~1.3m (0 steps ago)
  SideTable: seen ahead-left ~1.2m (0 steps ago)
  Desk ★TARGET: seen behind-right ~0.9m (0 steps ago)
HAND: empty — need to find and pick up target
```

### Step 3 (moved closer, Desks now behind)
```
--- EGO MEMORY ---
Progress: need AlarmClock | need Desk
Landmarks seen (direction from YOUR current facing):
  AlarmClock ★TARGET: seen ahead-left ~0.9m (0 steps ago)
  SideTable: seen left ~0.8m (3 steps ago, stale)
  Desk ★TARGET: seen behind-right ~2.0m (3 steps ago) ← getting stale!
HAND: empty — need to find and pick up target
```

### Step 5 (picked up AlarmClock)
```
--- EGO MEMORY ---
Progress: PICKED AlarmClock | need Desk
Landmarks seen (direction from YOUR current facing):
  AlarmClock ★TARGET: seen ahead ~0.5m (0 steps ago)
  SideTable: seen ahead-left ~0.7m (0 steps ago)
  Desk ★TARGET: seen behind ~3.0m (5 steps ago, stale) ← critical: Desks are behind!
HOLDING: AlarmClock — find Desk to place it
⚠️ Desks were last seen behind you. Turn around!
```

### Step 7 (wrong: put on SideTable — success but wrong target)
```
--- EGO MEMORY ---
Progress: PICKED AlarmClock | need Desk
Landmarks seen (direction from YOUR current facing):
  SideTable: seen ahead-left ~0.6m (0 steps ago)
  Desk ★TARGET: seen behind ~5m (7 steps ago, stale)
HAND: empty (object placed) — call Done if criteria met
⚠️ Object was placed on SideTable, but task requires: AlarmClock must be inside Desk
```

### Step 8 (Done rejected — "AlarmClock must be inside Desk")
```
--- EGO MEMORY ---
Progress: PICKED AlarmClock | need Desk
Landmarks seen (direction from YOUR current facing):
  SideTable: seen ahead-left ~0.6m (1 step ago)
  Desk ★TARGET: seen behind ~5m (8 steps ago, stale)
  AlarmClock ★TARGET: seen ahead-left ~0.5m (1 step ago)
HAND: empty — need to find and pick up target
Last failure (step 8): Done rejected: AlarmClock must be inside Desk. Fix the unmet criteria before calling Done again.
```

### Step 14 (navigated back toward Desk area, picked up AlarmClock again)
```
--- EGO MEMORY ---
Progress: PICKED AlarmClock | need Desk
Landmarks seen (direction from YOUR current facing):
  Desk ★TARGET: seen ahead-right ~1.5m (0 steps ago) ← Desk found!
  AlarmClock ★TARGET: seen ahead ~0.4m (0 steps ago)
HOLDING: AlarmClock — find Desk to place it
```
