def check_task_complete(metadata: dict, ep_data: dict, task_state: dict | None = None):
    """Returns (is_complete: bool, reason_if_incomplete: str). reason is '' when complete."""
    task_type = _official_task_type(ep_data)
    checker = _CHECKERS.get(task_type)
    if checker is None:
        return False, f"Unknown task type: {task_type}"
    return checker(
        metadata,
        ep_data.get("pddl_params", {}) or {},
        _normalize_task_state(task_state),
        _goal_instances(ep_data),
    )


def get_completion_criteria_text(task_type: str, pddl_params: dict) -> str:
    """Human-readable list of what this task requires for completion."""
    pddl_params = pddl_params or {}
    targets = _targets(pddl_params)
    obj = targets["object"]
    parent = targets["parent"]
    toggle = targets["toggle"]
    mrecep = targets["mrecep"]

    resolved_task_type = _official_task_type({
        "task_type": task_type,
        "pddl_params": pddl_params,
    })
    rules = _CRITERIA_RULES.get(resolved_task_type, [])
    lines = []
    # Prepend slicing prerequisite when pddl_params says the object must be sliced.
    # Without this, the agent only sees "put X in Y" and discovers the slicing
    # requirement when Done is rejected — wasting steps on recovery.
    if pddl_params.get("object_sliced"):
        if resolved_task_type == "pick_two_obj_and_place":
            lines.append(f"  - Two {obj} must be sliced")
        else:
            lines.append(f"  - {obj} must be sliced")
    for rule in rules:
        lines.append(f"  - {rule.format(object=obj, parent=parent, toggle=toggle, mrecep=mrecep)}")
    return "\n".join(lines)


def check_unrecoverable(metadata: dict, ep_data: dict) -> str | None:
    pddl = ep_data.get("pddl_params", {}) or {}
    objects = list(metadata.get("objects", []))
    goal = _goal_instances(ep_data)
    final_put = _final_put(goal)

    target = pddl.get("object_target", "")
    if target:
        instances = [o for o in objects if o["objectType"] == target]
        instances = _restrict_by_ids(
            instances,
            [final_put["objectId"]] if final_put else (goal.get("pickup_object_ids") or []),
        )
        if instances and all(o.get("isBroken") for o in instances):
            return f"All target instances of '{target}' are broken"

    toggle = pddl.get("toggle_target", "")
    if toggle:
        instances = [o for o in objects if o["objectType"] == toggle]
        instances = _restrict_by_ids(instances, goal.get("toggle_on_object_ids") or [])
        if instances and all(o.get("isBroken") for o in instances):
            return f"Toggle target '{toggle}' is broken"

    return None


def _make_hashable(v):
    """Convert dict/list to a hashable form for dead-loop detection."""
    if isinstance(v, dict):
        return tuple(sorted((k, _make_hashable(v2)) for k, v2 in v.items()))
    if isinstance(v, list):
        return tuple(_make_hashable(i) for i in v)
    return v


def detect_dead_loop(action_history: list[dict], window: int = 10, threshold: int = 5) -> bool:
    # Weak signal: same (action, params) failing ≥3 times in last 8 steps
    if len(action_history) >= 8:
        recent_fails: dict[tuple, int] = {}
        for s in action_history[-8:]:
            if not s.get("success"):
                params = _make_hashable(s.get("action_params") or {})
                key = (s.get("action"), params)
                recent_fails[key] = recent_fails.get(key, 0) + 1
        if recent_fails and max(recent_fails.values()) >= 3:
            return True

    if len(action_history) < window:
        return False

    fail_counts: dict[tuple, int] = {}
    for s in action_history[-window:]:
        if not s.get("success"):
            params = _make_hashable(s.get("action_params") or {})
            key = (s.get("action"), params)
            fail_counts[key] = fail_counts.get(key, 0) + 1

    if not fail_counts:
        return False

    max_count = max(fail_counts.values())
    if max_count >= threshold:
        return True

    return False


# ------------------------------------------------------------------
# 内部
# ------------------------------------------------------------------

def _official_task_type(ep_data: dict) -> str:
    raw = ep_data.get("alfred_task_type") or ep_data.get("task_type", "")
    if raw == "pick_and_place" and (ep_data.get("pddl_params", {}) or {}).get("mrecep_target"):
        return "pick_and_place_with_movable_recep"
    return _TASK_ALIASES.get(raw, raw)


def _normalize_task_state(task_state: dict | None) -> dict:
    task_state = task_state or {}
    return {
        "cleaned_objects": set(task_state.get("cleaned_objects") or []),
        "heated_objects": set(task_state.get("heated_objects") or []),
        "cooled_objects": set(task_state.get("cooled_objects") or []),
    }


def _goal_instances(ep_data: dict) -> dict:
    """Instance-level goal references from the gold plan (empty when unknown)."""
    goal = ep_data.get("goal_instances")
    return goal if isinstance(goal, dict) else {}


def _restrict_by_ids(objects: list[dict], object_ids: list[str]) -> list[dict]:
    """Keep only the given concrete instances (no-op when ids are missing)."""
    ids = [i for i in (object_ids or []) if i]
    if not ids:
        return objects
    id_set = set(ids)
    return [o for o in objects if o.get("objectId") in id_set]


def _final_put(goal: dict) -> dict | None:
    """The last PutObject of the gold plan: exact target object and goal container."""
    final_put = goal.get("final_put")
    if isinstance(final_put, dict) and final_put.get("objectId") and final_put.get("receptacleObjectId"):
        return final_put
    return None


def _targets(pddl: dict) -> dict:
    # NOTE: We intentionally do NOT append "Sliced" to the object name.
    # Criteria text and prompts use the original name (e.g. "Tomato"),
    # while completion checks use pddl.get("object_sliced") + _sliced_count()
    # to verify slicing state independently.
    return {
        "object": pddl.get("object_target") or "",
        "parent": pddl.get("parent_target") or "",
        "toggle": pddl.get("toggle_target") or "",
        "mrecep": pddl.get("mrecep_target") or "",
    }


def _objects_with_name_and_prop(name: str, prop: str, metadata: dict) -> list[dict]:
    if not name:
        return []
    return [
        obj for obj in metadata.get("objects", [])
        if obj.get("objectType") == name and obj.get(prop)
    ]


def _target_pickupables(metadata: dict, target: str, *, sliced: bool) -> list[dict]:
    """Return the concrete target instances that can satisfy this task."""
    if not target:
        return []
    expected_type = f"{target}Sliced" if sliced else target
    return [
        obj for obj in metadata.get("objects", [])
        if obj.get("pickupable") and obj.get("objectType") == expected_type
    ]


def _contains(receptacle: dict, object_id: str) -> bool:
    return object_id in (receptacle.get("receptacleObjectIds") or [])


def _sliced_count(pickupables: list[dict]) -> int:
    return len([obj for obj in pickupables if "Sliced" in obj.get("objectId", "")])


def _held_object_type(metadata: dict) -> str:
    inv = metadata.get("inventoryObjects") or []
    if inv:
        return inv[0].get("objectType", "")
    return ""


# ------------------------------------------------------------------
# 各任务检查器 → (bool, reason)
# ------------------------------------------------------------------

def _pick_and_place_simple(metadata: dict, pddl: dict, task_state: dict, goal: dict) -> tuple[bool, str]:
    targets = _targets(pddl)
    receptacles = _objects_with_name_and_prop(targets["parent"], "receptacle", metadata)
    pickupables = _target_pickupables(
        metadata, targets["object"], sliced=bool(pddl.get("object_sliced"))
    )

    if pddl.get("object_sliced") and not pickupables:
        return False, f"{targets['object']} must be sliced before placing"

    final_put = _final_put(goal)
    if final_put:
        receptacles = _restrict_by_ids(receptacles, [final_put["receptacleObjectId"]])
        pickupables = _restrict_by_ids(pickupables, [final_put["objectId"]])

    if not receptacles:
        return False, f"No {targets['parent']} is visible — move to find it"

    if any(_contains(r, p["objectId"]) for p in pickupables for r in receptacles):
        return True, ""

    held = _held_object_type(metadata)
    if held:
        return False, f"{targets['object']} is in hand. Place it inside {targets['parent']}"
    return False, f"{targets['object']} must be inside {targets['parent']}"


def _pick_two(metadata: dict, pddl: dict, task_state: dict, goal: dict) -> tuple[bool, str]:
    targets = _targets(pddl)
    receptacles = _objects_with_name_and_prop(targets["parent"], "receptacle", metadata)
    pickupables = _target_pickupables(
        metadata, targets["object"], sliced=bool(pddl.get("object_sliced"))
    )

    if pddl.get("object_sliced") and len(pickupables) < 2:
        return False, f"Two {targets['object']} must be sliced before placing"

    put_ids = list(dict.fromkeys(goal.get("put_receptacle_ids") or []))
    if put_ids:
        receptacles = _restrict_by_ids(receptacles, put_ids)
    pickup_ids = list(dict.fromkeys(goal.get("pickup_object_ids") or []))
    if pickup_ids:
        pickupables = _restrict_by_ids(pickupables, pickup_ids)

    if not receptacles:
        return False, f"No {targets['parent']} is visible — move to find it"

    max_in = 0
    for r in receptacles:
        count = sum(1 for p in pickupables if _contains(r, p["objectId"]))
        max_in = max(max_in, count)

    if max_in >= 2:
        return True, ""

    in_hand = sum(1 for p in pickupables if p.get("isPickedUp"))
    remaining = 2 - max_in
    if in_hand > 0:
        return False, f"Need {remaining} more {targets['object']} inside {targets['parent']}. {in_hand} in hand — place it."
    return False, f"Need {remaining} more {targets['object']} inside {targets['parent']}"


def _look_at_obj_in_light(metadata: dict, pddl: dict, task_state: dict, goal: dict) -> tuple[bool, str]:
    targets = _targets(pddl)
    toggleables = _objects_with_name_and_prop(targets["toggle"], "toggleable", metadata)
    pickupables = _target_pickupables(
        metadata, targets["object"], sliced=bool(pddl.get("object_sliced"))
    )
    inventory = metadata.get("inventoryObjects") or []

    if pddl.get("object_sliced") and not pickupables:
        return False, f"{targets['object']} must be sliced"

    pickup_ids = {p["objectId"] for p in pickupables}
    goal_pickup_ids = set(goal.get("pickup_object_ids") or [])
    if goal_pickup_ids:
        pickup_ids &= goal_pickup_ids
    in_hand = inventory and inventory[0].get("objectId") in pickup_ids

    lamp_ids = set(goal.get("toggle_on_object_ids") or [])
    lamp_on = any(
        o.get("isToggled") and o.get("visibleBounds2D")
        and (not lamp_ids or o.get("objectId") in lamp_ids)
        for o in toggleables
    )

    missing = []
    if not in_hand:
        if not pickupables:
            missing.append(f"{targets['object']} not found — look for it")
        else:
            missing.append(f"{targets['object']} must be picked up (in hand)")
    if not lamp_on:
        if not toggleables:
            missing.append(f"No {targets['toggle']} found")
        else:
            missing.append(f"{targets['toggle']} must be turned on and visible")
    if missing:
        return False, ". ".join(missing)
    return True, ""


def _pick_heat_then_place(metadata: dict, pddl: dict, task_state: dict, goal: dict) -> tuple[bool, str]:
    return _state_then_place(metadata, pddl, task_state["heated_objects"], "heated", goal)


def _pick_cool_then_place(metadata: dict, pddl: dict, task_state: dict, goal: dict) -> tuple[bool, str]:
    return _state_then_place(metadata, pddl, task_state["cooled_objects"], "cooled", goal)


def _pick_clean_then_place(metadata: dict, pddl: dict, task_state: dict, goal: dict) -> tuple[bool, str]:
    return _state_then_place(metadata, pddl, task_state["cleaned_objects"], "cleaned", goal)


def _state_then_place(
    metadata: dict,
    pddl: dict,
    state_object_ids: set[str],
    state_name: str,
    goal: dict,
) -> tuple[bool, str]:
    targets = _targets(pddl)
    receptacles = _objects_with_name_and_prop(targets["parent"], "receptacle", metadata)
    pickupables = _target_pickupables(
        metadata, targets["object"], sliced=bool(pddl.get("object_sliced"))
    )

    if pddl.get("object_sliced") and not pickupables:
        return False, f"{targets['object']} must be sliced before placing"

    final_put = _final_put(goal)
    if final_put:
        receptacles = _restrict_by_ids(receptacles, [final_put["receptacleObjectId"]])
        pickupables = _restrict_by_ids(pickupables, [final_put["objectId"]])

    objs_in_place = [
        p["objectId"] for p in pickupables
        for r in receptacles if _contains(r, p["objectId"])
    ]
    objs_with_state = [p["objectId"] for p in pickupables if p["objectId"] in state_object_ids]
    in_place_and_state = [oid for oid in objs_in_place if oid in state_object_ids]

    missing = []
    if not objs_in_place:
        held = _held_object_type(metadata)
        if held:
            missing.append(f"{targets['object']} is in hand — place it inside {targets['parent']}")
        elif not receptacles:
            missing.append(f"No {targets['parent']} is visible — move to find it")
        else:
            missing.append(f"{targets['object']} must be inside {targets['parent']}")
    if not objs_with_state:
        missing.append(f"{targets['object']} must be {state_name}")
    elif not in_place_and_state:
        missing.append(f"{targets['object']} is {state_name} but not inside {targets['parent']}")

    if missing:
        return False, ". ".join(missing)
    return True, ""


def _pick_and_place_with_movable_recep(
    metadata: dict,
    pddl: dict,
    task_state: dict,
    goal: dict,
) -> tuple[bool, str]:
    targets = _targets(pddl)
    receptacles = _objects_with_name_and_prop(targets["parent"], "receptacle", metadata)
    pickupables = _target_pickupables(
        metadata, targets["object"], sliced=bool(pddl.get("object_sliced"))
    )
    movables = _objects_with_name_and_prop(targets["mrecep"], "pickupable", metadata)

    if pddl.get("object_sliced") and not pickupables:
        return False, f"{targets['object']} must be sliced before placing"

    final_put = _final_put(goal)
    if final_put:
        receptacles = _restrict_by_ids(receptacles, [final_put["receptacleObjectId"]])
    movable_id = goal.get("movable_receptacle_id")
    if movable_id:
        movables = _restrict_by_ids(movables, [movable_id])
    movable_target_id = goal.get("movable_target_object_id")
    if movable_target_id:
        pickupables = _restrict_by_ids(pickupables, [movable_target_id])

    target_ids = {obj["objectId"] for obj in pickupables}
    complete_stack = any(
        target_ids.intersection(movable.get("receptacleObjectIds") or [])
        and any(
            movable["objectId"] in (parent.get("receptacleObjectIds") or [])
            for parent in receptacles
        )
        for movable in movables
    )

    missing = []
    if not any(
        target_ids.intersection(movable.get("receptacleObjectIds") or [])
        for movable in movables
    ):
        missing.append(f"{targets['object']} must be inside {targets['mrecep']}")
    if not any(
        movable["objectId"] in (parent.get("receptacleObjectIds") or [])
        for movable in movables
        for parent in receptacles
    ):
        missing.append(f"{targets['mrecep']} must be inside {targets['parent']}")
    if complete_stack:
        return True, ""
    return False, ". ".join(missing)


_TASK_ALIASES = {
    "pick_and_place": "pick_and_place_simple",
    "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
    "examine": "look_at_obj_in_light",
    "pick_two": "pick_two_obj_and_place",
}

_CHECKERS = {
    "pick_and_place_simple": _pick_and_place_simple,
    "pick_two_obj_and_place": _pick_two,
    "look_at_obj_in_light": _look_at_obj_in_light,
    "pick_heat_then_place_in_recep": _pick_heat_then_place,
    "pick_cool_then_place_in_recep": _pick_cool_then_place,
    "pick_clean_then_place_in_recep": _pick_clean_then_place,
    "pick_and_place_with_movable_recep": _pick_and_place_with_movable_recep,
}

_CRITERIA_RULES = {
    "pick_and_place_simple": ["{object} must be inside {parent}"],
    "pick_two_obj_and_place": ["Two {object} must be inside {parent}"],
    "look_at_obj_in_light": ["{object} must be picked up (in hand)", "{toggle} must be turned on and visible"],
    "pick_heat_then_place_in_recep": ["{object} must be heated", "{object} must be inside {parent}"],
    "pick_cool_then_place_in_recep": ["{object} must be cooled", "{object} must be inside {parent}"],
    "pick_clean_then_place_in_recep": ["{object} must be cleaned", "{object} must be inside {parent}"],
    "pick_and_place_with_movable_recep": ["{object} must be inside {mrecep}", "{mrecep} must be inside {parent}"],
}
