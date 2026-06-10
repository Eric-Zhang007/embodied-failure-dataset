def check_task_complete(metadata: dict, ep_data: dict, task_state: dict | None = None):
    """Returns (is_complete: bool, reason_if_incomplete: str). reason is '' when complete."""
    task_type = _official_task_type(ep_data)
    checker = _CHECKERS.get(task_type)
    if checker is None:
        return False, f"Unknown task type: {task_type}"
    return checker(metadata, ep_data.get("pddl_params", {}) or {}, _normalize_task_state(task_state))


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
    for rule in rules:
        lines.append(f"  - {rule.format(object=obj, parent=parent, toggle=toggle, mrecep=mrecep)}")
    return "\n".join(lines)


def check_unrecoverable(metadata: dict, ep_data: dict) -> str | None:
    pddl = ep_data.get("pddl_params", {}) or {}
    objects = list(metadata.get("objects", []))

    target = pddl.get("object_target", "")
    if target:
        instances = [o for o in objects if o["objectType"] == target]
        if instances and all(o.get("isBroken") for o in instances):
            return f"All instances of target object '{target}' are broken"

    toggle = pddl.get("toggle_target", "")
    if toggle:
        instances = [o for o in objects if o["objectType"] == toggle]
        if instances and all(o.get("isBroken") for o in instances):
            return f"Toggle target '{toggle}' is broken"

    return None


def detect_dead_loop(action_history: list[dict], window: int = 10, threshold: int = 5) -> bool:
    if len(action_history) < window:
        return False

    fail_counts: dict[tuple, int] = {}
    for s in action_history[-window:]:
        if not s.get("success"):
            key = (s.get("action"), frozenset((s.get("action_params") or {}).items()))
            fail_counts[key] = fail_counts.get(key, 0) + 1

    if not fail_counts:
        return False

    max_count = max(fail_counts.values())
    if max_count >= threshold:
        return True

    if len(action_history) >= 8:
        recent_actions = [s.get("action", "") for s in action_history[-8:]]
        if len(set(recent_actions)) <= 2 and max_count >= 2:
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


def _targets(pddl: dict) -> dict:
    target_object = pddl.get("object_target") or ""
    if pddl.get("object_sliced") and target_object:
        target_object += "Sliced"
    return {
        "object": target_object,
        "parent": pddl.get("parent_target") or "",
        "toggle": pddl.get("toggle_target") or "",
        "mrecep": pddl.get("mrecep_target") or "",
    }


def _objects_with_name_and_prop(name: str, prop: str, metadata: dict) -> list[dict]:
    if not name:
        return []
    return [
        obj for obj in metadata.get("objects", [])
        if name in obj.get("objectId", "") and obj.get(prop)
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

def _pick_and_place_simple(metadata: dict, pddl: dict, task_state: dict) -> tuple[bool, str]:
    targets = _targets(pddl)
    receptacles = _objects_with_name_and_prop(targets["parent"], "receptacle", metadata)
    pickupables = _objects_with_name_and_prop(targets["object"], "pickupable", metadata)

    if "Sliced" in targets["object"] and _sliced_count(pickupables) < 1:
        return False, f"{targets['object']} must be sliced before placing"

    if not receptacles:
        return False, f"No {targets['parent']} is visible — move to find it"

    if any(_contains(r, p["objectId"]) for p in pickupables for r in receptacles):
        return True, ""

    held = _held_object_type(metadata)
    if held:
        return False, f"{targets['object']} is in hand. Place it inside {targets['parent']}"
    return False, f"{targets['object']} must be inside {targets['parent']}"


def _pick_two(metadata: dict, pddl: dict, task_state: dict) -> tuple[bool, str]:
    targets = _targets(pddl)
    receptacles = _objects_with_name_and_prop(targets["parent"], "receptacle", metadata)
    pickupables = _objects_with_name_and_prop(targets["object"], "pickupable", metadata)

    if "Sliced" in targets["object"] and _sliced_count(pickupables) < 2:
        return False, f"Two {targets['object']} must be sliced before placing"

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


def _look_at_obj_in_light(metadata: dict, pddl: dict, task_state: dict) -> tuple[bool, str]:
    targets = _targets(pddl)
    toggleables = _objects_with_name_and_prop(targets["toggle"], "toggleable", metadata)
    pickupables = _objects_with_name_and_prop(targets["object"], "pickupable", metadata)
    inventory = metadata.get("inventoryObjects") or []

    if "Sliced" in targets["object"] and _sliced_count(pickupables) < 1:
        return False, f"{targets['object']} must be sliced"

    pickup_ids = {p["objectId"] for p in pickupables}
    in_hand = inventory and inventory[0].get("objectId") in pickup_ids
    lamp_on = any(o.get("isToggled") and o.get("visibleBounds2D") for o in toggleables)

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


def _pick_heat_then_place(metadata: dict, pddl: dict, task_state: dict) -> tuple[bool, str]:
    return _state_then_place(metadata, pddl, task_state["heated_objects"], "heated")


def _pick_cool_then_place(metadata: dict, pddl: dict, task_state: dict) -> tuple[bool, str]:
    return _state_then_place(metadata, pddl, task_state["cooled_objects"], "cooled")


def _pick_clean_then_place(metadata: dict, pddl: dict, task_state: dict) -> tuple[bool, str]:
    return _state_then_place(metadata, pddl, task_state["cleaned_objects"], "cleaned")


def _state_then_place(metadata: dict, pddl: dict, state_object_ids: set[str], state_name: str) -> tuple[bool, str]:
    targets = _targets(pddl)
    receptacles = _objects_with_name_and_prop(targets["parent"], "receptacle", metadata)
    pickupables = _objects_with_name_and_prop(targets["object"], "pickupable", metadata)

    if "Sliced" in targets["object"] and _sliced_count(pickupables) < 1:
        return False, f"{targets['object']} must be sliced before placing"

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


def _pick_and_place_with_movable_recep(metadata: dict, pddl: dict, task_state: dict) -> tuple[bool, str]:
    targets = _targets(pddl)
    receptacles = _objects_with_name_and_prop(targets["parent"], "receptacle", metadata)
    pickupables = _objects_with_name_and_prop(targets["object"], "pickupable", metadata)
    movables = _objects_with_name_and_prop(targets["mrecep"], "pickupable", metadata)

    if "Sliced" in targets["object"] and _sliced_count(pickupables) < 1:
        return False, f"{targets['object']} must be sliced before placing"

    pickup_in_movable = [
        p for p in pickupables
        for m in movables
        if p["objectId"] in (m.get("receptacleObjectIds") or [])
    ]
    movable_in_parent = [
        m for m in movables
        for r in receptacles
        if m["objectId"] in (r.get("receptacleObjectIds") or [])
    ]

    missing = []
    if not pickup_in_movable:
        missing.append(f"{targets['object']} must be inside {targets['mrecep']}")
    if not movable_in_parent:
        missing.append(f"{targets['mrecep']} must be inside {targets['parent']}")
    if pickup_in_movable and movable_in_parent:
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
