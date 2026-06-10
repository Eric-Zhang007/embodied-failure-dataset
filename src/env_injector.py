from ai2thor.controller import Controller


def inject(controller: Controller, method: str, pddl_params: dict = None, **params) -> dict:
    handler = _INJECTORS.get(method)
    if handler is None:
        return {"success": False, "error": f"Unknown injection method: {method}", "blocked": None}

    blocked = _guard_task_critical(method, params, pddl_params or {})
    if blocked:
        return {"success": False, "error": blocked, "blocked": blocked}

    try:
        if method == "hide_object":
            handler(controller, pddl_params=pddl_params, **params)
        else:
            handler(controller, **params)
        return {"success": True, "error": None, "blocked": None}
    except ValueError as e:
        return {"success": False, "error": str(e), "blocked": None}


_IRREVERSIBLE_PROPS = {"is_static", "is_broken", "is_cooked", "is_sliced", "is_used_up"}
_TARGET_MODIFY_METHODS = {"remove_object", "hide_object", "occlude_object", "swap_object"}


def _guard_task_critical(method: str, params: dict, pddl_params: dict) -> str | None:
    if not pddl_params:
        return None

    target_type = params.get("object_type") or params.get("target_type") or ""
    prop = params.get("property", "")

    critical_types = {
        pddl_params[key]
        for key in ("object_target", "parent_target", "mrecep_target", "toggle_target")
        if pddl_params.get(key)
    }
    if target_type not in critical_types:
        return None

    if method == "set_object_property" and prop in _IRREVERSIBLE_PROPS:
        return f"Blocked: irreversible property {prop!r} on task-critical object {target_type!r}"
    if method in _TARGET_MODIFY_METHODS:
        return f"Blocked: {method} on task-critical object {target_type!r}"
    return None


def _step_or_raise(controller: Controller, action: str, **params):
    event = controller.step(action=action, **params)
    if not event.metadata.get("lastActionSuccess", False):
        error = event.metadata.get("errorMessage") or "unknown error"
        raise ValueError(f"{action} failed: {error}")
    return event


def _place_object_at_point(controller: Controller, obj: dict, position: dict):
    return _step_or_raise(
        controller,
        "PlaceObjectAtPoint",
        objectId=obj["objectId"],
        position={"x": position["x"], "y": position["y"], "z": position["z"]},
    )


def _objects(controller: Controller) -> list[dict]:
    return controller.step(action="Pass").metadata["objects"]


def _find_first(objects: list[dict], object_type: str, *, visible: bool | None = None) -> dict:
    candidates = [obj for obj in objects if obj["objectType"] == object_type]
    if visible is not None:
        candidates = [obj for obj in candidates if bool(obj.get("visible")) == visible]
    if not candidates:
        suffix = " visible" if visible else ""
        raise ValueError(f"No{suffix} {object_type} found")
    candidates.sort(key=lambda o: o["objectId"])
    return candidates[0]


def _set_object_property(controller: Controller, object_type: str, property: str, value):
    obj = _find_first(_objects(controller), object_type)
    _set_prop(controller, obj["objectId"], property, value)


def _set_prop(controller: Controller, object_id: str, prop: str, value):
    if prop == "is_static" and value:
        _step_or_raise(controller, "SetObjectStatic", objectId=object_id, isStatic=True)
    elif prop == "is_broken" and value:
        _step_or_raise(controller, "BreakObject", objectId=object_id, forceAction=True)
    elif prop == "is_open":
        if object_id.startswith("Blinds"):
            raise ValueError("Blinds OpenObject/CloseObject causes AI2-THOR timeout")
        action = "OpenObject" if value else "CloseObject"
        _step_or_raise(controller, action, objectId=object_id, forceAction=True)
    elif prop == "is_toggled":
        action = "ToggleObjectOn" if value else "ToggleObjectOff"
        _step_or_raise(controller, action, objectId=object_id, forceAction=True)
    elif prop == "is_dirty" and value:
        _step_or_raise(controller, "DirtyObject", objectId=object_id, forceAction=True)
    elif prop == "is_cooked" and value:
        _step_or_raise(controller, "CookObject", objectId=object_id, forceAction=True)
    elif prop == "is_sliced" and value:
        _step_or_raise(controller, "SliceObject", objectId=object_id, forceAction=True)
    elif prop == "is_used_up" and value:
        _step_or_raise(controller, "UseUpObject", objectId=object_id, forceAction=True)
    elif prop == "fillLiquid":
        _step_or_raise(controller, "FillObjectWithLiquid", objectId=object_id, fillLiquid=str(value), forceAction=True)
    else:
        raise ValueError(f"Cannot set property {prop}={value} for {object_id}")


def _swap_object(controller: Controller, target_type: str, new_type: str):
    import math

    event = controller.step(action="Pass")
    objects = event.metadata["objects"]
    target = _find_first(objects, target_type, visible=True)
    replacement = _find_first([obj for obj in objects if obj.get("name") != target.get("name")], new_type)

    agent_pos = event.metadata["agent"]["position"]
    agent_rot = event.metadata["agent"]["rotation"]["y"]
    rad = math.radians(agent_rot)
    target_pos = target["position"]
    away_pos = {
        "x": agent_pos["x"] + 2.0 * math.sin(rad),
        "y": target_pos["y"],
        "z": agent_pos["z"] + 2.0 * math.cos(rad),
    }
    _place_object_at_point(controller, target, away_pos)
    _place_object_at_point(controller, replacement, target_pos)


def _occlude_object(controller: Controller, object_type: str):
    event = controller.step(action="Pass")
    target = _find_first(event.metadata["objects"], object_type, visible=True)

    blockers = [
        obj for obj in event.metadata["objects"]
        if obj.get("pickupable") and obj.get("name") != target.get("name")
    ]
    if not blockers:
        raise ValueError("No movable object to use as occluder")
    blockers.sort(key=lambda o: o["objectId"])

    agent_pos = event.metadata["agent"]["position"]
    target_pos = target["position"]
    _place_object_at_point(
        controller,
        blockers[0],
        {
            "x": (target_pos["x"] + agent_pos["x"]) / 2,
            "y": target_pos["y"],
            "z": (target_pos["z"] + agent_pos["z"]) / 2,
        },
    )


_CLOSE_BLOCKLIST = {"Blinds"}


def _close_container(controller: Controller, object_type: str):
    if object_type in _CLOSE_BLOCKLIST:
        raise ValueError(f"{object_type} is on the close blocklist")
    for obj in _objects(controller):
        if obj["objectType"] == object_type and obj.get("openable") and obj.get("isOpen"):
            _step_or_raise(controller, "CloseObject", objectId=obj["objectId"], forceAction=True)
            return
    # Already closed — desired state achieved, nothing to do


def _remove_object(controller: Controller, object_type: str):
    candidates = [obj for obj in _objects(controller) if obj["objectType"] == object_type]
    if not candidates:
        raise ValueError(f"No {object_type} found to remove")
    candidates.sort(key=lambda o: (not o.get("visible", False), o["objectId"]))
    _step_or_raise(controller, "DisableObject", objectId=candidates[0]["objectId"])


def _hide_object(controller: Controller, object_type: str, pddl_params: dict = None):
    event = controller.step(action="Pass")
    target_obj = _find_first(
        [obj for obj in event.metadata["objects"] if obj.get("pickupable")],
        object_type,
        visible=True,
    )

    exclude_types = set()
    if pddl_params:
        for key in ("parent_target", "mrecep_target"):
            if pddl_params.get(key):
                exclude_types.add(pddl_params[key])

    candidates = []
    for obj in event.metadata["objects"]:
        if not obj.get("receptacle"):
            continue
        if obj["objectType"] in exclude_types:
            continue
        if obj["objectId"] in (target_obj.get("parentReceptacles") or []):
            continue
        candidates.append(obj)

    if not candidates:
        raise ValueError(f"Failed to hide {object_type}: no valid non-target receptacle")
    candidates.sort(key=lambda o: (o["objectType"], o["objectId"]))
    _place_object_at_point(controller, target_obj, candidates[0]["position"])


_INJECTORS = {
    "set_object_property": _set_object_property,
    "swap_object": _swap_object,
    "occlude_object": _occlude_object,
    "close_container": _close_container,
    "remove_object": _remove_object,
    "hide_object": _hide_object,
}
