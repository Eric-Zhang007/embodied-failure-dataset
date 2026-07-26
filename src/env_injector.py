from ai2thor.controller import Controller


_CLOSE_BLOCKLIST = {"Blinds"}


def inject(controller: Controller, method: str, pddl_params: dict | None = None, **params) -> dict:
    """Apply one Oracle-selected, known AI2-THOR mutation.

    Oracle chooses the method and scene object type. The runtime keeps the
    authority boundary narrow: unknown and non-recoverable mutations are not
    exposed here rather than being hidden behind a candidate allowlist.
    """
    handler = _INJECTORS.get(method)
    if handler is None:
        return {"success": False, "error": f"Unknown injection method: {method}", "blocked": None}

    try:
        if method in {"hide_object", "stash_held_object"}:
            handler(controller, pddl_params=pddl_params, **params)
        else:
            handler(controller, **params)
        return {"success": True, "error": None, "blocked": None}
    except ValueError as error:
        return {"success": False, "error": str(error), "blocked": None}


def _step_or_raise(controller: Controller, action: str, **params):
    event = controller.step(action=action, **params)
    if not event.metadata.get("lastActionSuccess", False):
        raise ValueError(f"{action} failed: {event.metadata.get('errorMessage') or 'unknown error'}")
    return event


def _restore_object_pose(controller: Controller, obj: dict):
    _step_or_raise(
        controller,
        "TeleportObject",
        objectId=obj["objectId"],
        position=obj["position"],
        rotation=obj["rotation"],
        forceAction=True,
        allowTeleportOutOfHand=True,
    )


def _objects(controller: Controller) -> list[dict]:
    event = getattr(controller, "last_event", None)
    metadata = getattr(event, "metadata", None)
    if metadata is not None:
        return metadata.get("objects", [])
    return controller.step(action="Pass").metadata["objects"]


def _find_first(
    objects: list[dict],
    object_type: str,
    *,
    object_id: str | None = None,
    visible: bool | None = None,
) -> dict:
    candidates = [obj for obj in objects if obj.get("objectType") == object_type]
    if object_id is not None:
        candidates = [obj for obj in candidates if obj.get("objectId") == object_id]
    if visible is not None:
        candidates = [obj for obj in candidates if bool(obj.get("visible")) == visible]
    if not candidates:
        suffix = " visible" if visible else ""
        if object_id:
            suffix += f" with id {object_id}"
        raise ValueError(f"No{suffix} {object_type} found")
    return sorted(candidates, key=lambda obj: obj["objectId"])[0]


def _set_object_property(
    controller: Controller,
    object_type: str,
    property: str,
    value,
    object_id: str | None = None,
):
    obj = _find_first(_objects(controller), object_type, object_id=object_id)
    object_id = obj["objectId"]
    if property == "is_open":
        _set_open(controller, object_id, bool(value))
    elif property == "is_toggled":
        action = "ToggleObjectOn" if value else "ToggleObjectOff"
        _step_or_raise(controller, action, objectId=object_id, forceAction=True)
    elif property == "fillLiquid":
        _step_or_raise(
            controller,
            "FillObjectWithLiquid",
            objectId=object_id,
            fillLiquid=str(value),
            forceAction=True,
        )
    elif property == "emptyLiquid":
        _step_or_raise(controller, "EmptyLiquidFromObject", objectId=object_id, forceAction=True)
    else:
        raise ValueError(f"Cannot set property {property}={value} for {object_id}")


def _set_open(controller: Controller, object_id: str, value: bool):
    if object_id.startswith("Blinds"):
        raise ValueError("Blinds OpenObject/CloseObject causes an AI2-THOR timeout")
    action = "OpenObject" if value else "CloseObject"
    _step_or_raise(controller, action, objectId=object_id, forceAction=True)


def _close_container(controller: Controller, object_type: str, object_id: str | None = None):
    if object_type in _CLOSE_BLOCKLIST:
        raise ValueError(f"{object_type} is on the close blocklist")
    for obj in _objects(controller):
        if (
            obj.get("objectType") == object_type
            and (object_id is None or obj.get("objectId") == object_id)
            and obj.get("openable")
            and obj.get("isOpen")
        ):
            _step_or_raise(controller, "CloseObject", objectId=obj["objectId"], forceAction=True)
            return
    raise ValueError(f"No open {object_type} found to close")


def _drop_held_object(controller: Controller):
    inventory = getattr(getattr(controller, "last_event", None), "metadata", {}).get("inventoryObjects", [])
    if not inventory:
        raise ValueError("Cannot drop an object when the agent is not holding one")
    _step_or_raise(controller, "DropHandObject", forceAction=True)


def _stash_held_object(
    controller: Controller,
    container_type: str,
    container_id: str | None = None,
    pddl_params: dict | None = None,
):
    inventory = getattr(getattr(controller, "last_event", None), "metadata", {}).get("inventoryObjects", [])
    if not inventory:
        raise ValueError("Cannot stash an object when the agent is not holding one")
    excluded = {
        pddl_params[key]
        for key in ("parent_target", "mrecep_target")
        if pddl_params and pddl_params.get(key)
    }
    if container_type in excluded:
        raise ValueError(f"Cannot stash an object in task destination {container_type}")
    container = _find_first([
        obj for obj in _objects(controller)
        if (
            obj.get("objectType") == container_type
            and obj.get("openable")
            and obj.get("receptacle")
            and not obj.get("isOpen")
        )
    ], container_type, object_id=container_id)
    held = inventory[0]
    opened = False
    placed = False
    try:
        _set_open(controller, container["objectId"], True)
        opened = True
        _step_or_raise(controller, "PutObject", objectId=container["objectId"], forceAction=True)
        placed = True
        _set_open(controller, container["objectId"], False)
    except ValueError:
        if placed:
            _step_or_raise(controller, "PickupObject", objectId=held["objectId"], forceAction=True)
        if opened:
            _set_open(controller, container["objectId"], False)
        raise


def _hide_object(
    controller: Controller,
    object_type: str,
    container_type: str | None = None,
    object_id: str | None = None,
    container_id: str | None = None,
    pddl_params: dict | None = None,
):
    target = _find_first(
        _objects(controller), object_type, object_id=object_id, visible=True,
    )
    if not target.get("pickupable"):
        raise ValueError(f"{target['objectId']} is not pickupable")
    excluded = {
        pddl_params[key]
        for key in ("parent_target", "mrecep_target")
        if pddl_params and pddl_params.get(key)
    }
    if not container_type:
        raise ValueError("hide_object requires an openable container_type")
    if container_type in excluded:
        raise ValueError(f"Cannot hide {object_type} in task destination {container_type}")
    inventory = getattr(getattr(controller, "last_event", None), "metadata", {}).get("inventoryObjects", [])
    if inventory:
        raise ValueError("Cannot hide an object while the agent is already holding one")
    container = _find_first([
        obj for obj in _objects(controller)
        if obj.get("objectType") == container_type
        and obj.get("openable")
        and obj.get("receptacle")
        and not obj.get("isOpen")
        and obj.get("objectId") not in (target.get("parentReceptacles") or [])
    ], container_type, object_id=container_id)
    opened = False
    moved = False
    try:
        _set_open(controller, container["objectId"], True)
        opened = True
        _step_or_raise(controller, "PickupObject", objectId=target["objectId"], forceAction=True)
        moved = True
        _step_or_raise(controller, "PutObject", objectId=container["objectId"], forceAction=True)
        _set_open(controller, container["objectId"], False)
    except ValueError as error:
        try:
            if opened:
                _set_open(controller, container["objectId"], False)
            if moved:
                _restore_object_pose(controller, target)
        except ValueError as rollback_error:
            raise ValueError(f"{error}; rollback failed: {rollback_error}") from rollback_error
        raise


_INJECTORS = {
    "set_object_property": _set_object_property,
    "close_container": _close_container,
    "drop_held_object": _drop_held_object,
    "stash_held_object": _stash_held_object,
    "hide_object": _hide_object,
}
