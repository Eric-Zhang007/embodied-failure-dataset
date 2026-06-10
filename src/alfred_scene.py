import math

from src.action_adapter import adapt


ALFRED_INIT_SETTINGS = {
    "gridSize": 0.125,
    "cameraY": 0.75,
    "renderImage": True,
    "renderDepthImage": False,
    "renderClassImage": False,
    "renderObjectImage": True,
    "visibilityDistance": 100.0,
    "makeAgentsVisible": False,
}


def empty_task_state() -> dict:
    return {
        "cleaned_objects": set(),
        "heated_objects": set(),
        "cooled_objects": set(),
    }


def serializable_task_state(state: dict) -> dict:
    return {
        "cleaned_objects": sorted(state.get("cleaned_objects", set())),
        "heated_objects": sorted(state.get("heated_objects", set())),
        "cooled_objects": sorted(state.get("cooled_objects", set())),
    }


def restore_alfred_scene(controller, scene_state: dict):
    object_poses = scene_state.get("object_poses") or []
    if not object_poses:
        raise ValueError("ALFRED scene_state.object_poses is required to restore the expert initial scene")

    _step_or_raise(controller, "Initialize", **ALFRED_INIT_SETTINGS)

    object_toggles = scene_state.get("object_toggles") or []
    if object_toggles:
        _apply_object_toggles(controller, object_toggles)

    if scene_state.get("dirty_and_empty"):
        _apply_dirty_and_empty(controller)

    merged_poses = _merge_object_poses_for_current_scene(controller, object_poses)
    _step_or_raise(controller, "SetObjectPoses", objectPoses=merged_poses)


def apply_init_action(controller, init_action: dict | None):
    if not init_action:
        return None
    action = init_action.get("action")
    params = init_action.get("params", {})
    if not action:
        raise ValueError("ALFRED init_action.action is required")
    action, params = adapt(action, params)
    if action == "TeleportFull":
        params.setdefault("forceAction", True)
    return _step_or_raise(controller, action, **params)


def update_alfred_task_state(task_state: dict, action: str, params: dict, metadata: dict):
    if not metadata.get("lastActionSuccess"):
        return

    object_id = params.get("objectId") or ""

    if action == "ToggleObjectOn" and "Faucet" in object_id:
        sink_basin = find_closest_object_of_type("SinkBasin", object_id, metadata)
        task_state["cleaned_objects"].update(sink_basin.get("receptacleObjectIds") or [])

    if action == "ToggleObjectOn" and "Microwave" in object_id:
        microwave = object_by_id(object_id, metadata)
        task_state["heated_objects"].update(microwave.get("receptacleObjectIds") or [])

    if action == "CloseObject" and "Fridge" in object_id:
        fridge = object_by_id(object_id, metadata)
        task_state["cooled_objects"].update(fridge.get("receptacleObjectIds") or [])


def clean_sink_contents_after_faucet(controller, action: str, params: dict, metadata: dict):
    object_id = params.get("objectId") or ""
    if action != "ToggleObjectOn" or "Faucet" not in object_id:
        return
    if not metadata.get("lastActionSuccess"):
        return

    pass_event = _step_or_raise(controller, "Pass")
    sink_basin = find_closest_object_of_type("SinkBasin", object_id, pass_event.metadata)
    for in_sink_id in sink_basin.get("receptacleObjectIds") or []:
        obj = object_by_id(in_sink_id, pass_event.metadata)
        if obj.get("dirtyable") and obj.get("isDirty"):
            _step_or_raise(controller, "CleanObject", objectId=in_sink_id)


def _apply_object_toggles(controller, object_toggles: list[dict]):
    event = _step_or_raise(controller, "Pass")
    objects = event.metadata.get("objects", [])
    for toggle in object_toggles:
        object_type = toggle.get("objectType")
        desired = bool(toggle.get("isOn"))
        matches = [
            obj for obj in objects
            if obj.get("objectType") == object_type and obj.get("toggleable")
        ]
        if not matches:
            raise ValueError(f"No toggleable {object_type} found while restoring ALFRED scene")
        for obj in matches:
            if bool(obj.get("isToggled")) == desired:
                continue
            action = "ToggleObjectOn" if desired else "ToggleObjectOff"
            _step_or_raise(controller, action, objectId=obj["objectId"], forceAction=True)


def _apply_dirty_and_empty(controller):
    event = _step_or_raise(controller, "Pass")
    for obj in event.metadata.get("objects", []):
        object_id = obj["objectId"]
        if obj.get("dirtyable") and not obj.get("isDirty"):
            _step_or_raise(controller, "DirtyObject", objectId=object_id, forceAction=True)
        if obj.get("canFillWithLiquid") and obj.get("isFilledWithLiquid"):
            _step_or_raise(controller, "EmptyLiquidFromObject", objectId=object_id, forceAction=True)


def _merge_object_poses_for_current_scene(controller, alfred_object_poses: list[dict]) -> list[dict]:
    event = _step_or_raise(controller, "Pass")
    alfred_by_name = {pose.get("objectName"): pose for pose in alfred_object_poses}
    merged = []

    for obj in event.metadata.get("objects", []):
        if not (obj.get("pickupable") or obj.get("moveable")):
            continue
        object_name = _object_name(obj)
        pose = alfred_by_name.get(object_name)
        if pose is None:
            pose = {
                "objectName": object_name,
                "position": obj["position"],
                "rotation": obj["rotation"],
            }
        merged.append({
            "objectName": object_name,
            "position": pose["position"],
            "rotation": pose["rotation"],
        })

    if not merged:
        raise ValueError("No pickupable or moveable objects available for SetObjectPoses")
    return merged


def _object_name(obj: dict) -> str:
    return obj.get("name", "").split("(Clone)")[0]


def object_by_id(object_id: str, metadata: dict) -> dict:
    for obj in metadata.get("objects", []):
        if obj.get("objectId") == object_id:
            return obj
    raise ValueError(f"Object not found in metadata: {object_id}")


def find_closest_object_of_type(object_type: str, ref_object_id: str, metadata: dict) -> dict:
    ref = object_by_id(ref_object_id, metadata)
    ref_pos = ref.get("position") or {}
    candidates = [
        obj for obj in metadata.get("objects", [])
        if obj.get("objectType") == object_type and obj.get("visible")
    ]
    if not candidates:
        raise ValueError(f"No visible {object_type} found near {ref_object_id}")

    def dist(obj):
        pos = obj.get("position") or {}
        return math.sqrt(
            (pos.get("x", 0.0) - ref_pos.get("x", 0.0)) ** 2
            + (pos.get("y", 0.0) - ref_pos.get("y", 0.0)) ** 2
            + (pos.get("z", 0.0) - ref_pos.get("z", 0.0)) ** 2
        )

    return min(candidates, key=dist)


def _step_or_raise(controller, action: str, **params):
    event = controller.step(action=action, **params)
    if not event.metadata.get("lastActionSuccess", False):
        error = event.metadata.get("errorMessage") or "unknown error"
        raise RuntimeError(f"AI2-THOR action {action} failed during ALFRED scene handling: {error}")
    return event
