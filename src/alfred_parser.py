"""
ALFRED traj_data.json 解析器。

职责：
- 读 ALFRED 轨迹 JSON
- 把 api_action 字符串解析成 (action, params) 元组
- 提取任务元信息（task_goal, scene, task_type）
"""

import json
import re


# ALFRED task_type → 我们的简化分类
TASK_TYPE_MAP = {
    "pick_and_place_simple": "pick_and_place",
    "pick_and_place_with_movable_recep": "pick_and_place",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "examine",
    "pick_two_obj_and_place": "pick_two",
}


def load_traj(traj_path: str) -> dict:
    with open(traj_path, "r") as f:
        return json.load(f)


def _generate_task_goal(task_type: str, pddl_params: dict) -> str:
    """Generate a clear imperative task instruction from PDDL parameters.

    ALFRED's turk_annotations are often vague descriptions like
    "Yellow apple sitting in a pan on the table". This produces
    unambiguous instructions for the Planner and Executor.
    """
    obj = pddl_params.get("object_target", "")
    parent = pddl_params.get("parent_target", "")
    mrecep = pddl_params.get("mrecep_target", "")
    toggle = pddl_params.get("toggle_target", "")
    sliced = pddl_params.get("object_sliced", False)

    if task_type == "pick_and_place_simple":
        return f"Pick up the {obj} and put it on the {parent}."
    elif task_type == "pick_and_place_with_movable_recep":
        return f"Pick up the {obj}, put it in the {mrecep}, and place the {mrecep} on the {parent}."
    elif task_type == "pick_clean_then_place_in_recep":
        return f"Clean the {obj} and put it in the {parent}."
    elif task_type == "pick_heat_then_place_in_recep":
        return f"Heat the {obj} and put it in the {parent}."
    elif task_type == "pick_cool_then_place_in_recep":
        return f"Cool the {obj} and put it in the {parent}."
    elif task_type == "look_at_obj_in_light":
        return f"Turn on the {toggle} and check the {obj} under its light."
    elif task_type == "pick_two_obj_and_place":
        return f"Pick up the {obj} and the {mrecep}, and put both on the {parent}."
    else:
        # Fallback: construct basic instruction from available params
        parts = []
        if obj:
            action = "Slice" if sliced else "Pick up"
            parts.append(f"{action} the {obj}")
        if mrecep and mrecep != obj:
            parts.append(f"use the {mrecep}")
        if parent:
            parts.append(f"put it on the {parent}")
        return ". ".join(parts) + "." if parts else f"Complete the task: {task_type}"


def extract_metadata(traj: dict) -> dict:
    """
    从 ALFRED 轨迹提取任务元信息，映射到我们 schema 的 metadata 字段。
    """
    task_type_raw = traj.get("task_type", "unknown")
    task_type = TASK_TYPE_MAP.get(task_type_raw, task_type_raw)
    pddl_params = traj.get("pddl_params", {})
    task_desc = _generate_task_goal(task_type_raw, pddl_params)

    return {
        "task_goal": task_desc,
        "scene": traj["scene"]["floor_plan"],
        "task_type": task_type,
        "alfred_task_type": task_type_raw,
        "alfred_task_id": traj.get("task_id", ""),
        "pddl_params": traj.get("pddl_params", {}),
        "alfred_scene": extract_scene_state(traj),
        "initial_traps": [],
    }


def extract_scene_state(traj: dict) -> dict:
    scene = traj.get("scene", {})
    return {
        "floor_plan": scene.get("floor_plan"),
        "scene_num": scene.get("scene_num"),
        "random_seed": scene.get("random_seed"),
        "object_poses": scene.get("object_poses") or [],
        "object_toggles": scene.get("object_toggles") or [],
        "dirty_and_empty": bool(scene.get("dirty_and_empty")),
        "init_action": extract_init_action(traj),
    }


def extract_init_action(traj: dict) -> dict | None:
    """解析 scene.init_action。可以已经是 dict，也可能是字符串。"""
    raw = traj.get("scene", {}).get("init_action")
    if not raw:
        return None
    if isinstance(raw, dict):
        return _normalize_action_dict(raw)
    return _parse_action_string(raw)


def extract_low_actions(traj: dict) -> list[dict]:
    """
    解析 plan.low_actions，返回列表 [{action, params, high_idx}]。
    """
    actions = []
    for entry in traj.get("plan", {}).get("low_actions", []):
        raw = entry["api_action"]
        if isinstance(raw, dict):
            parsed = _normalize_action_dict(raw)
        else:
            parsed = _parse_action_string(raw)
        parsed["high_idx"] = entry["high_idx"]
        actions.append(parsed)
    return actions


# ------------------------------------------------------------------
# api_action 解析（支持 dict 和 string 两种格式）
# ------------------------------------------------------------------

def _normalize_action_dict(d: dict) -> dict:
    """
    {'action': 'PickupObject', 'objectId': '...', 'forceAction': True}
    → {'action': 'PickupObject', 'params': {'objectId': '...', 'forceAction': True}}
    """
    # Already normalised: {'action': '...', 'params': {...}}
    if "params" in d and isinstance(d["params"], dict):
        return {"action": d["action"], "params": dict(d["params"])}
    params = {k: v for k, v in d.items() if k != "action"}
    return {"action": d["action"], "params": params}


# 匹配: action='PickupObject', objectId='...', forceAction=True
_ACTION_RE = re.compile(
    r"action\s*=\s*(?:'([^']*)'|\"([^\"]*)\")", re.IGNORECASE
)

# 匹配: key='value' 或 key=True/False/数字
_KWARG_RE = re.compile(
    r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(True|False|[-+]?\d+\.?\d*))"
)


def _parse_action_string(api_action: str) -> dict:
    """
    "controller.step(action='PickupObject', objectId='Apple|...', forceAction=True)"
    → {"action": "PickupObject", "params": {"objectId": "Apple|...", "forceAction": True}}
    """
    action_match = _ACTION_RE.search(api_action)
    action = action_match.group(1) or action_match.group(2) if action_match else "Unknown"

    params = {}
    for m in _KWARG_RE.finditer(api_action):
        key = m.group(1)
        if key == "action":
            continue
        val = m.group(2) or m.group(3) or m.group(4)
        if val == "True":
            val = True
        elif val == "False":
            val = False
        elif val is not None:
            try:
                val_int = int(val)
                val = val_int
            except ValueError:
                try:
                    val_float = float(val)
                    val = val_float
                except ValueError:
                    pass
        params[key] = val

    return {"action": action, "params": params}
