"""
AI2-THOR 动作适配层。
"""

# AI2-THOR 5.0.0 中已废弃的参数名（全部移除）

def clean_history_params(params: dict) -> dict:
    """Strip internal resolved IDs from history display — show only what the model proposed."""
    return {k: v for k, v in params.items() if k not in ("objectId", "receptacleId")}
_DEPRECATED_PARAMS = {
    "rotateOnTeleport",
    "forceVisible",
    "coordinateObjectId",
    "coordinateReceptacleObjectId",
    "cleanObjectId",
    # 模型输出 objectType/receptacleType，resolver 填为 objectId 后清理
    "objectType",
    "receptacleType",
}

# 需要 objectId 参数的动作
_OBJECT_ACTIONS = {
    "PickupObject", "PutObject", "OpenObject", "CloseObject",
    "ToggleObjectOn", "ToggleObjectOff", "SliceObject", "BreakObject",
    "FillObjectWithLiquid", "EmptyLiquidFromObject",
}


def adapt(action: str, params: dict) -> tuple[str, dict]:
    """清理废弃参数，应用特定适配，返回 (action, params)。"""
    p = {k: v for k, v in params.items() if k not in _DEPRECATED_PARAMS}
    handler = _ADAPTERS.get(action)
    if handler:
        action, p = handler(p)
    return action, p


def resolve_object_ids(action: str, params: dict, visible_objects: list[dict]):
    """
    模型输出 objectType（如 "AlarmClock"），code 匹配可见物体列表填 objectId。
    PutObject 的 receptacleType 同理。
    返回 (params, warnings) — warnings 是未匹配的提示列表，用 \n 拼成字符串。
    """
    p = dict(params)
    warnings = []
    all_types = sorted(set(o["objectType"] for o in visible_objects))

    # Resolve receptacleType → receptacleId (PutObject)
    # ONLY matches objects with receptacle=true. No fallback to non-receptacles.
    if action == "PutObject" and "receptacleType" in p:
        rt = p.pop("receptacleType")
        candidates = [o for o in visible_objects
                      if o["objectType"] == rt and o.get("receptacle") and o.get("visibleBounds2D")]
        if not candidates:
            candidates = [o for o in visible_objects
                      if o["objectType"] == rt and o.get("receptacle")]
        if candidates:
            p["receptacleId"] = candidates[0]["objectId"]
        else:
            warnings.append(
                f"receptacleType '{rt}' NOT FOUND in visible objects. "
                f"Available receptacles: {[o['objectType'] for o in visible_objects if o.get('receptacle')]}. "
                f"The receptacle may be out of sight (behind you, inside a container, or in another room). "
                f"Rotate or move to find it. Check the image: what do you actually SEE?"
            )

    # Resolve objectType → objectId
    if action in _OBJECT_ACTIONS and "objectType" in p:
        ot = p.pop("objectType")
        # Prioritize: visible + pickupable, then visible, then pickupable, then any
        candidates = [o for o in visible_objects
                      if o["objectType"] == ot and o.get("visibleBounds2D") and o.get("pickupable")]
        if not candidates:
            candidates = [o for o in visible_objects
                      if o["objectType"] == ot and o.get("visibleBounds2D")]
        if not candidates:
            candidates = [o for o in visible_objects
                      if o["objectType"] == ot and o.get("pickupable")]
        if not candidates:
            candidates = [o for o in visible_objects
                      if o["objectType"] == ot]
        if candidates:
            p["objectId"] = candidates[0]["objectId"]
        else:
            warnings.append(
                f"objectType '{ot}' NOT FOUND in visible objects. "
                f"All visible types: {all_types}. "
                f"Two possibilities: (1) You used the wrong name — the task may say 'red cloth' but the internal objectType is 'Cloth'. "
                f"Use the EXACT name from the list above. "
                f"(2) The object is out of sight — behind you, inside a closed container, or in another part of the room. "
                f"Rotate, move, or open containers to find it. "
                f"CHECK the image: what do you actually SEE right now? Match your action to something visible."
            )

    return p, "\n".join(warnings) if warnings else None


# ------------------------------------------------------------------
# 适配函数
# ------------------------------------------------------------------

def _adapt_teleport_full(params: dict) -> tuple[str, dict]:
    p = dict(params)
    p.setdefault("horizon", 0)
    p.setdefault("standing", True)
    # Scalar rotation → Vector3 (AI2-THOR 5.0.0 expects Vector3)
    if "rotation" in p and not isinstance(p["rotation"], dict):
        p["rotation"] = {"x": 0, "y": p["rotation"], "z": 0}
    return "TeleportFull", p


def _adapt_put_object(params: dict) -> tuple[str, dict]:
    """PutObject: receptacleId → objectId（AI2-THOR 格式）"""
    p = dict(params)
    if "receptacleId" in p:
        p["objectId"] = p.pop("receptacleId")
    return "PutObject", p


_ADAPTERS = {
    "TeleportFull": _adapt_teleport_full,
    "PutObject": _adapt_put_object,
}
