"""
Trap 规划器。

从 failure_type_library 中为 episode 选择陷阱，
在场景初始化后应用。
"""

import json
import random
import os
from typing import Optional

from ai2thor.controller import Controller
from src.env_injector import inject


class TrapPlanner:
    def __init__(self, library_path: str = None, seed: int = None):
        if library_path is None:
            library_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "failure_type_library.json",
            )
        with open(library_path, "r") as f:
            self.library = json.load(f)
        if seed is not None:
            random.seed(seed)

    def plan_traps(
        self,
        task_type: str,
        scene_objects: list[dict],
        min_traps: int = 1,
        max_traps: int = 3,
        exclude_types: set = None,
    ) -> list[dict]:
        """
        为一个 episode 选择陷阱。
        exclude_types: 不能动的物体类型集合（任务目标）。
        """
        if exclude_types is None:
            exclude_types = set()

        candidates = [e for e in self.library if task_type in e["applicable_tasks"]]
        if not candidates:
            return []

        # 解析场景中的物体，生成可用的注入实例
        resolved = []
        for entry in candidates:
            instance = self._resolve(entry, scene_objects, exclude_types)
            if instance:
                resolved.append(instance)

        if not resolved:
            return []

        n = random.randint(min_traps, min(max_traps, len(resolved)))
        selected = random.sample(resolved, n)

        traps = []
        for i, s in enumerate(selected):
            traps.append({
                "trap_id": f"trap_{i}",
                "failure_type": s["failure_type"],
                "description": s["description"],
                "injection": s["injection"],
                "severity": s["severity"],
            })
        return traps

    def apply_traps(
        self,
        controller: Controller,
        traps: list[dict],
    ) -> list[dict]:
        """
        应用陷阱到环境。返回应用结果列表。
        """
        results = []
        for t in traps:
            inj = t["injection"]
            result = inject(controller, method=inj["method"], **inj["params"])
            results.append({
                "trap_id": t["trap_id"],
                "success": result["success"],
                "error": result.get("error"),
            })
        return results

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _resolve(self, entry: dict, scene_objects: list[dict], exclude_types: set) -> Optional[dict]:
        """
        尝试把 entry 的 <target> / <appliance> / <container> 占位符
        解析为场景中实际存在的物体。
        """
        method = entry["injection"]["method"]
        params = dict(entry["injection"]["params"])

        # 需要解析的参数值（如 "<target>", "<appliance>"）
        for key, val in list(params.items()):
            if isinstance(val, str) and val.startswith("<"):
                extra_filter = {}
                prop = params.get("property", "")
                if prop == "is_broken":
                    extra_filter["breakable"] = True
                obj = self._find_object(val, scene_objects, exclude_types, extra_filter)
                if obj is None:
                    return None
                params[key] = obj

        return {
            "failure_type": entry["failure_type"],
            "description": entry["description"],
            "injection": {"method": method, "params": params},
            "severity": entry.get("severity", 0.3),
        }

    def _find_object(
        self, placeholder: str, scene_objects: list[dict],
        exclude_types: set, extra_filter: dict = None,
    ) -> Optional[str]:
        """根据占位符类型找场景中合适的物体 objectType，排除 exclude_types 中的类型。

        extra_filter: 额外的属性要求，如 {"breakable": True}。
        """
        extra_filter = extra_filter or {}
        need = placeholder.strip("<>")

        if need == "target":
            candidates = [o for o in scene_objects if o.get("pickupable")]
        elif need == "appliance":
            candidates = [o for o in scene_objects if o.get("toggleable")]
        elif need == "container":
            _close_blocklist = {"Blinds"}
            candidates = [o for o in scene_objects
                          if o.get("openable") and o["objectType"] not in _close_blocklist]
        elif need == "receptacle":
            candidates = [o for o in scene_objects if o.get("receptacle")]
        elif need == "tool":
            tools = {"ButterKnife", "Knife", "Spatula", "ScrubBrush", "Plunger",
                      "SoapBottle", "SprayBottle"}
            candidates = [o for o in scene_objects if o["objectType"] in tools]
        elif need == "lamp":
            lamps = {"DeskLamp", "FloorLamp"}
            candidates = [o for o in scene_objects if o["objectType"] in lamps]
        else:
            return None

        # 排除任务关键物体
        candidates = [o for o in candidates if o["objectType"] not in exclude_types]
        # 额外属性过滤（如 breakable）
        for prop, val in extra_filter.items():
            candidates = [o for o in candidates if o.get(prop) == val]
        if not candidates:
            return None
        return random.choice(candidates)["objectType"]
