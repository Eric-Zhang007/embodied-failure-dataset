"""
因果陷阱引擎。

每步检查 agent 的行为，触发有因果关系的环境变化。
不是凭空出现——agent 的动作导致物理后果。

用法：
    engine = CausalTrapEngine()
    for step in episode:
        traps = engine.step(action, success, error, env)
        for trap in traps:
            inject(env.controller, **trap)
"""

import random
from typing import Optional

from src.env_injector import inject


class CausalTrapEngine:
    """六种因果陷阱，每种有独立的触发条件和冷却。"""

    def __init__(self, enabled: bool = True, seed: int = None):
        self.enabled = enabled
        if seed is not None:
            random.seed(seed)

        # ── 状态追踪 ──
        self._consecutive_collisions = 0
        self._consecutive_rotations = 0
        self._rapid_steps = 0
        self._doors_opened: set[str] = set()
        self._step_count = 0

        # ── 冷却（避免同一种陷阱连续触发） ──
        self._cooldowns: dict[str, int] = {}

        # ── 概率 ──
        self.SLIP_PROB = 0.20          # 手滑概率
        self.KNOCK_PROB = 0.30         # 跑太快撞翻概率

    def step(self, action: str, success: bool, error: Optional[str],
             controller, params: dict = None) -> list[dict]:
        """
        每步调用。返回该激活的陷阱列表，每个陷阱是 inject() 的参数。
        """
        if not self.enabled:
            return []

        self._step_count += 1
        active: list[dict] = []
        params = params or {}

        # ── ① 撞墙 → shelf 上掉东西 ──
        if not success and self._is_blocked(error):
            self._consecutive_collisions += 1
            if self._consecutive_collisions >= 2 and self._can_trigger("shelf_drop"):
                trap = self._shelf_drop(controller)
                if trap:
                    active.append(trap)
                    self._consecutive_collisions = 0
                    self._set_cooldown("shelf_drop", 8)
        elif success and action not in ("RotateLeft", "RotateRight",
                                          "LookUp", "LookDown", "LookAround"):
            self._consecutive_collisions = 0

        # ── ② PickupObject → 手滑 ──
        if action == "PickupObject" and success:
            if random.random() < self.SLIP_PROB and self._can_trigger("object_slip"):
                obj_id = params.get("objectId", "")
                if obj_id:
                    active.append({
                        "method": "set_object_property",
                        "object_type": "",
                        "property": "_slip",
                        "value": True,
                        "_slip_object_id": obj_id,
                    })
                    self._set_cooldown("object_slip", 15)

        # ── ③ OpenObject → 门后有障碍 ──
        if action == "OpenObject" and success:
            obj_id = params.get("objectId", "")
            if obj_id and obj_id not in self._doors_opened:
                self._doors_opened.add(obj_id)
                if self._can_trigger("behind_door"):
                    trap = self._behind_door(controller, obj_id)
                    if trap:
                        active.append(trap)
                        self._set_cooldown("behind_door", 20)

        # ── ④ 卡窄缝 ──
        if not success and self._is_blocked(error) and self._is_tight_space(controller):
            if self._can_trigger("tight_space"):
                active.append({
                    "method": "tight_space_stuck",
                    "params": {"step": self._step_count},
                })
                self._set_cooldown("tight_space", 10)

        # ── ⑤ 原地转 → 视觉模糊 ──
        if action in ("RotateLeft", "RotateRight"):
            self._consecutive_rotations += 1
        else:
            self._consecutive_rotations = 0

        if self._consecutive_rotations >= 3 and self._can_trigger("visual_blur"):
            active.append({
                "method": "visual_blur",
                "params": {"duration_steps": 2},
            })
            self._set_cooldown("visual_blur", 12)
            self._consecutive_rotations = 0

        # ── ⑥ 跑太快 → 撞翻东西 ──
        if action == "MoveSequence" and success:
            self._rapid_steps += 1
        elif action not in ("MoveSequence", "MoveAhead"):
            self._rapid_steps = 0

        if self._rapid_steps >= 4 and self._can_trigger("knock_over"):
            if random.random() < self.KNOCK_PROB:
                trap = self._knock_over(controller)
                if trap:
                    active.append(trap)
                    self._rapid_steps = 0
                    self._set_cooldown("knock_over", 12)

        return active

    # ── 陷阱实现 ──────────────────────────────────────────────

    def _shelf_drop(self, controller) -> Optional[dict]:
        """在 agent 前方的 receptacle 上找一个可拾取物体，挪到地上挡住去路。"""
        event = controller.step(action="Pass")
        objects = event.metadata["objects"]
        agent = event.metadata["agent"]
        ax, az = agent["position"]["x"], agent["position"]["z"]

        # 找 agent 前方 1m 内的 receptacle
        nearby = [
            o for o in objects
            if o.get("receptacle")
            and abs(o["position"]["x"] - ax) < 1.5
            and abs(o["position"]["z"] - az) < 1.5
        ]
        if not nearby:
            return None

        # 在 receptacle 上找小物体
        for rec in nearby:
            children = [
                o for o in objects
                if o.get("parentReceptacles")
                and rec["objectId"] in o["parentReceptacles"]
                and o.get("pickupable")
            ]
            if children:
                child = random.choice(children)
                # 挪到地上（agent 前方）
                drop_pos = {
                    "x": ax + 0.5,
                    "y": 0.01,
                    "z": az + 0.3,
                }
                try:
                    from src.env_injector import _place_object_at_point
                    _place_object_at_point(controller, child, drop_pos)
                    return {
                        "method": "shelf_drop",
                        "object_type": child["objectType"],
                        "params": {"from_receptacle": rec["objectType"]},
                    }
                except Exception:
                    return None
        return None

    def _behind_door(self, controller, door_id: str) -> Optional[dict]:
        """在刚打开的门后面放一个障碍物。"""
        event = controller.step(action="Pass")
        objects = event.metadata["objects"]

        # 找到门的位置
        door = next((o for o in objects if o["objectId"] == door_id), None)
        if not door:
            return None

        # 找一个不重要的小物体挡在门后
        candidates = [
            o for o in objects
            if o.get("pickupable") and o.get("visible")
            and o["objectId"] != door_id
            and o.get("objectType") not in (
                "AlarmClock", "Apple", "Bread", "Tomato", "Lettuce",
                "Mug", "Bowl", "Plate", "Cup", "Pot", "Pan",
            )  # 排除常见任务目标
        ]
        if not candidates:
            return None

        blocker = random.choice(candidates)
        door_pos = door["position"]
        block_pos = {
            "x": door_pos["x"] + 0.3,
            "y": door_pos["y"],
            "z": door_pos["z"] + 0.3,
        }
        try:
            from src.env_injector import _place_object_at_point
            _place_object_at_point(controller, blocker, block_pos)
            return {
                "method": "behind_door",
                "object_type": blocker["objectType"],
                "params": {"door_type": door.get("objectType", "")},
            }
        except Exception:
            return None

    def _knock_over(self, controller) -> Optional[dict]:
        """agent 跑太快撞翻手边的物体。"""
        event = controller.step(action="Pass")
        objects = event.metadata["objects"]
        agent = event.metadata["agent"]
        ax, az = agent["position"]["x"], agent["position"]["z"]

        # agent 身旁 <0.8m 的小物体
        nearby = [
            o for o in objects
            if o.get("pickupable")
            and abs(o["position"]["x"] - ax) < 0.8
            and abs(o["position"]["z"] - az) < 0.8
            and not o.get("isPickedUp")
        ]
        if not nearby:
            return None

        victim = random.choice(nearby)
        # 推远一点
        push_pos = {
            "x": victim["position"]["x"] + random.uniform(-0.8, 0.8),
            "y": victim["position"]["y"],
            "z": victim["position"]["z"] + random.uniform(-0.8, 0.8),
        }
        try:
            from src.env_injector import _place_object_at_point
            _place_object_at_point(controller, victim, push_pos)
            return {
                "method": "knock_over",
                "object_type": victim["objectType"],
                "params": {},
            }
        except Exception:
            return None

    # ── 辅助 ──────────────────────────────────────────────────

    @staticmethod
    def _is_blocked(error: Optional[str]) -> bool:
        if not error:
            return False
        keywords = ("blocked", "collision", "obstructed", "cannot move",
                     "Floor", "is blocking")
        return any(k in error.lower() for k in keywords)

    @staticmethod
    def _is_tight_space(controller) -> bool:
        """判断 agent 是否被两边的障碍物夹住。"""
        try:
            event = controller.step(action="Pass")
            # 简单判断：前方 0.3m 有障碍且左右也有障碍
            # 这里先返回 False，后续可以加更精确的检测
            return False
        except Exception:
            return False

    def _can_trigger(self, trap_name: str) -> bool:
        return self._cooldowns.get(trap_name, 0) <= self._step_count

    def _set_cooldown(self, trap_name: str, duration: int):
        self._cooldowns[trap_name] = self._step_count + duration
