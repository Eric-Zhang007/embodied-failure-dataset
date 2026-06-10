# Phase 1 CONTEXT — PickupObject 诊断与修复

**日期**: 2026-06-10
**来源**: /gsd-discuss-phase 1
**状态**: 代码改动已完成，待 E2E 验证

---

## 决策记录

### D1: visible 的真实含义

**问题**: AI2-THOR metadata 中的 `visible` 字段是基于 `visibilityDistance` 的球半径判断（1.5m 内的物体都标为 visible），不是相机视锥体判断。

**证据**: AI2-THOR `server.py:510-514` 中 `visibleBounds2D` 字段的计算逻辑：`obj["visible"] AND obj["objectId"] in instance_detections2D`。说明 `visible` 只是距离筛选，`visibleBounds2D` 才是"真正在画面中"。

**决定**:
- `visibilityDistance`: 1.5 → 100（室内场景全覆盖）
- `renderObjectImage`: False → True（启用 instance segmentation，使 `instance_detections2D` 可用）
- 所有"判断画面中能看到什么"的逻辑从 `visible` 改为 `visibleBounds2D`

### D2: 不要启发式兜底

**决定**: 不在 `action_adapter.py` 中加距离预检。修复根因，不搞 workaround。

### D3: 距离提示

**决定**: Phase 1 prompt 的 Object interaction 区加一句 `Must be within 0.5m reach`。

### D4: 验证方式

**决定**: 跑一条 E2E。

### D5: 性能

**决定**: `renderObjectImage: True` 的 GPU 开销不需要关注。

### D6: LookAround 死循环修复

**问题**: EB 在 MoveAhead → LookAround → MoveAhead → LookAround 之间循环。多图确认目标 1.9m ahead，走一步 0.25m 后场景变化极小，EB 怀疑方向错了 → 又调 LookAround。本质是模型无法承受"连续多步接近"的不确定性。

**决定**:
- 修复初始 LookAround 恢复旋转的 success 检查（原来 silent ignore）
- 增加 LookAround cooldown：完成一次 LookAround 后，接下来 3 步禁止再次 LookAround
- 强制 agent 在这 3 步内 commit 到实际移动或交互

### D7: 初始 LookAround 恢复旋转检查

**Bug**: `branch_runner.py:122` 的 `env.step("RotateLeft")` 没检查返回值。非初始 LookAround 的对应代码有检查（line 370-372）。

**修复**: 加上 success 检查和 RuntimeError raise。

---

## 已完成的代码改动

1. `alfred_scene.py`: `visibilityDistance: 100`, `renderObjectImage: True`
2. `eb_agent.py` Phase1 prompt: `o.get("visible")` → `o.get("visibleBounds2D")`
3. `eb_agent.py` Phase3 prompt: 同上
4. `eb_agent.py` LookAround prompt: `o.get("visible")` → `o.get("visibleBounds2D")`
5. `eb_agent.py` Phase1 prompt: Object interaction 区加 "Must be within 0.5m reach"
6. `action_adapter.py` resolve_object_ids: 3 处 `visible` → `visibleBounds2D`
7. `executor.py`: 1 处 `visible` → `visibleBounds2D`
8. `oracle_agent.py` Phase2: 1 处 `visible` → `visibleBounds2D`
9. `task_conditions.py` examine checker: 1 处 `visible` → `visibleBounds2D`
10. `branch_runner.py`: 初始 LookAround 恢复旋转 success 检查
11. `branch_runner.py`: LookAround cooldown 机制（3 步禁令）

## 保留 `visible` 的地方

- `alfred_scene.py:171` — `find_closest_object_of_type()` 场景操作
- `env_injector.py:73` — `_find_first()` 陷阱注入目标选择

## 下一步

跑一条 pick_and_place_simple E2E 验证所有改动。
