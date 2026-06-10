# REQUIREMENTS.md — Milestone: 系统完善

**日期**: 2026-06-10
**更新**: E2E 分析完成，重新定优先级
**目标**: 修复 PickupObject 这个所有任务类型的共同 blocker，建立多任务类型 baseline，补关键测试，验证 Fork。

---

## E2E 现状（2026-06-10 分析）

| ALFRED 任务类型 | 跑过 | 成功 | 成功率 |
|---|---|---|---|
| pick_and_place_simple | 12 | 3 | 25% |
| look_at_obj_in_light | 3 | 0 | 0% |
| pick_and_place_with_movable_recep | 3 | 0 | 0% |
| pick_two_obj_and_place | 1 | 0 | 0% |
| pick_heat_then_place_in_recep | 0 | — | 从未测试 |
| pick_cool_then_place_in_recep | 0 | — | 从未测试 |
| pick_clean_then_place_in_recep | 0 | — | 从未测试 |

pick_and_place_simple 的 3 次成功全是 "alarm clock → desk"，且 agent 初始位置刚好在目标旁边。25% 成功率不能算"能 work"。

---

## R1: PickupObject 失败修复 ← 最高优先

**现状**: 元数据标为 visible 但 PickupObject 返回 "not found"。这是所有 7 种任务类型的共同 blocker。E2E 中几乎所有失败都是从 PickupObject 开始的。

**已知失败模式**（从 E2E 日志）:
- "NullReferenceException: Target object not found within the specified visibility" — 最常见
- "ArgumentException: objectId is not the objectId on any object" — objectId 解析错误
- AI2-THOR 内部 visibility check 比 metadata.visible 更严格

**要求**:
- R1.1: 对比 `metadata.objects[].visible` 和 AI2-THOR 实际可交互状态，找出差异原因（距离阈值？遮挡计算？碰撞体？）
- R1.2: 在 action_adapter 中增加距离预检：如果最近可见实例距离 > 0.5m，提前拒绝并反馈给 EB
- R1.3: 分析 AI2-THOR 5.0.0 的 visibilityDistance 设置（当前 1.5m），可能需要调大或动态调整
- R1.4: 如果 metadata 和实际可见性有系统差异，在 prompt 中增加对应的提示策略
- R1.5: 修复后 pick_and_place_simple 中 PickupObject 首次成功率 > 50%

**验证**: 跑 20 条 pick_and_place_simple，统计 PickupObject 首次成功率。单独跑 5 条不同物体类型的 pickup。

---

## R2: 全任务类型 Baseline

**现状**: 7 种 ALFRED 任务类型中，3 种从未测试过 E2E（heat/cool/clean）。其他 4 种中仅 pick_and_place_simple 有成功案例。

**要求**:
- R2.1: 对全部 7 种任务类型各跑至少 3 条 E2E（不加陷阱），建立 baseline 成功率
- R2.2: 记录每种任务类型的典型失败模式和 blocker
- R2.3: 检查各任务类型的 completion criteria 逻辑是否正确（特别是 edge case：空 parent_target、sliced 变体、movable receptacle 嵌套）
- R2.4: 对 heat/cool/clean 任务，确认 task_state 追踪逻辑（alfred_scene.py 的 update_alfred_task_state）在 EB 执行时是否正确更新

**验证**: 21 条 E2E（7 类型 × 3），每种任务类型有一个明确的成功率数字和 blocker 列表

---

## R3: 关键模块自动化测试

**现状**: 零测试。纯逻辑模块可以无 AI2-THOR 依赖地测试。

**要求**:
- R3.1: `task_conditions.py` — 所有 7 个 checker 函数的单元测试（mock metadata/pddl_params）
  - 覆盖: 完成/未完成判断、边界条件（空参数、缺失物体、sliced 变体、movable receptacle 双层嵌套）
- R3.2: `action_adapter.py` — `adapt()` 和 `resolve_object_ids()` 单元测试
  - 覆盖: 正常解析、objectType 不存在、重名物体选择、参数清理
- R3.3: `context_builder.py` — 历史渲染测试
  - 覆盖: EB 字段过滤、Oracle 字段包含、共享上下文合并
- R3.4: `episode_manager.py` — JSON 序列化/反序列化往返测试
- R3.5: `trap_planner.py` — 陷阱选择逻辑测试
  - 覆盖: 任务类型过滤、exclude_types 排除、占位符解析
- R3.6: `alfred_parser.py` — traj_data 解析测试（用 fixture 文件）

**不测试**: VLM 调用、AI2-THOR 交互、完整 E2E 流程

---

## R4: Fork 机制端到端验证

**现状**: 代码完整但从未端到端测试。

**要求**:
- R4.1: 以 `enable_fork=True` 跑至少 30 条 pick_and_place_simple 轨迹
- R4.2: 验证 fork 只在 counterfactual_grade=AC 时创建
- R4.3: 验证 fork 分支隔离（EB 不知道自己在 fork）
- R4.4: 抽查 fork root step 的 eb_reasoning（不含"反事实""替代"等词）
- R4.5: 修复过程中发现的 fork 相关 bug

---

## R5: EB 空间推理改善

**现状**: 即使多图 LookAround + 方向标签，模型仍看错方向/距离。examine 任务中典型死法：捡起物体后无意义旋转找灯。

**要求**:
- R5.1: egocentric_memory 增加"最后已知位置"追踪
- R5.2: 增加"丢失目标后系统性搜索"的 prompt 策略（旋转 360° 检查每个方向，打开可交互容器）
- R5.3: 修复 rotate 死循环检测（当前 detect_dead_loop 对纯 rotate 序列不敏感）

---

## R6: Guard 调优 + Stage 0

- R6.1: 基于 R2 baseline 数据，分析 cascade_level 分布
- R6.2: 调优 Phase 2 guard
- R6.3: Oracle 自动生成 failure_type_library（目标 20+ 类型）

---

## 优先级

| 顺序 | 需求 | 理由 |
|------|------|------|
| 1 | R1 PickupObject | 所有任务类型的共同 blocker，修一个受益全局 |
| 2 | R2 全任务 Baseline | 必须先知道每种任务卡在哪，才能系统修复 |
| 3 | R3 关键模块测试 | R1/R2 改动大，测试保护与修复并行 |
| 4 | R4 Fork 验证 | R1 修完后有更多可成功的轨迹来触发 fork |
| 5 | R5 空间推理 | R2 baseline 数据会暴露更多具体问题 |
| 6 | R6 Guard + Stage 0 | 依赖前面收集的数据 |

R1 和 R3 可并行（一个改环境层，一个写测试）。R2 在 R1 有初步进展后开始。
