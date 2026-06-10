# ROADMAP.md

**项目**: Embodied Failure Dataset
**当前里程碑**: M1 — 系统完善 (2026-06-10)

---

## Milestone 1: 系统完善 ← 当前

**目标**: 解决 PickupObject 这个全局 blocker，建立 7 种任务类型的 baseline，关键模块有测试，Fork 验证通过。

| Phase | 名称 | 对应需求 | 依赖 |
|-------|------|---------|------|
| P1 | PickupObject 诊断与修复 | R1 | 无 |
| P2 | 关键模块测试 | R3 | 无（与 P1 并行） |
| P3 | 全任务类型 Baseline | R2 | P1 初步修复后 |
| P4 | Fork 端到端验证 | R4 | P1, P3 |
| P5 | 空间推理改善 | R5 | P3（需要 baseline 数据） |
| P6 | Guard 调优 + Stage 0 | R6 | P3 |

**执行顺序**: P1 和 P2 可同时开工。P3 等 P1 有初步结论后跑。P4-P6 按顺序。

## Phase 1: PickupObject 诊断与修复

**为什么先做**: 所有 7 种任务类型的第一步都是 PickupObject。它不通，其他都免谈。

**任务**:
1. 分析 AI2-THOR visibility 机制：对比 metadata.visible、visibilityDistance、实际可交互距离
2. 插桩 env_controller.step()，收集 PickupObject 失败时的完整诊断（距离、遮挡物、inventory 状态）
3. 在 action_adapter 中增加距离预检
4. 根据诊断数据确定根因并修复
5. 跑 20 条 pick_and_place_simple 验证

## Phase 2: 关键模块测试

**与 P1 并行执行。**

**任务**:
1. 配置 pytest
2. task_conditions.py 测试（7 checkers + edge cases）
3. action_adapter.py 测试
4. context_builder.py 测试
5. episode_manager.py 测试
6. trap_planner.py 测试
7. alfred_parser.py 测试

## Phase 3: 全任务类型 Baseline

**前置**: P1 PickupObject 有初步修复。

**任务**:
1. 对 7 种任务类型各跑 3 条 E2E（无陷阱）
2. 记录每种类型的成功率和典型失败模式
3. 修正有 bug 的 completion criteria
4. 生成 `E2E_BASELINE.md` 报告

## Phase 4-6

| Phase | 内容 |
|-------|------|
| P4 | enable_fork=True 跑 30 条，验证隔离、推理重写、fork 完成率 |
| P5 | egocentric_memory 增强 + 系统性搜索 prompt + rotate 死循环修复 |
| P6 | cascade_level 分析 + guard 调优 + Oracle 自动生成 failure library |

---

## Milestone 2: 规模数据生成 (规划中)

**前置**: M1 完成，PickupObject 成功率 > 50%，至少 3 种任务类型有稳定成功率。

| Phase | 内容 |
|-------|------|
| P7 | Pipeline 稳健性（resume/checkpoint/超时） |
| P8 | max_parallel 实现 |
| P9 | 500 条批量生成 |
| P10 | 数据质量审核 |

---

## Milestone 3: 分析与发布 (规划中)

| Phase | 内容 |
|-------|------|
| P11 | 失败模式统计 |
| P12 | Counterfactual 人工评估 |
| P13 | 数据格式文档 + 示例 |
| P14 | 论文图表 + 写作 |
