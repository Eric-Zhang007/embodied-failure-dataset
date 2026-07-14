# ROADMAP.md

**项目**: Embodied Failure Dataset
**当前里程碑**: M1 — 系统完善 (2026-06-12)

---

## Milestone 1: 系统完善 ← 当前

**目标**: 稳定主分支 → 空间记忆语义化 → 任务能跑通。

| Phase | 名称 | 对应需求 | 状态 |
|-------|------|---------|------|
| P1 | 系统稳定性 + 意图记忆 | R1, R3, R4 | ✅ 完成 |
| P2 | 空间记忆语义化 | R2 | ← 当前 |
| P3 | 模糊 Intent 分解 + 执行精度 | R3, R1 | 依赖 P2 |
| P4 | 系统性探索 + scan→scan 修复 | R4, R5 | 依赖 P2 |
| P5 | Fork 验证 | R6 | 依赖 P1-P4 |
| P6 | 全任务 Baseline | 全部 | 依赖 P1-P5 |

## Phase 1: 系统稳定性 + 意图记忆 ✅

已完成。改动清单见 STATE.md。

## Phase 2: 空间记忆语义化 ← 当前

**目标**: 空间记忆从纯几何（方向+距离）升级为语义化（物体关系 + 位置描述 + 新鲜度）。

**任务**:
1. egocentric_memory 记录每个物体的 `last_seen_on`（在哪个 receptacle 上）、`last_seen_step`
2. render 输出升级为自然语言
3. 在 Executor/Planner prompt 中接入新记忆格式

**验证**: 查看 E2E prompt 日志，确认记忆输出包含语义信息。

## Phase 3: 模糊 Intent 分解 + 执行精度

**目标**: Planner 拿到语义记忆后，能自动分解 `locate X` 为 `approach <具体地点>`。

**任务**:
1. PLANNER_SYSTEM 加分解规则
2. Executor 拿到接近目标地点的 intent 后正确执行

## Phase 4-6

| Phase | 内容 |
|-------|------|
| P4 | 初始探索时系统性遍历房间区域；scan 后短暂禁 scan |
| P5 | enable_fork=True 验证 |
| P6 | 7 种任务各 5 条 E2E baseline |

---

## Milestone 2: 规模数据生成 (规划中)

**前置**: M1 完成，E2E 通过率 > 30%。

---

## Milestone 3: 分析与发布 (规划中)
