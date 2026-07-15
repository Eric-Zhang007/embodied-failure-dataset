# REQUIREMENTS.md — Milestone: 系统完善

**日期**: 2026-06-12 (更新)
**目标**: 稳定主分支，提升任务完成率，为规模数据生成做准备。

---

## 已完成

| 需求 | 状态 |
|------|------|
| R1 LookAround/Done 泄漏修复 | ✅ |
| R1 初始 LookAround 路径 | ✅ |
| R1 Recovery 执行 | ✅ |
| R1 Camera horizon epsilon fix | ✅ |
| R1 PickupObject Priority 4 fallback 移除 | ✅ |
| R2 记忆架构拆分为 3 层（interface + geometric + semantic） | ✅ |
| R2 语义记忆 receptacle 分组 + 新鲜度追踪 + 任务进度 | ✅ |
| R2 四向扫描 metadata 全部录入记忆 | ✅ |
| R2 `--memory` CLI flag 接入全部脚本 | ✅ |
| R2 系统 prompt 全面切换第一人称 "I" | ✅ |
| R3 intent 级记忆（Planner 意图历史 + Executor 意图作用域） | ✅ |
| R4 Executor repeat 支持 + 距离公式 | ✅ |
| R4 Executor SOLID 障碍物标签 | ✅ |
| R5 scan→scan 循环（初始 LookAround 拦截） | ✅ |
| 基础设施 replay_steps/resume 修复 | ✅ |
| Scheduler per-worker agent 隔离 + 统计修复 | ✅ |
| HTTP-200 5 级验证 + reasoning_effort fallback + 400 重试 | ✅ |
| VLM API 迁移 www.9527code.com/v1 | ✅ |
| 58 项单元测试 | ✅ |

## 待完成

### R2: 空间记忆语义化（任务级集成）

**已完成**: 记忆模块架构（semantic_memory 输出 receptacle 分组 + 新鲜度 + 任务进度 + I 视角）。
**待做**: Planner prompt 级集成 — 当记忆说 "Apple last seen inside Fridge, 63 steps ago"，Planner 应自动生成 "open Fridge" 而非 "approach CounterTop"（因为看见红色 Tomato 就误认为是 Apple）。

### R3: 模糊 Intent 分解

**问题**: Planner 输出 `locate Mug` → Executor 不知道 Mug 在哪 → zigzag 盲走。如果空间记忆说"Mug 上次在 Counter 上"，Planner 应该输出 `approach Counter` 而非 `locate Mug`。

**方向**: PLANNER_SYSTEM 加规则 — 当 intent 是 "locate X" 时，先查空间记忆，输出具体地点 intent。

### R4: 系统性探索

**问题**: 目标从未见过时（初始扫描没覆盖），没有探索策略。

**方向**: 把可见区域分成几个象限，逐个探索。

### R5: scan→scan 循环

**问题**: 初始扫描后 Planner 立即又要求扫描，浪费一轮。
**部分解决**: 初始 LookAround 已拦截，但 mid-episode scan 守卫（仅允许 1 次）未完成。

### R6: Fork 验证 (后续)

**前置**: R1-R5 完成，主分支稳定。

## 优先级

| 顺序 | 需求 | 理由 |
|------|------|------|
| 1 | R2 空间记忆语义化 | R3 模糊分解依赖它 |
| 2 | R3 模糊 Intent 分解 | 解决 zigzag 盲走 |
| 3 | R1 执行精度 | repeat 计数 + SOLID 绕行 |
| 4 | R5 scan→scan 循环 | 小修 |
| 5 | R4 系统性探索 | 新功能，依赖 R2 |

R2+R3 是关键——有了语义记忆，模糊 intent 自然能分解。
