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
| R3 intent 级记忆（Planner 意图历史 + Executor 意图作用域） | ✅ |
| R4 Executor repeat 支持 + 距离公式 | ✅ |
| R4 Executor SOLID 障碍物标签 | ✅ |
| R4 Executor max_tokens 不再截断 + sanitize 安全网 | ✅ |
| 基础设施 replay_steps/resume 修复 | ✅ |

## 待完成

### R1: 执行精度

**问题**: Executor 收到 `approach SinkBasin, 1.3m ahead` → 输出 `repeat=8` (1.0m)，不到目标。且无视 SOLID 标签，直线撞家具。

**方向**: prompt 加强"先看有没有 SOLID 物体挡路，有则绕行"；repeat 计算直觉是否可训练——待观察。

### R2: 空间记忆语义化

**现状**: 空间记忆只有方向+距离（"Knife hidden, ahead-right 1.8m"）。缺：
- 物体之间的关系（"Knife 在 DiningTable 上"）
- 最后一次看到的位置和时间
- 物体属于哪个区域（厨房/餐厅/卧室）

**方向**: egocentric_memory 记录每个物体的 `last_seen_on`（在哪个 receptacle 上看到）、`last_seen_at`（时间戳/步数）。render 时生成自然语言描述。

### R3: 模糊 Intent 分解

**问题**: Planner 输出 `locate Mug` → Executor 不知道 Mug 在哪 → zigzag 盲走。如果空间记忆说"Mug 上次在 Counter 上"，Planner 应该输出 `approach Counter` 而非 `locate Mug`。

**方向**: PLANNER_SYSTEM 加规则——当 intent 是 "locate X" 时，先查空间记忆，输出具体地点 intent。

### R4: 系统性探索

**问题**: 目标从未见过时（初始扫描没覆盖），没有探索策略——不知道"先检查餐桌、再检查柜台、再检查水槽"。

**方向**: Executor/Planner 需要一个空间划分概念——把可见区域分成几个象限，逐个探索。

### R5: scan→scan 循环

**问题**: 初始扫描后 Planner 立即又要求扫描，浪费一轮。

**方向**: 初始扫描的 intent 直接执行（跳过 Planner），或扫描后短暂禁 scan。

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
