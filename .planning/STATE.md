# STATE.md — Project Memory

**Last updated**: 2026-06-12
**Current milestone**: M1 — 系统完善
**Current phase**: P1 (稳定主分支 + 意图级记忆 + 障碍物感知)

---

## 已完成的改动 (P1 第一轮)

| 类别 | 改动 | 文件 |
|------|------|------|
| Bug 修复 | `_META_ACTIONS` 拦截 Done/LookAround 泄漏 | branch_runner |
| Bug 修复 | `detect_dead_loop` 支持嵌套参数 hash | task_conditions |
| Bug 修复 | `step_recorder` 安全访问 `error`/`frame` | step_recorder |
| 架构 | `analyze_scan_room` — 32B 多图分析→方向+intent | eb_agent |
| 架构 | scan room / Done 在 branch_runner 层拦截，不交 Executor | branch_runner |
| 架构 | Recovery 执行 — Phase 3 恢复动作真正被执行 | branch_runner |
| 架构 | 初始 LookAround 走新 scan 分析路径 | branch_runner |
| 记忆 | 意图级历史 — Planner 看意图摘要，Executor 看当前意图步骤 | branch_runner, eb_agent |
| Executor | `repeat` 支持 — 批量同向移动，输出量降 ~15 倍 | executor |
| Executor | SOLID 障碍物标签 — 家具类标"不可通行" | executor |
| Executor | 距离→repeat 计算公式 | executor |
| Executor | `_sanitize_actions` 安全截断 | executor |
| Prompt | Phase 3 注入当前 intent 上下文 | eb_agent |
| Prompt | JSON 重试反馈改为"输出更少"而非"输出更多" | vlm_client |
| 基础设施 | `replay_steps` 支持 MoveSequence/LookAround 回放 | branch_runner |
| 基础设施 | `resume` 支持 executor_agent + 正确 replay | branch_runner |

## 已验证

- pick_and_place_simple: task_complete (2 steps, 运气好)
- knife→sink (movable_recep): 多次跑通无崩溃，repeat 生效，但任务本身未完成

## 已知待解决问题

### 阻塞级

| # | 问题 | 表现 |
|---|------|------|
| 1 | Executor 忽略 SOLID 标签，仍走直线撞家具 | repeat=8 直冲 Cabinet/DiningTable |
| 2 | Executor 数错 repeat（1.3m 应 11，输出 8） | 走不到目标 |
| 3 | scan→scan 循环 — 初始扫描后 Planner 立即又要求扫描 | 浪费一轮 |
| 4 | Phase 3 Planner 偶尔幻觉任务完成 | 撞柜子→"刀已经在水槽里了" |

### 设计级

| # | 问题 | 方向 |
|---|------|------|
| 5 | 空间记忆缺语义关联 | 只存方向+距离，不存"在桌子上"/"在隔壁房间" |
| 6 | 模糊 intent 无分解机制 | "locate Mug"→Executor zigzag 盲走；Planner 应分解为"上次在 Counter→先去 Counter" |
| 7 | 目标从未见过时无探索策略 | 应该系统性遍历房间（先看餐桌→再看柜台→再看水槽），而非随机走 |

## 当前架构

```
s0: 初始 LookAround → 32B 多图分析 → 方向+intent → 旋转到位
while True:
  Planner(32B) → intent (含 scan room / Done 拦截)
    ├── scan room → 4-view → 32B 分析 → 方向+intent → Executor
    ├── Done → task_conditions 验证 → complete / 拒绝
    └── 其他 → Executor(8B, repeat 支持, SOLID 标签)
         → Planner Review → Execute MoveSequence
         ├── 成功 → continue
         └── 失败 → Phase 3(32B, 有 intent 上下文) + Phase 4(Oracle 评估)
              ├── unrecoverable → 结束
              ├── recovered → continue
              └── recoverable → 执行恢复动作 → continue
```

## Next

讨论空间记忆语义化方案 → 改 prompt → 继续 E2E 验证
