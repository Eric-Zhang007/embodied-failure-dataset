---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: complete
stopped_at: runtime-debug phase complete: E2E validated task_complete (2026-07-15)
last_updated: "2026-07-15T15:00:00.000Z"
---

# STATE.md — Project Memory

**Last updated**: 2026-07-15
**Current milestone**: M1 — 系统完善
**Current phase**: P2 (空间记忆语义化 — 模块完成，任务级集成待做)

---

## M1 已完成

### P1: 系统稳定性 + 意图记忆 ✅
### Runtime-Debug Phase ✅
### P2: 语义记忆 (模块完成 🟡)

---

## 已完成的改动 (M1 全部)

| 类别 | 改动 | 文件 |
|------|------|------|
| **记忆架构** | 记忆拆分为 3 层：`memory_interface` + `geometric_memory` + `semantic_memory` | memory_interface, geometric_memory, semantic_memory |
| **记忆架构** | 语义记忆：receptacle 分组、新鲜度追踪、任务进度、"I" 视角 | semantic_memory |
| **记忆架构** | 几何记忆：legacy flat-list，`--memory geometric` 可用 | geometric_memory |
| **记忆架构** | egocentric_memory → backward-compat shim | egocentric_memory |
| **记忆架构** | `--memory` CLI flag 接入所有脚本 + Scheduler | e2e_test, run_pipeline, resume_episode, scheduler |
| **Prompt** | 系统 prompt 全面切换为第一人称 "I" | eb_agent, executor |
| Bug 修复 | `_META_ACTIONS` 拦截 Done/LookAround 泄漏 | branch_runner |
| Bug 修复 | LookAround 四向扫描 metadata 全部录入记忆 | branch_runner |
| Bug 修复 | Camera horizon epsilon 修复 float 边界 | branch_runner |
| Bug 修复 | PickupObject 移除 Priority 4 fallback（不可见容器内对象） | action_adapter |
| 架构 | `analyze_scan_room` — 4-image VLM 分析→方向+intent | eb_agent |
| 架构 | scan room / Done 在 branch_runner 层拦截，不交 Executor | branch_runner |
| 架构 | Recovery 执行 — Phase 3 恢复动作真正被执行 | branch_runner |
| Executor | `repeat` 支持 — 批量同向移动 | executor |
| VLM | HTTP-200 5 级验证 + reasoning_effort auto-strip fallback + 400 重试 | vlm_client |
| VLM | API 迁移：api.fullcupai.com → www.9527code.com/v1 | 全部 |
| VLM | reasoning_effort 默认 xhigh→medium | 全部 |
| 调度 | Scheduler per-worker agent 隔离 + 统计修复 + 9 tests | scheduler |
| 基础设施 | `replay_steps` 支持 MoveSequence/LookAround 回放 | branch_runner |
| 基础设施 | `resume` 支持 executor_agent + 正确 replay | branch_runner |
| 场景 | ALFRED pose 按 type+最近位置 fallback | alfred_scene |

## 已验证

- pick_and_place_simple: task_complete (SoapBottle→Toilet, 6 steps + Book→Desk, 19 steps)
- pick_heat_then_place_in_recep: partial (63 steps, Fridge navigation, objectId resolution issue identified)
- knife→sink (movable_recep): repeat 生效，任务未完成

## 已知待解决问题

| # | 问题 | 表现 |
|---|------|------|
| 1 | Model 视觉误判：Tomato 与 Apple 混淆 | 红色圆形物体在 CounterTop 上 → Planner 当成 Apple 尝试 pickup，实际 Apple 在 Fridge 里 |
| 2 | PickupObject 不可见容器内对象误解析 | Priority 3 (pickupable but no visibleBounds2D) 仍可能错误解析 |
| 3 | Scan→scan 循环 — 初始扫描后 Planner 立即又要求扫描 | 浪费一轮 |
| 4 | Phase 3 Planner 偶尔幻觉任务完成 | 撞柜子→"已经在水槽里了" |
| 5 | heat/cool/clean 任务未充分 E2E 测试 | — |
| 6 | Fork 机制未端到端测试 | enable_fork=False |

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

Runtime-debug phase complete. All 6 tasks done. Pipeline end-to-end validated with pick_and_place_simple (SoapBottle→Toilet): task_complete in 6 steps. Ready for broader runs or next ROADMAP phase.

## Session Continuity

Last session: 2026-07-15T15:00:00.000Z
Stopped at: M1 milestone summary generated. Semantic memory + directional scan fix validated E2E (Book→Desk, 19 steps task_complete). 58/58 tests. MILESTONE_SUMMARY-vM1 written to .planning/reports/.
Resume file: `.planning/.continue-here.md`
Milestone summary: `.planning/reports/MILESTONE_SUMMARY-vM1.md`
