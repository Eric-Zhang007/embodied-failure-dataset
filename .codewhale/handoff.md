# Session Relay

## Goal
使所有 ALFRED 任务类型（pick_and_place, pick_two, examine, heat, cool, clean）在没有 trap 和 injection 的前提下调通。修改必须：治本→改架构，不 hack 单案例；保持第一人称视角；不泄露全知信息。

## In-flight state
全部改动已提交（commit 664ed8d）。

### Key Decisions
1. **方案 A 实施**：VLM 只做擅长的事（物体识别+交互决策），代码层处理导航
2. **MoveAhead 恢复**：VLM 仍可用 MoveAhead 探索，但距离判断由代码层 auto-approach 接管
3. **旋转螺旋检测**：已移至循环头部（每步都检查），不再只在 failure 路径中
4. **旋转疲劳打断**：连续 4 旋转 → 自动 LookAround → 结果仍为 Rotate → 强制 MoveAhead
5. **Auto-explore**：强制 MoveAhead 被阻挡后，代码自动 Rotate+MoveAhead 重试，不经过 VLM/Gate

### Changes Made (commits 87485b0 + 664ed8d)
| Commit | Changes |
|--------|---------|
| `664ed8d` | auto-explore: forced MoveAhead blocked → auto-Rotate+MoveAhead retry |
| `87485b0` | PHASE1_SYSTEM trim, EgocentricMemory compact, auto-approach, rotate-spiral head detection, e2e_test.py --task fix |

### Problems Still Open
1. **重复执行循环（最紧急）**：pick_and_place (Apple→Pan→DiningTable) 中 VLM 成功执行 Apple→Pan 后，无法推进到 "Pan→DiningTable"，陷入 PickupApple→PutObj(Pan)→MoveAhead(blocked)→PickupApple 的无限循环。168 步耗尽。根因：VLM 没有 task-level 完成状态，总是从当前图像推断"下一步是拿起苹果放 Pan 里"而非"Apple→Pan 已完成，下一步是搬 Pan 到 DiningTable"。

2. **FloorPlan301 不可达**：agent 初始位置被 Floor 阻挡。

3. **物体缺失**：某些 ALFRED 2.1 场景在 AI2-THOR 5.0 中缺少目标物体。

### Next Action
修复重复执行问题——核心方案：在 EgocentricMemory 中记录子任务完成状态（"Apple→Pan done: true"），在 Phase 1 prompt 中加入子任务进度条。这样 VLM 知道 Apple→Pan 已经做过了，下一步必须做 Pan→DiningTable。

### Repository State
```
664ed8d feat: auto-explore, forced MoveAhead fallback
87485b0 fix: rotate detection head, auto-approach, rotation fatigue break, memory compact, prompt trim
```
