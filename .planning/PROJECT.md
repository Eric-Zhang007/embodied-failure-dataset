# PROJECT.md — Embodied Failure Dataset

**更新**: 2026-06-10 (重写)

## 项目定位

生成具身智能体级联失败 + 反事实推理数据集。在 AI2-THOR 5.0.0 中运行 ALFRED 7 种任务，通过 Oracle Agent 注入陷阱制造失败场景，记录完整诊断、恢复尝试和反事实标注。

## 技术概要

- **仿真**: AI2-THOR 5.0.0，WSLg 优先 (`:0`)，Xvfb fallback (`:99`)，gridSize=0.125m，visibilityDistance=100
- **模型**: OpenAI 兼容 API (www.9527code.com/v1)
  - Planner (EB): gpt-5.5 — 高层意图 + action 审核 (reasoning_effort=medium)
  - Executor: gpt-5.5 — 具体动作序列 (reasoning_effort=medium)
  - Oracle: gpt-5.5 — 注入决策 + 失败评估 (reasoning_effort=medium)
- **架构**: Planner→Executor→Review→Execute 四阶段循环
- **数据源**: ALFRED json_2.1.0 (7 种任务类型)
- **输出**: episode JSON + 截图 + 失败日志 + counterfactual 标注
- **核心模块**: 语义记忆 (semantic_memory)、几何记忆 (geometric_memory)、visibleBounds2D 过滤、LookAround cooldown

详见 `.planning/codebase/` 下的代码库地图。

## 当前状态

核心循环稳定运行。visibleBounds2D 修复已实施。语义记忆模块完成（receptacle 分组 + I 视角），几何记忆保留作为 fallback。HTTP-200 响应硬化完成（5 级验证 + reasoning_effort fallback + 400 重试）。Scheduler 已做 per-worker agent 隔离。E2E 验证通过：pick_and_place_simple task_complete (6 steps, 19 steps)。58 项测试通过。heat/cool/clean 任务尚未充分 E2E 测试。Fork 机制未端到端验证。

## 目标用户

- 具身 AI / 机器人学习研究者
- 需要 failure recovery 训练数据的团队
- 反事实推理研究方向

## 约束

- 必须在 WSL2 中运行 (AI2-THOR 无 Windows build)
- VLM 推理依赖 OpenAI 兼容 API（网络、付费）
- gpt-5.5 推理: 单图 ~5-15s，多图 ~10-30s
- 每步交互 3-4 次 VLM 调用（Planner + Executor + Reviewer + 可能的 Phase 3/4）
- ThreadPoolExecutor 实现 max_parallel，per-worker agent 隔离

## 技术决策记录

1. **Planner+Executor 拆分而非单 Agent**: Planner 管高层意图和审核，Executor(8B) 专做动作序列，降低 32B 调用频率
2. **Oracle 不知道内部分工**: 看到统一 EB 历史，不暴露 Planner/Executor 拆分
3. **空间记忆而非每步重置**: 累积所有曾 visibleBounds2D=True 的物体，按 objectId 去重编号，离开视野标为 remembered
4. **visibleBounds2D 过滤**: 用 instance segmentation 渲染判断真正可见物体，`visible` 只在场景恢复/注入时用
5. **Planner review 关键错误拦截**: 只在 PickupObject>0.5m、MoveAhead 撞已知障碍、明显远离目标时拒绝
6. **gridSize=0.125m**: 细粒度移动，每步 0.125m
7. **即时注入而非预设**: Oracle 在运行时动态决定注入时机和类型
8. **JSONL 失败日志独立于 episode JSON**: 失败是事件流，episode 是状态
