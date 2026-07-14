# PROJECT.md — Embodied Failure Dataset

**更新**: 2026-06-10 (重写)

## 项目定位

生成具身智能体级联失败 + 反事实推理数据集。在 AI2-THOR 5.0.0 中运行 ALFRED 7 种任务，通过 Oracle Agent 注入陷阱制造失败场景，记录完整诊断、恢复尝试和反事实标注。

## 技术概要

- **仿真**: AI2-THOR 5.0.0，WSLg 优先 (`:0`)，Xvfb fallback (`:99`)，gridSize=0.125m，visibilityDistance=100
- **模型**: 硅基流动 API
  - Planner (EB): Qwen/Qwen3-VL-32B-Instruct — 高层意图 + action 审核
  - Executor: Qwen/Qwen3-VL-8B-Instruct — 具体动作序列
  - Oracle: Qwen/Qwen3-VL-32B-Instruct — 注入决策 + 失败评估
- **架构**: Planner→Executor→Review→Execute 四阶段循环
- **数据源**: ALFRED json_2.1.0 (7 种任务类型)
- **输出**: episode JSON + 截图 + 失败日志 + counterfactual 标注
- **核心模块**: 空间记忆 (egocentric_memory)、visibleBounds2D 过滤、LookAround cooldown

详见 `.planning/codebase/` 下的代码库地图。

## 当前状态

核心循环可运行，visibleBounds2D 修复已实施。E2E --all 执行中发现新 bug：Executor 输出的 LookAround 动作被直接传给 AI2-THOR 导致崩溃。零自动化测试。Fork 机制未端到端验证。

## 目标用户

- 具身 AI / 机器人学习研究者
- 需要 failure recovery 训练数据的团队
- 反事实推理研究方向

## 约束

- 必须在 WSL2 中运行 (AI2-THOR 无 Windows build)
- VLM 推理依赖硅基流动 API（网络、付费）
- 32B 推理: 单图 ~5-15s，多图 ~10-30s；8B ~3-10s
- 每步交互 3-4 次 VLM 调用（Planner + Executor + Reviewer + 可能的 Phase 3/4）
- ThreadPoolExecutor 实现 max_parallel

## 技术决策记录

1. **Planner+Executor 拆分而非单 Agent**: Planner 管高层意图和审核，Executor(8B) 专做动作序列，降低 32B 调用频率
2. **Oracle 不知道内部分工**: 看到统一 EB 历史，不暴露 Planner/Executor 拆分
3. **空间记忆而非每步重置**: 累积所有曾 visibleBounds2D=True 的物体，按 objectId 去重编号，离开视野标为 remembered
4. **visibleBounds2D 过滤**: 用 instance segmentation 渲染判断真正可见物体，`visible` 只在场景恢复/注入时用
5. **Planner review 关键错误拦截**: 只在 PickupObject>0.5m、MoveAhead 撞已知障碍、明显远离目标时拒绝
6. **gridSize=0.125m**: 细粒度移动，每步 0.125m
7. **即时注入而非预设**: Oracle 在运行时动态决定注入时机和类型
8. **JSONL 失败日志独立于 episode JSON**: 失败是事件流，episode 是状态
