# PROJECT.md — Embodied Failure Dataset

## 项目定位

生成具身智能体级联失败 + 反事实推理数据集。在 AI2-THOR 仿真器中运行 ALFRED 任务，通过 Oracle Agent 注入陷阱制造失败场景，记录 EB Agent 的诊断、恢复尝试和反事实推理。

**核心理念**: 机器人不仅需要知道"做错了什么"，还需要知道"当时应该怎么做"——这正是 counterfactual 标注的价值。

## 技术概要

- **仿真**: AI2-THOR 5.0.0 无头模式 (Xvfb)，WSL2 Ubuntu 24.04
- **模型**: Qwen/Qwen3-VL-32B-Instruct (硅基流动 API)
- **架构**: 双 Agent 四阶段循环 (EB 提议 → Oracle 注入 → EB 诊断 → Oracle 评估)
- **数据源**: ALFRED json_2.1.0 (7 种任务类型)
- **输出**: episode JSON + 截图 + 失败日志 + counterfactual 标注

详见 `.planning/codebase/` 下的代码库地图。

## 当前状态

系统核心循环可运行，已知 6 个待修复问题 (见 CLAUDE.md)。Fork 机制代码完整但从未端到端验证。零自动化测试。

## 目标用户

- 具身 AI / 机器人学习研究者
- 需要 failure recovery 训练数据的团队
- 反事实推理 (counterfactual reasoning) 研究方向

## 约束

- 必须在 WSL2 中运行 (AI2-THOR 无 Windows build)
- VLM 推理依赖硅基流动 API (网络、付费)
- 单线程执行 (scheduler 虽然定义了 max_parallel，实际串行)
- 32B 模型推理延迟: 单图 ~15-30s，多图 ~60-90s

## 技术决策记录

1. **双 Agent 而非单 Agent**: EB 需要不知道自己被测试（生态效度），Oracle 需要全局视角做准确评估
2. **VLM 而非规则**: 失败诊断和反事实推理需要语义理解，规则系统覆盖不全
3. **同模型双角色**: 用 prompt 区分 EB/Oracle 而非不同模型，降低复杂度
4. **即时注入而非预设**: Oracle 在运行时根据 EB 行为动态决定注入，比预设陷阱更自然
5. **JSONL 失败日志独立于 episode JSON**: 失败是事件流，episode 是状态，两者更新频率和语义不同
