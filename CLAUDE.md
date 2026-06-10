# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境与运行

项目必须在 WSL2 中运行（AI2-THOR 无 Windows build）。所有命令在 WSL2 Ubuntu 24.04 中执行。

```bash
wsl
cd ~/embodied-failure-dataset
export PATH="$HOME/.local/bin:$PATH"
```

- **Python**: 3.10（通过 uv 管理）
- **仿真**: AI2-THOR 5.0.0 无头模式（Xvfb 渲染），运行前自动检测 DISPLAY
- **API**: 硅基流动 (api.siliconflow.cn)，OpenAI 兼容接口
- **API Key**: sk-umtqhyponkhgyhlbwyykumcqebzqxpkcheexbhxgcybsgypa
- **EB 模型**: Qwen/Qwen3-VL-32B-Instruct
- **Oracle 模型**: Qwen/Qwen3-VL-32B-Instruct

### 常用命令

```bash
# 端到端测试（单条轨迹，不启用 fork）
uv run python scripts/e2e_test.py --api-key sk-xxx
uv run python scripts/e2e_test.py --api-key sk-xxx --task pick_and_place_simple
uv run python scripts/e2e_test.py --api-key sk-xxx --no-traps

# 批量跑 pipeline
uv run python scripts/run_pipeline.py --max 10 --api-key sk-xxx
uv run python scripts/run_pipeline.py --task pick_and_place_simple --api-key sk-xxx
uv run python scripts/run_pipeline.py --max 50 --no-fork --api-key sk-xxx

# API 推理速度测试
uv run python scripts/bench_api.py --api-key sk-xxx

# 下载 ALFRED 数据
uv run python scripts/download_alfred.py
```

## 架构：双 Agent 四阶段循环

每一步交互经历 Phase 1-4。所有 Agent 通过 `VLMClient` 调用 VLM（支持 ollama 和 openai 兼容后端，实际使用 siliconflow）。image 传 base64 编码的 numpy array。完整 API 请求写入 `logs/api_calls.jsonl`，摘要写入 `logs/api_calls.log`。

### Phase 1 — EB Agent 提议动作（`eb_agent.py`）

输入：任务目标 + 当前截图 + 可见物体列表 + 完整 EB 可见历史 + 上次错误 + `HAND STATUS` + `TASK COMPLETION CRITERIA`。响应格式：`{"action": "...", "params": {}, "reasoning": "..."}`。prompt 中列出了完整可用动作集合，严禁模型幻觉出不存在的动作名。

`LookAround` 是正常动作，但由 `branch_runner.py` 特殊执行：原地收集 ahead/left/behind/right 四张图，写成一个成功 step，再调用 EB 的多图 prompt 决定下一步。多图请求中每张图前都有单独方向标签：`VIEW 1: ahead`、`VIEW 2: left`、`VIEW 3: behind`、`VIEW 4: right`。如果模型看完后仍输出 `LookAround`，不会重复扫描；系统记录 `model_repeated_lookaround`，并把错误反馈给下一轮普通 Phase 1。

### Phase 2 — Oracle 相机注入决策（`oracle_agent.py`）

`cascade_level == 0` 时正常执行；`cascade_level == 1` 时允许注入（宽松守卫，设计意图是避免恢复期间叠加注入）；`cascade_level >= 2` 时跳过注入。注入通过 `env_injector.py` 中的 `inject()` 立即修改环境，支持：set_object_property、lock_container、occlude_object、remove_object、swap_object。inject 失败会自动重试，最多 2 次。

### Phase 3 — EB 失败诊断（`eb_agent.py`）

诊断为何失败 + 提出恢复动作 + 可选 counterfactual（格式：`{"target_step": int, "alternative_action": {...}, "reasoning": "..."}`）。Phase 3 prompt 同样包含完整 EB 可见历史、`HAND STATUS` 和 `TASK COMPLETION CRITERIA`，并要求所有文本字段使用第一人称。

### Phase 4 — Oracle 评估（`oracle_agent.py`）

评估诊断正确性、counterfactual 等级（WA/PA/AC）、恢复判决（recovered/recoverable/unrecoverable）、是否创建 fork。Oracle prompt 使用完整 EB + Oracle 可见历史。`branch_runner.py` 还有代码层死循环检测，触发后直接结束为 `dead_loop_unrecoverable`。

## 核心文件职责

| 文件 | 职责 |
|------|------|
| `src/branch_runner.py` | 单分支 Phase 1-4 循环 + 成功/失败步写入。`run_single_branch()` 是 scheduler 和 e2e_test 的共同入口 |
| `src/scheduler.py` | 全局分支队列（main + fork），委托 `run_single_branch()` 执行 |
| `src/vlm_client.py` | VLM 调用客户端，含 timeout、JSON 解析自动重试、完整 API JSONL 日志、多图方向标签 |
| `src/env_controller.py` | AI2-THOR Controller 封装 + Xvfb 自动管理 |
| `src/env_injector.py` | 5 种注入方法实现（set_object_property 等） |
| `src/episode_manager.py` | Episode JSON 增量读写（每步立即 flush） |
| `src/task_conditions.py` | ALFRED 任务完成条件检查 + prompt 中的完成标准文案 |
| `src/context_builder.py` | EB / Oracle 历史上下文格式化，避免各 prompt 自己切片历史 |
| `src/trap_planner.py` | 从 failure_type_library.json 选陷阱，支持 `exclude_types` 排除任务目标物体 |
| `src/action_adapter.py` | objectType/receptacleType → objectId 解析 + AI2-THOR 2.1→5.0 废弃参数清理 |
| `src/alfred_parser.py` | ALFRED traj_data.json 解析（api_action 支持 string 和 dict 两种格式） |
| `src/fork_manager.py` | Fork 分岔点推理重写（让 Oracle 为替代动作生成自洽推理文本） |
| `src/step_recorder.py` | 步骤标准化 + 截图保存 |

## 数据流

```
ALFRED JSON → alfred_parser → init_action(TeleportFull) → TrapPlanner 选陷阱
→ Phase 1-4 循环 (branch_runner)
  ├── 成功步 → episode_manager.add_step() → 写 JSON
  ├── LookAround → 原地四向截图 → 写成功 step → EB 多图决策
  ├── Done → task_conditions.check_task_complete() → 成功结束或写 done_rejected
  ├── 无效动作 → failure log (JSONL)
  └── 环境失败 → Phase 3 诊断 + Phase 4 评估 → 写 JSON（含完整诊断字段）
       └── counterfactual_grade=AC 且 should_fork → 生成 fork task 加入队列
```

续跑：`BranchRunner.resume()` 加载已有 JSON → replay 成功步恢复环境 → 继续 Phase 1-4。

## 关键设计约束

1. **信息权限**：EB Agent 不知道 fork 存在、不知道注入存在。Oracle 知道一切（fork 分支上也知道自己是 fork）。分岔点前 context 共享，分岔后隔离。
2. **Fork 推理重写**：fork root step 的 `eb_reasoning` 由 Oracle 重写，不能提及"反事实""替代"等词。
3. **step_id 格式**：main 分支为 `s0, s1, s2...`，fork 分支也从 `s0` 开始（通过 branch_id 区分）。
4. **动作参数**：EB 输出 `objectType` / `receptacleType`，不要输出坐标或 objectId；`action_adapter.py` 根据当前 metadata 解析成 AI2-THOR 需要的 objectId。
5. **任务完成条件**：`Done` 只信 `task_conditions.py` 的硬检查。prompt 中的 `TASK COMPLETION CRITERIA` 必须和同一套检查逻辑保持一致。
6. **LookAround**：只允许单次四向扫描。重复 `LookAround` 是模型无效决策，不再重复扫同一圈图。

## 已知待修复问题

1. **PickupObject 持续失败**：元数据标为 visible 但 PickupObject 返回 "not found"，可能是距离、遮挡或 AI2-THOR 可交互状态问题。需要继续用 e2e 日志和截图定位。

2. **EB 空间推理不稳定**：即使多图输入正确，模型仍可能看错方向、距离或可达路径。提示词只能部分缓解，需要继续比较 8B/32B 的 e2e 表现。

3. **Fork 机制未端到端测试**：代码在但 e2e 和 pipeline 默认 `enable_fork=False`。当前调 prompt 和单 branch 生成级联失败数据，暂不调反事实 fork。

4. **Phase 2 guard**：`cascade_level <= 1` 守卫可能仍偏严格。

5. **调度器 max_parallel**：参数存在但始终串行执行。

6. **Stage 0 未做**：failure_type_library.json 是手工写的 8 种失败类型，应该让 Oracle 阅读 AI2-THOR 文档自动生成。

## 最近补丁

1. EB prompt 统一加入 `HAND STATUS` 和 `TASK COMPLETION CRITERIA`，普通 Phase 1、LookAround 多图决策、Phase 3 失败恢复复用同一个上下文生成函数。
2. `task_conditions.py` 负责 ALFRED 任务完成检查，所有 checker 返回 `(bool, reason)`。`Done` 被拒时会写具体缺失条件。
3. `get_completion_criteria_text()` 复用 `_official_task_type()`，避免 movable receptacle 任务的文案和真实检查逻辑不一致。
4. `LookAround` 不再循环重复扫描；模型重复输出 `LookAround` 时记录 `model_repeated_lookaround` 并回到普通 Phase 1。
5. 多图请求给每张图单独加方向标签：`VIEW 1: ahead`、`VIEW 2: left`、`VIEW 3: behind`、`VIEW 4: right`。
6. `TrapPlanner._find_object()` 支持 `exclude_types`，`run_single_branch` 中调用 `_extract_task_target_types()` 提取 pddl_params 中的目标物体类型。
7. 新增动作：MoveLeft, MoveRight, MoveBack, BreakObject, FillObjectWithLiquid, EmptyLiquidFromObject。已加入 `_VALID_ACTIONS` 和 prompt。
