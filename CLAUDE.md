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
- **仿真**: AI2-THOR 5.0.0，WSLg 优先（`:0`），fallback Xvfb 无头渲染
- **API**: OpenAI 兼容接口 (www.9527code.com/v1)
- **API Key**: sk-IxXjSiRZ3IhCd4hxMajweiXamF0R1U1cHmNECBrRbIz8v0hy
- **Planner (EB) 模型**: gpt-5.5 (reasoning_effort=medium)
- **Executor 模型**: gpt-5.5 (reasoning_effort=medium)
- **Oracle 模型**: gpt-5.5 (reasoning_effort=medium)
- **gridSize**: 0.125m（细粒度移动）
- **visibilityDistance**: 100.0（全场景物体可见）

### 常用命令

```bash
# 单任务测试（随机 episode，semantic memory 默认）
uv run python scripts/e2e_test.py --api-key sk-IxX... --task pick_and_place_simple --random --no-traps

# 使用 geometric (legacy) 记忆
uv run python scripts/e2e_test.py ... --memory geometric

# 全任务并行测试（7 种任务类型）
uv run python scripts/e2e_test.py --api-key sk-IxX... --all --random --no-traps --parallel 3

# 批量跑 pipeline
uv run python scripts/run_pipeline.py --max 10 --api-key sk-IxX... --parallel 3

# 续跑中断的 episode
uv run python scripts/resume_episode.py <episode.json> --api-key sk-IxX... --no-traps --memory semantic

# 自定义 reasoning effort（默认 medium）
uv run python scripts/e2e_test.py --api-key sk-IxX... --all --planner-reasoning-effort xhigh
```

输出目录默认为 `output_e2e_YYYYMMDD_HHMMSS/`（时间戳命名）。

## 架构：Planner + Executor + Oracle 四阶段循环

EB Agent 拆分为 Planner（高层意图）和 Executor（具体动作序列），统一使用 gpt-5.5 模型。Oracle 不知道内部分工，看到统一的 EB 历史。

### Phase 1 — Planner → Executor → Review → Execute

1. **Planner 提议意图** (`eb_agent.py: plan_intent`)：输出高层意图如 `approach AlarmClock`、`pickup AlarmClock`
2. **Executor 提议动作序列** (`executor.py: execute_intent`)：拿到意图 + 完整上下文 → 输出 1-5 个具体动作 + status/reasoning
3. **Planner 审核** (`eb_agent.py: review_actions`)：同意则执行 Executor 的动作，拒绝则 Planner 直接给出 `corrected_actions`
4. **Execute** (`branch_runner.py: _execute_move_sequence`)：顺序执行动作序列，首次失败即停。支持 movement + object interaction（经 action_adapter 解析）。失败时走 Phase 3+4

所有三个组件（Planner plan_intent、Executor execute_intent、Planner review）共享同一上下文：空间记忆 + 视野内物体（含方向标签）+ 最近 5 步历史 + last_error + 图像。

### Phase 2 — Oracle 注入决策（`oracle_agent.py`）

`cascade_level <= 1` 时允许注入；`cascade_level >= 2` 时跳过。注入通过 `env_injector.py` 中的 `inject()` 修改环境。inject 失败自动重试最多 2 次。

### Phase 3 — 失败诊断（Planner）

环境失败后：Planner 诊断（`eb_agent.py: diagnose_failure`），提出恢复动作 + counterfactual。字段归属：`eb_diagnosis`、`eb_counterfactual` 填 Planner 的输出（因为 Planner 有高层视角）。

### Phase 4 — Oracle 评估（`oracle_agent.py`）

评估诊断正确性、counterfactual 等级（WA/PA/AC）、恢复判决、是否创建 fork。

## 核心文件职责

| 文件 | 职责 |
|------|------|
| `src/branch_runner.py` | 主循环：Planner→Executor→Review→Execute + Phase 2-4 + MoveSequence。`run_single_branch()` 是统一入口 |
| `src/eb_agent.py` | Planner：plan_intent（意图）、review_actions（审核）、diagnose_failure（Phase 3）。同时保留旧 propose_action 兼容初始 LookAround |
| `src/executor.py` | Executor：意图 → 动作序列。共享 Planner 的上下文和身份 |
| `src/oracle_agent.py` | Oracle：Phase 2 注入决策 + Phase 4 评估 |
| `src/semantic_memory.py` | **语义记忆（默认）**：receptacle 分组、新鲜度追踪、任务进度、I 视角 |
| `src/geometric_memory.py` | 几何记忆（legacy）：平面列表 + 方向距离 |
| `src/egocentric_memory.py` | 向后兼容 shim：EgocentricMemory → GeometricMemory |
| `src/memory_interface.py` | 记忆统一接口，`--memory semantic/geometric` 切换 |
| `src/vlm_client.py` | VLM 调用，HTTP-200 5级验证，reasoning_effort fallback，400 重试 |
| `src/env_controller.py` | AI2-THOR Controller + `_fix_visible_bounds()`（修 AI2-THOR 5.0.0 visibleBounds2D bug）+ WSLg/Xvfb 管理 |
| `src/scheduler.py` | 并行调度器：per-worker agent 隔离，ThreadPoolExecutor，Fork 串行 |
| `src/task_conditions.py` | 7 种 ALFRED 任务完成检查 + dead_loop/unrecoverable 检测 |
| `src/action_adapter.py` | objectType→objectId 解析 + PickupObject 不可见对象拦截 |
| `src/env_injector.py` | 6 种注入方法（含 hide_object、swap_object） |
| `src/trap_planner.py` | 陷阱选择，支持 exclude_types、breakable 过滤、Blinds blocklist |
| `src/alfred_scene.py` | ALFRED 场景恢复，gridSize=0.125，renderObjectImage=True，visibilityDistance=100 |
| `src/episode_manager.py` | Episode JSON 增量读写 |
| `src/context_builder.py` | EB/Oracle 历史 JSONL 格式化，按权限过滤字段 |
| `src/alfred_parser.py` | ALFRED traj_data.json 解析 |
| `src/fork_manager.py` | Fork 推理重写 |
| `src/step_recorder.py` | 步骤标准化 + 截图 |

## 数据流

```
ALFRED JSON → alfred_parser → TeleportFull → TrapPlanner
→ Phase 1-4 循环 (branch_runner)
  ├── Planner plan_intent → Executor execute_intent → Planner review
  │     ├── approved → MoveSequence 执行
  │     └── rejected → Planner corrected_actions → MoveSequence 执行
  ├── 成功 → episode_manager.add_step()（eb_reasoning 填 Executor 的输出）
  ├── LookAround → 四向截图 + 四向 metadata → analyze_scan_room 决策
  ├── Done → check_task_complete() → 完成或 done_rejected
  └── 环境失败 → Phase 3 Planner 诊断 + Phase 4 Oracle 评估
       └── counterfactual_grade=AC + should_fork → fork task

Phase 3 字段归属：eb_diagnosis/counterfactual 填 Planner 的输出
Phase 1 字段归属：eb_reasoning 填 Executor 的输出
```

## 关键设计约束

1. **信息权限**：Oracle 不知道 Planner/Executor 分工，看到统一 EB 历史。EB 不知道 fork/注入存在。
2. **visibleBounds2D 过滤**：所有"视野内物体"列表用 `visibleBounds2D`（instance segmentation 渲染，正确处理遮挡）。`visibilityDistance=100` 确保 metadata 包含全场景物体，但 prompt 只展示真正在画面中的。
3. **空间记忆**：累积所有曾 `visibleBounds2D=True` 的物体，按 objectId 存，同 type 异 ID 自动编号。离开视野标为 remembered 而非删除。
4. **MoveSequence**：动作序列顺序执行，首次失败即停。支持 movement + object interaction（经 action_adapter）。
5. **Planner review**：只在有关键错误时拒绝（PickupObject >0.5m、MoveAhead 撞已知障碍、明显远离目标）。STUCK 检测：连续 2+ 次失败时 MoveBack 是正确的。
6. **gridSize=0.125m**：MoveAhead/MoveBack/MoveLeft/MoveRight 均 0.125m/步。
7. **任务完成**：`Done` 只信 `task_conditions.py` 硬检查。prompt 中的 CRITERIA 与检查逻辑一致。
8. **LookAround**：仅允许单次扫描，重复则记录 `model_repeated_lookaround`。
9. **_fix_visible_bounds**：AI2-THOR 5.0.0 的 `process_visible_bounds2D` 在 `instance_detections2D` 赋值前调用。env_controller 每次 step/snapshot 后手动修复。

## 已知待修复问题

1. **部分 teleport 位置差**：ALFRED 的 TeleportFull 可能把 agent 放在墙角/家具边缘，导致所有移动被 Floor 阻挡
2. **Fork 机制未端到端测试**：代码在但 e2e 和 pipeline 默认 `enable_fork=False`
3. **heat/cool/clean 任务未充分测试**
4. **Phase 2 guard**：`cascade_level <= 1` 可能偏严格
5. **Stage 0 未做**：failure_type_library.json 手工 8 种类型
6. **PickupObject objectId 解析**：不可见对象（如容器内的 Apple）可能被 Priority 3 fallback 错误解析，导致 AI2-THOR 报 "target not found"。已移除 PickupObject 的 Priority 4 fallback，但 Priority 3（pickupable 但不 visibleBounds2D）仍可能误解析。

## AI2-THOR 5.0.0 注意

- `SetObjectStatic` 不存在；`BreakObject` 代替做 `immovable_object` trap
- `Blinds` 不能 Close（blocklist）
- `process_visible_bounds2D` 初始化顺序 bug → `_fix_visible_bounds` 手动修
- PickupObject 交互范围 ~0.5m，metadata.visible 不保证可交互
