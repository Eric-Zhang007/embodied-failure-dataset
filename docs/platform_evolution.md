# 平台演进蓝图：长程任务 + 多模拟器 + 闭环测评

> 状态：规划文档（draft）。文中标注「已有」的内容以当前代码为准；其余为演进目标，不代表已实现。
> 更新日期：2026-08-06

## 1. 背景与目标

当前仓库是一个「具身失败数据集」采集工程：在 AI2-THOR + ALFRED 场景上运行 Planner+Executor VLM 智能体，记录每一步的推理、动作、失败与恢复，并沉淀为可微调的数据。

演进目标是把这套东西做成一个更具 infra 意味的平台，支撑三件事：

1. **长程任务**：把现有单任务（如 pick_and_place、heat、examine）组合成多阶段任务，而不是只能跑一个 ALFRED 原子任务。
2. **多模拟器后端**：在保持同一套 agent / 评测逻辑的前提下，切换 AI2-THOR、RoboTHOR、MolmoSpaces 等后端，覆盖更多场景或 sim2real 验证。
3. **闭环测评平台**：固定任务集与预算，标准化运行产物，计算指标，形成 leaderboard，并让「失败 → 数据 → 微调 → 复测」的闭环可重复执行。

## 2. 现状盘点（对照代码）

### 2.1 已经具备的基础

| 能力 | 位置 | 说明 |
|------|------|------|
| 环境封装 | `src/env_controller.py` | 唯一的 AI2-THOR 接触面：`step()`、`get_state_snapshot()`、`reset_to_alfred_scene()` |
| 任务元信息 | `src/alfred_parser.py` | 解析 ALFRED traj：pddl、场景、low_actions、`goal_instances`（实例级目标，2026-08 新增） |
| 完成判定 | `src/task_conditions.py` | 按任务类型检查完成；有实例级目标时精确判定，否则回退类型级 |
| 单任务运行器 | `src/branch_runner.py` | 意图循环、扫描、Done 校验、反事实分支、恢复、oracle 注入、critic |
| 批量调度 | `src/scheduler.py` | 并行 worker、公平队列、断点续跑、pending fork、task lanes、统计 |
| 记忆 | `src/semantic_memory.py` / `src/geometric_memory.py` / `src/egocentric_memory.py` | 多种记忆模式，`--memory semantic|geometric` |
| 失败注入 | `src/env_injector.py` | 陷阱（traps）注入，用于制造失败样本 |
| 运行产物 | `output*/trial_*.json` + `failures_*.jsonl` + `api_calls.jsonl` | episode JSON、失败分支日志、API 调用日志 |
| 数据/FT 闭环 | `scripts/generate_ft_data.py`、`src/context_builder.py` | 把已有 episode 补全 reasoning 用于微调 |
| 消融 | `scripts/run_timed_ablation.py`、`scripts/analyze_ablation.py` | 按 ablation 开关批量对比 |
| 分析 | `scripts/analyze_collection_snapshot.py` | 统计完成率、步数、失败、fork 验证等 |
| 配置 | `config.toml` | `[api]`（base_url、api_key）与模型（planner/executor/oracle） |

### 2.2 当前 episode 记录格式（主要字段）

```jsonc
{
  "episode_id": "trial_...",
  "task_goal": "...",
  "scene": "FloorPlan307",
  "task_type": "pick_and_place",
  "alfred_task_type": "pick_and_place_simple",
  "alfred_task_id": "...",
  "pddl_params": { "object_target": "AlarmClock", "parent_target": "Desk", "...": "" },
  "goal_instances": {           // 2026-08 新增，实例级目标
    "pickup_object_ids": [...],
    "toggle_on_object_ids": [...],
    "put_receptacle_ids": [...],
    "final_put": { "objectId": "...", "receptacleObjectId": "..." }
  },
  "alfred_scene": { "...": "..." },
  "initial_traps": [],
  "runtime_traps": [],
  "semantic_memory_states": {},
  "pending_forks": [],
  "steps": [
    {
      "step_id": "s0", "branch_id": "main", "action": "LookDown",
      "success": true, "error_message": null,
      "eb_reasoning": "...", "eb_diagnosis": null,
      "oracle_injection_decision": null, "...": "..."
    }
  ],
  "final_outcome": {
    "main_branch": { "branch_id": "main", "termination_reason": "task_complete", "total_steps": 18 },
    "forks": []
  },
  "status": "completed"
}
```

`final_outcome.main_branch.termination_reason` 常见取值：`task_complete`、`skipped_pre_satisfied`、`step_hard_limit`、失败类终止等。统计时只把 `task_complete` 算作成功，`skipped_pre_satisfied` 不计入完成率。

### 2.3 已有闭环的雏形

```
ALFRED/AI2-THOR 运行
    → episode/failures/api_calls 产物
    → 失败样本 + 恢复轨迹
    → generate_ft_data 补 reasoning
    → 微调模型（scripts/finetune_qwen3vl.py）
    → 回到运行
```

这是将来闭环测评平台的核心资产，目前手动衔接，缺的是「测评结果 → 数据集筛选 → 训练 → 复测」的自动化与版本化。

## 3. 目标架构

```
┌──────────────────────────────────────────────────────────────┐
│ Leaderboard / 结果门户                                       │
│ 提交 → 跑分 → 指标 → 对比/回归告警                            │
└───────────────▲──────────────────────────────┬───────────────┘
                │ 评测结果                     │ 提交（固定 task suite + budget）
┌───────────────┴──────────────────────────────▼───────────────┐
│ 闭环测评层（Eval Runner）                                     │
│ · task suite registry · seed/budget · 指标计算 · artifact store│
└───────────────▲──────────────────────────────┬───────────────┘
                │ 标准化 run 产物（schema 版本化）│ TaskSpec
┌───────────────┴──────────────────────────────▼───────────────┐
│ 任务/Agent 层                                                 │
│ · Goal 抽象（可组合子目标） · BranchRunner · Scheduler        │
│ · 记忆/恢复/fork/oracle                                      │
└───────────────▲──────────────────────────────┬───────────────┘
                │ obs/action/state 契约         │ 重置/步进
┌───────────────┴──────────────────────────────▼───────────────┐
│ Simulator Backend 层（可插拔）                                │
│ AI2-THOR（现在） · RoboTHOR（sim2real） · MolmoSpaces（大规模）│
└──────────────────────────────────────────────────────────────┘
```

分层原则：**agent 与任务逻辑不直接依赖具体模拟器**；模拟器差异（观测、动作空间、场景加载）被收敛到 Backend 适配层。**评测与运行解耦**：跑分只消费标准化的 run 产物，不关心 agent 内部实现。

## 4. 核心抽象草案

### 4.1 Simulator Backend

把现在的 `EnvController` 抽象成协议。AI2-THOR 是第一个实现，RoboTHOR / MolmoSpaces 各自实现 adapter。

```python
class SimulatorBackend(Protocol):
    obs_spec: ObsSpec          # 帧尺寸、通道、是否 RGB-D、额外传感器
    action_spec: ActionSpec    # 动作空间与参数约束

    def reset(self, task_spec: TaskSpec) -> StepResult: ...
    def step(self, action: str, **params) -> StepResult: ...
    def state_snapshot(self) -> StateSnapshot: ...   # frame + metadata + task_state
    def close(self) -> None: ...

@dataclass
class StepResult:
    success: bool
    error: str | None
    frame: np.ndarray
    metadata: dict          # 归一化后的物体/容器/状态视图
    task_state: dict
```

关键点：
- `metadata` 需要归一化，不能把 AI2-THOR 的字段名（如 `receptacleObjectIds`）泄漏到上层协议；否则换模拟器时 agent 和完成判定都要跟着改。
- RoboTHOR 侧重 sim2real：动作空间与 AI2-THOR 接近但场景/传感器不同；需要保留「对应真实场景」的测评入口。
- MolmoSpaces 侧重规模：23 万+ 室内场景、13 万+ 物体资产、4200 万+ 抓取标注（AI2 2026 发布），动作与物理仿真（MuJoCo 等）不同，适配重点是观测与操作动作契约。

### 4.2 Goal / TaskSpec

长程任务的关键是把「一个 checker 函数」升级为「可组合的 goal 描述」。当前 `task_conditions.check_task_complete(metadata, ep_data, task_state)` 已经以实例级目标（`goal_instances`）为输入，这是组合化的基础。

```python
@dataclass
class Goal:
    kind: str                 # place / toggle / hold / state / composite
    target_object_id: str | None
    receptacle_id: str | None
    state: str | None         # heated / cooled / cleaned
    children: list["Goal"]    # composite: sequence / and / or

@dataclass
class TaskSpec:
    task_id: str
    schema_version: str
    backend: str              # "ai2thor" | "robothor" | "molmospaces"
    scene: SceneSpec
    goals: list[Goal]         # 顺序即阶段顺序
    budget: Budget            # max_steps / max_time / max_api_cost
    seed: int
    source: str | None        # 原始 ALFRED traj 路径等
```

完成判定演进为：

```python
def evaluate_goal(metadata, task_state, goal: Goal) -> GoalStatus:
    # 叶子 goal 用实例级判定；composite 按 children 聚合
    ...
```

长程任务至少需要：
- **阶段状态**：每个子 goal 独立结算，已完成阶段不再重复检查；
- **中间状态持久化**：场景不重置，记忆/物体状态跨阶段延续；
- **重试粒度**：失败时只回退到失败阶段，而不是重跑整个长程任务；
- **组合规则**：sequence（默认）、parallel、condition（下一阶段依赖前一阶段结果）。

### 4.3 运行产物 schema 版本化

在 episode JSON 顶层增加：

```jsonc
{
  "schema_version": "1.0",
  "run_manifest": {
    "task_spec_id": "...",
    "backend": "ai2thor",
    "agent": {"planner": "...", "executor": "...", "memory": "semantic"},
    "seed": 42,
    "started_at": "...", "finished_at": "..."
  }
}
```

目的：
- leaderboard 只认 `schema_version` 与 `run_manifest`，保证跨版本可比；
- 历史产物（未带 schema 版本）按旧格式兼容读取，新产物必须带版本；
- `goal_instances` 已经是前向兼容字段：缺失时回退类型级判定。

### 4.4 评测与提交

```python
@dataclass
class EvalRun:
    run_id: str
    task_spec_id: str
    agent_id: str
    budget: Budget
    status: str
    artifacts: list[str]      # episode JSON / logs / images 的引用

@dataclass
class EvalReport:
    run_id: str
    metrics: dict[str, float]
    schema_version: str
```

提交流程（闭环）：

```
提交 agent/配置
    → 锁定 task suite + 模拟器版本 + seed + budget
    → 容器化运行（可选）
    → 产出标准化 run 产物
    → 计算指标 → 写入结果库
    → 与历史提交对比（leaderboard / 回归告警）
    → 失败样本进入数据集 → 微调 → 新提交
```

## 5. 指标草案

第一版 leaderboard 建议至少包含：

| 指标 | 定义 | 说明 |
|------|------|------|
| Success Rate | `task_complete / total` | 主分支完成任务的比例 |
| Step Efficiency | 成功任务的平均步数（或 `1/steps`） | 越短越好 |
| Stage Success Rate（长程） | 每阶段完成率 | 定位长程任务瓶颈 |
| Recovery Rate | 失败后成功恢复的比例 | 衡量恢复机制 |
| Verified Fork Rate | `counterfactual_verified` fork 占比 | 分支/反事实质量 |
| Failure Diversity | 失败类别覆盖数 | 数据采集价值 |
| API Cost | 每任务平均 API 调用/成本 | 预算约束下的公平性 |
| Generalization Gap | sim 成绩 − real 成绩（RoboTHOR） | sim2real 差距 |
| Safety Violations | 危险/非法动作次数 | 后续平台必备 |

指标计算必须绑定 `schema_version` 与 `run_manifest`；任何指标定义变更都应升版本，避免 leaderboard 前后不可比。

## 6. 里程碑

### M0 — 契约地基（部分元素已有）
- [x] `EnvController` 作为单一模拟器接触面（需进一步抽象为协议）
- [x] 实例级完成判定（`goal_instances` + `task_conditions`）
- [x] episode schema 增加 `schema_version`（`run_manifest` 已占位，待填充）
- [x] `SimulatorBackend` 协议 + `Ai2ThorBackend` 适配器（归一化 `metadata` 视图待做）
- [x] `TaskSpec` / `Goal` 数据结构与 `evaluate_goal` 接口（已接入 runner）

### M1 — 长程任务
- [x] 子任务组合（sequence/and/or）与阶段结算（已接入 runner）
- [x] 跨阶段状态持久化（`stage_index` / `stage_progress` 已持久化，场景与记忆沿用现有机制）
- [x] 阶段步数预算（`max_steps_per_stage` / `max_steps_total`，超预算即终止，不做特殊重试）
- [ ] 长程任务专用指标（stage success、stage 步数分布）
- [ ] 在现有 ALFRED 任务上构造首批长程任务（如 examine → pick_and_place → heat/place）

### M2 — 多模拟器
- [ ] RoboTHOR adapter（sim2real 验证入口）
- [ ] MolmoSpaces adapter（大规模场景/操作）
- [ ] 观测与动作空间差异的适配层（含 RGB-D、本体感受等）
- [ ] 同一 task suite 跨后端的兼容性测试

### M3 — 闭环测评
- [ ] Eval Runner：task suite registry、seed/budget 锁定、容器化运行
- [ ] 结果库与 artifact store
- [ ] 指标计算模块（版本化）
- [ ] 失败样本自动流入数据集/FT 流程

### M4 — Leaderboard
- [ ] 提交接口（CLI/API）与结果对比页
- [ ] 回归告警（同一 agent 新版本跑分下降即拦截）
- [ ] 多 agent / 多提交的历史趋势
- [ ] 安全与成本门禁

## 7. 风险与开放问题

1. **完成判定的泛化**：当前实例级判定依赖 ALFRED 金轨迹的 `low_actions`。长程任务或非 ALFRED 来源没有金轨迹时，需要显式 Goal 描述，否则容易退回类型级误判（本次修掉的「任意 Desk」问题会在组合任务中放大）。
2. **模拟器差异**：RoboTHOR 的 sim2real 与 MolmoSpaces 的大规模操作各自需要新的 obs/action 契约；归一化 `metadata` 是第一个要做的兼容层，越晚做迁移成本越高。
3. **可复现性**：场景/资产/模拟器版本必须锁版本；否则 leaderboard 分数不可比。
4. **成本控制**：VLM 长程任务 API 成本随阶段数线性增长，budget 必须成为一等公民。
5. **数据闭环质量**：失败样本不能直接进数据集，需要去重、标注和「是否可恢复」过滤，否则会放大模型偏见。
6. **公平对比**：不同 agent 应在同一 task suite、同一 seed/budget、同一评测器版本下比较。

## 8. 相关文件索引

| 关注点 | 文件 |
|--------|------|
| 模拟器封装 | `src/env_controller.py` |
| ALFRED 解析 / 实例级目标 | `src/alfred_parser.py` |
| 完成判定 | `src/task_conditions.py` |
| 单任务运行 / 分支 / 恢复 | `src/branch_runner.py` |
| 批量调度 | `src/scheduler.py` |
| 记忆 | `src/semantic_memory.py`、`src/geometric_memory.py`、`src/egocentric_memory.py` |
| episode 管理 | `src/episode_manager.py` |
| 失败注入 | `src/env_injector.py` |
| FT 数据生成 | `scripts/generate_ft_data.py` |
| 采集分析 | `scripts/analyze_collection_snapshot.py` |
| 消融 | `scripts/run_timed_ablation.py`、`scripts/analyze_ablation.py` |
| 运行入口 | `scripts/run_pipeline.py` |

---

本文是演进蓝图，不是现状说明书。每个里程碑落地前，都应先更新对应接口与 schema 再实现，避免平台化过程中把模拟器或任务类型的特殊性写进核心逻辑。
