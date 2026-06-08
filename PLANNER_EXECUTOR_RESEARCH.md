# Planner / Executor 分离方案研究

## 1. 现有工作分析

### 1.1 CoT-VLA: Visual Chain-of-Thought for VLA Models
**CVPR 2025, arxiv 2503.22020**

核心思想：VLA 不直接输出 action，而是先**自回归地预测未来的图像帧**作为 visual subgoal，再生成到达该 subgoal 的短动作序列。

架构：
```
当前帧 → VLA(Planner) → 预测未来帧(subgoal) → VLA(Executor) → 动作序列(5-10步)
                                                                ↓
                                                          执行后,新帧反馈给 Planner
```

关键设计：
- Planner 输出的是**图像**（visual goal），不是文字。这对 VLM 特别自然——VLM 更擅长理解图像而非生成精确的空间文本。
- Executor 拿到 subgoal 图像 + 当前帧，生成到达 subgoal 的动作序列。
- 动作序列很短（5-10 步），失败后 Planner 从当前帧重新预测 subgoal。

**对我们的启示**：如果我们不想预测图像，可以把 visual subgoal 替换为**带方向标注的 object-centric goal**：
```
Planner 输出: "AlarmClock in hand, standing in front of Desk at <0.5m"
Executor 输入: 当前帧 + 上述 goal → 输出动作序列
```

### 1.2 Inner Monologue (Google Robotics, 2022)
**arxiv 2207.05608**

核心思想：LLM 做 planning，但**每次动作后把环境反馈注入 prompt**，形成闭环。

架构：
```
LLM(Planner) → 高层动作序列: [pick(apple), place(table)]
                    ↓
           Low-Level Skills 依次执行
                    ↓
        每次执行后, 环境反馈注入 LLM prompt:
        "pick(apple) failed: apple not in view. Current objects: [bowl, cup, spoon]"
                    ↓
        LLM 根据反馈修改计划: [open(fridge), pick(apple), place(table)]
```

关键设计：
- Planner 的 prompt 里包含：任务描述 + 当前计划 + 最近N步的执行结果（成功/失败 + 当前可见物体）
- 失败时 Planner 自动重规划
- 不区分 Planner/Executor 的模型——同一个 LLM 做两层推理

**对我们的启示**：这是最成熟的 closed-loop planning 范式。我们可以让 Planner 复用现有的 Phase 1 system prompt + memory，Executor 用简化的 action-only prompt。

### 1.3 SayCan (Google, 2022)
**arxiv 2204.01691**

核心思想：LLM 计划 + affordance 接地。LLM 提出候选技能，一个 learned affordance function 评估"当前状态下这个技能是否可执行"。

这是我们不需要的——因为我们的动作空间是确定性的（MoveAhead 总是可以尝试），不需要 learned affordance。但"候选技能排序"的思路有参考价值。

### 1.4 LLM-Planner (2023)
**arxiv 2305.15808**

核心思想：LLM 做 high-level planning，low-level 用预定义的 skills。重点是**grounded re-planning**——当 skill 执行失败时，把当前看到的物体列表反馈给 LLM，让它重新规划。

架构：
```
LLM → plan: [find(apple), pick(apple), find(table), place(table)]
  ↓
每个 skill 展开为 navigation + manipulation 原语
  ↓
失败 → 当前可见物体列表注入 LLM → 重规划
```

**对我们的启示**：re-planning 的 prompt 格式很简单——就是把当前可见物体列表加进去。不需要完整历史。

### 1.5 Code as Policies (Google, 2023)
**arxiv 2209.07431**

核心思想：LLM 写 Python 代码来定义 robot policy。代码里调用 parameterized motion primitives。

这不适合我们——我们不需要代码生成，也不需要参数化 motion。但分层思想一致：高层用语言推理，低层用程序化执行。

---

## 2. 推荐架构

### 2.1 总体设计

```
┌─────────────────────────────────────────────────┐
│ Planner (当前 Phase 1 升级)                       │
│ 输入: task goal + memory + 当前帧                  │
│ 输出: intent 序列 (3-5 个)                        │
│ 例: [locate AlarmClock, pick AlarmClock,          │
│      locate Desk, place on Desk, Done]            │
│ 每个 intent 格式: {intent, target_object, goal_state}│
└────────────────────┬────────────────────────────┘
                     │ 当前 intent
                     ▼
┌─────────────────────────────────────────────────┐
│ Executor (新增)                                   │
│ 输入: 当前 intent + 当前帧 + visible objects       │
│ 输出: 动作序列 (1-5 步) 或 FAILED                 │
│ 执行后反馈: {status: done/failed, reason,          │
│              final_frame_visible_objects}          │
└────────────────────┬────────────────────────────┘
                     │ 反馈
                     ▼
              Planner 决定: 继续下一个 intent
                          或重规划当前 intent
                          或放弃
```

### 2.2 Planner 的 Prompt 结构

```
System: (同 Phase 1, 删除具体动作列表, 加 intent 格式)

User:
  Task goal: ...
  SPATIAL MEMORY: ...
  TASK COMPLETION CRITERIA: ...
  Current progress: [picked AlarmClock, need Desk]

  Previous intent result: "locate Desk" SUCCESS — Desk now visible ahead-right 1.2m

  Output an intent:
  {
    "intent": "<locate X | pick X | place X on Y | toggle X | open X | clean X | Done>",
    "target": "<objectType>",
    "reasoning": "<1 sentence>"
  }

  Available intents: locate, pick, place, toggle, open, close, clean, heat, cool, slice, Done
```

### 2.3 Executor 的 Prompt 结构

```
System: You are an action executor. Your ONLY job: given a current intent and first-person view, output 1-5 concrete actions to achieve it.

RULES:
- Interaction range 0.5m. Check distance before interaction actions.
- When MoveAhead blocked: Rotate, do NOT retry.
- Output 1-5 actions max, then report status.

AVAILABLE ACTIONS: (同当前 Phase 1 动作列表, 紧凑版)

OUTPUT:
{
  "actions": [
    {"action": "MoveAhead", "params": {}},
    {"action": "PickupObject", "params": {"objectType": "AlarmClock"}}
  ],
  "status": "done" | "partial" | "failed",
  "status_reason": "<1 sentence>"
}

- "done": intent fully achieved, stop
- "partial": made progress but need more actions, will be called again
- "failed": intent cannot be achieved from current position
```

### 2.4 交互协议

```
Planner 输出 intent "pick AlarmClock"
  → Executor 输出 3 步: [MoveAhead, MoveAhead, PickupObject]
  → 执行成功, status: "done"
  → Planner 输出下一个 intent "locate Desk"
  
Planner 输出 intent "locate Desk"
  → Executor 输出 3 步: [RotateLeft, MoveAhead, MoveAhead]
  → 执行成功, status: "done"
  → Planner 输出 "place AlarmClock on Desk"

Planner 输出 intent "locate AlarmClock"
  → Executor 输出 3 步: [MoveAhead × 3]
  → 第2步 MoveAhead FAILED (blocked by chair)
  → Executor 停止, status: "failed", reason: "blocked by a chair ahead"
  → Planner 收到失败反馈, 重新输出 intent: "navigate around chair"
  → Executor 输出: [MoveLeft, MoveAhead]
  → 成功, status: "done"
  → Planner 继续 "locate AlarmClock"
```

### 2.5 关键设计决策

**Q: Planner 要不要看图？**
A: 要。参考 Inner Monologue——Planner 需要看到当前帧来判断"intent 是否已完成"和"重规划是否必要"。但不需高清图，可以降低分辨率。

**Q: Executor 要不要知道 task goal？**
A: 不需要。Executor 只需要知道当前 intent。这让 Executor 的 prompt 极其紧凑（~1000 chars system + 当前帧 + 一行 intent）。

**Q: 多少个 intent 算一个 plan？**
A: 一次输出 1 个 intent（不是序列）。因为执行过程中随时可能失败，输出整个序列没有意义。Planner 每次被调用时输出**下一个** intent。

**Q: Executor 的 action chunk 多长？**
A: 1-5 步，参考 CoT-VLA 的做法。太短（1步）= 退化回当前系统；太长（10+）= Executor 也需要复杂推理。3-5 步是 sweet spot。

---

## 3. 实现路径

### Phase 1: 最小改动验证
1. 新增 `src/executor.py`：`ExecutorAgent` 类，`executor_system` prompt（紧凑版动作列表 + intent 格式），`execute_intent()` 方法
2. 修改 `src/eb_agent.py`：Planner 输出 intent 而非 action（改 output schema）
3. 修改 `src/branch_runner.py`：Planner → Executor 循环替代当前 Phase 1 循环
4. Executor 失败时触发 Planner 重规划（替代当前 Phase 3）

### Phase 2: 优化
1. 给 Planner 降分辨率图像（减少 token）
2. Executor 的 action chunk 长度自适应
3. Intent 复用检测（同一 intent 失败 3 次 → 放弃）

---

## 4. 参考文献

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| CoT-VLA (arxiv 2503.22020) | 2025 | Visual subgoal prediction, VLA 用图像而非文本做规划 |
| Inner Monologue (arxiv 2207.05608) | 2022 | Closed-loop LLM planning, 环境反馈注入 prompt |
| SayCan (arxiv 2204.01691) | 2022 | LLM plan + affordance grounding |
| LLM-Planner (arxiv 2305.15808) | 2023 | Grounded re-planning with visual feedback |
| Code as Policies (arxiv 2209.07431) | 2023 | LLM writes hierarchical robot code |
