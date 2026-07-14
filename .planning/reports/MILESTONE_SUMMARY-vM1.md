# Milestone M1 — 系统完善 · Session Summary (2026-07-13/14)

**Generated:** 2026-07-14
**Purpose:** Team onboarding and project review
**Contributors:** Eric Zhang (@Eric-Zhang007), Claude, laoma9604-design (PR #2)

---

## 1. Project Overview

**Embodied Failure Dataset** — 在 AI2-THOR 5.0.0 中运行 ALFRED 7 种任务，通过 Oracle Agent 注入陷阱制造失败场景，记录完整诊断、恢复尝试和反事实标注。目标用户：具身 AI / 机器人学习研究者。

**核心架构**: Planner(32B) → Executor(8B) → Review(32B) → Execute 四阶段循环。Oracle(32B) 在 Phase 2 注入陷阱，Phase 4 评估失败。

**本次 Session 跨越的 Phase:**
- P1 (系统稳定性 + 意图记忆): ✅ 已在前序 session 完成
- P2 (空间记忆语义化): 部分推进 — 语义先验表已建
- P3 (模糊 Intent 分解): 通过 curiosity scoreboard 间接实现
- P4 (系统性探索 + scan→scan 修复): **本次核心突破** — 6 个导航 spike 形成纵深防御

---

## 2. Architecture & Technical Decisions

### 本次 Session 新增

- **AABB 表面距离**: 物体列表中的距离从"到物体中心"改为"到最近碰撞面"。用 `axisAlignedBoundingBox.cornerPoints` 计算。Fridge 距离从 1.1m（中心）→ 0.5m（表面），消除 Executor 的步数误判。
  
- **Intent Tree 历史**: Planner 看到的不是两个独立的 JSONL dump，而是一个统一的按 intent 组织的树状结构（`▼ approach Fridge — ▸ s1: MoveSequence [FAIL] — └─ diagnosed: ...`）。Executor 只看当前 intent 的历史。

- **CoT 推理链保存**: 所有 API 调用的 `reasoning_content` 现在存入 `api_calls.jsonl`。默认开启，缺则回退。

- **StuckTracker** (来自 PR #2): 累计 intent 失败计数——切换 intent 不会重置。中性动作（RotateLeft 等）不重置连续失败计数。

- **Ablation 框架**: 每个 spike (001-006) 有 feature flag，`--ablation 003 006` 可关闭特定 spike 做消融实验。

### 已有架构（前序 session）

- **Planner+Executor 拆分**: Planner 管高层意图，Executor(8B) 做动作序列
- **空间记忆**: 累积所有 `visibleBounds2D=True` 的物体，按 objectId 去重
- **gridSize=0.125m**: 细粒度移动
- **visibleBounds2D 修复**: AI2-THOR 5.0.0 的 bug 手动修正

---

## 3. Bugfixes Delivered (8)

| # | Bug | 文件 | 影响 |
|---|-----|------|------|
| B1 | 5xx/524 API 错误不重试 | `vlm_client.py` | 服务端超时直接杀进程→现在自动重试 |
| B2 | 距离标签误导（中心 vs 表面） | `eb_agent.py`, `executor.py`, `egocentric_memory.py` | Fridge 1.1m→0.5m |
| B3 | CoT 推理链未保存 | `vlm_client.py` | 现在全部 API 调用保存 reasoning_content |
| B4 | Sliced 命名 bug | `task_conditions.py` | "TomatoSliced"→"Tomato"，影响 904/6574 episodes |
| B5 | Intent 历史缺失 Planner 诊断 | `branch_runner.py`, `eb_agent.py` | Planner 不知道自己的 Phase 3 诊断 |
| B6 | 不完整的 criteria 文本 | `task_conditions.py` | 切片前提条件现在显示在 criteria 中 |
| B7 | 移动后空间记忆位置过时 | `branch_runner.py` | 3 个调用点修正 |
| B8 | MoveSequence 的 failed_object_ids 断裂 | `branch_runner.py` | 现在返回已解析的 params |

---

## 4. Navigation Spikes (6) — 纵深防御

```
Layer 1 (Proactive):  003 curiosity-scoreboard  →  3D 评分引导初始探索
Layer 2 (Memory):     001 searched-markers      →  标记已搜索 + 遮挡感知 unmark
                      006 search-trail-cost     →  路径重复惩罚
Layer 3 (Prompt):     005 progress-gating        →  5 阶段动态提示词（零 API 成本）
Layer 4 (Validation): 002b critic-guard          →  执行前多层验证
                      004 contrastive-planner    →  双 Planner 辩论
Layer 5 (Enforcement): 002a intent-dedup         →  硬阻断重复 intent
```

| # | Spike | 类型 | 核心机制 | 成本 |
|:--:|-------|------|----------|:--:|
| 001 | searched-markers | 被动 | 实例级标记 + occlusion unmark + 显式 UNMARK intent | 零 |
| 002a | intent-dedup | 被动 | 渐进升级（2 次警告→3 次阻断）+ 语义规范化 + 任务感知回退 | 零 |
| 002b | critic-guard | 被动 | 3 层规则检查 + 可选 LLM 验证（默认关闭） | 零/低 |
| 003 | curiosity-scoreboard | **主动** | 3D 评分（语义先验 40% + 新奇度 40% + 发现 20%）+ 静态先验表 | 零 |
| 004 | contrastive-planner | 被动 | 双 Planner（exploit vs explore）+ 门控激活（仅连续 3 次同一目标） | 1 VLM call |
| 005 | progress-gating | 主动 | 5 阶段检测 + 阶段特定规则注入 | 零 |
| 006 | search-trail-cost | 被动 | 0.5m 网格追踪 + 软评分（不硬阻断） | 零 |

---

## 5. 审计发现（14 项，已修 8 项）

来自上下文系统审计（agent a86f9）和回归审计（agent a49483）：

| 严重度 | 数量 | 关键发现 |
|--------|:--:|------|
| 🔴 Bug | 4 | 过期内存、断裂的 failed_object_ids、不完整的 criteria、扫描冷却未设置 |
| 🟡 不一致 | 6 | Planner 看不到 failed_object_ids、Phase 3 缺上下文、Executor error 跨 intent 污染、Done 绕过 intent_history |
| 🟢 次要 | 4 | 异常丢 episode 数据、死循环阈值太松、completed 语义粗糙、记忆方向过时 |

回归审计发现 B5 的诊断提取实际从未生效——intent dict 没有 eb_diagnosis 字段。已修复。

---

## 6. 已知待解决问题

| # | 问题 | 状态 |
|---|------|------|
| 1 | Executor 空 actions 死循环（已到目标 → 输出 `[]` → Phase 3 → 又空） | 未修 |
| 2 | Camera pitch 耗竭（反复 LookDown 到 -60° 后不知道 LookUp） | 未修 |
| 3 | 语义先验表弱（HandTowel 评分低于 Floor，99 条覆盖不全） | 待扩充 |
| 4 | intent-dedup 滑动窗口被 interleaved intent 绕过 | StuckTracker 部分修复 |
| 5 | FloorPlan22 等场景的 teleport 位置把 agent 困在角落 | ALFRED 已知问题 |
| 6 | 无成功完成的任务——所有 spike 无法量化评估 | 需持续 E2E 测试 |

---

## 7. 文件变更统计

| 文件 | 变更 |
|------|------|
| `src/branch_runner.py` | +3,465 / -2,317 (最重) |
| `src/eb_agent.py` | +2,518 / -2,333 |
| `src/egocentric_memory.py` | +463 / -12 (含 StuckTracker + SearchTrail) |
| `src/context_builder.py` | +214 (intent tree 渲染) |
| `src/vlm_client.py` | +54 (5xx retry + CoT) |
| `src/executor.py` | +36 (AABB + trail) |
| `src/task_conditions.py` | +27 (Sliced + criteria) |
| `src/critic_guard.py` | **NEW** 432 行 |
| `src/curiosity_scorer.py` | **NEW** 549 行 |
| `scripts/analyze_ablation.py` | **NEW** 消融分析脚本 |
| `.planning/spikes/*/` | **NEW** 7 个 README + MANIFEST |

**总计**: 13 文件, +5,796 / -2,317 行 (~3,500 净增)

---

## 8. Getting Started

```bash
# 进入 WSL2
wsl
cd ~/embodied-failure-dataset
export PATH="$HOME/.local/bin:$PATH"

# 单任务测试
uv run python scripts/e2e_test.py \
  --api-key sk-f26... \
  --api-base-url https://api.fullcupai.com/v1 \
  --task pick_and_place_simple --random --no-traps

# 消融实验（关闭 003 curiosity + 006 trail）
uv run python scripts/e2e_test.py \
  --api-key sk-f26... \
  --api-base-url https://api.fullcupai.com/v1 \
  --task pick_heat_then_place_in_recep --random --no-traps \
  --ablation 003 006

# 分析消融结果
uv run python scripts/analyze_ablation.py output_e2e_*/

# 查看 API 调用日志（含 CoT）
python3 -c "
import json
with open('output_e2e_*/api_calls.jsonl') as f:
    for line in f:
        r = json.loads(line)
        print(f'status={r[\"status\"]} cot={len(r.get(\"reasoning_content\",\"\"))}chars')
"
```

**关键目录**: `src/` (核心代码), `.planning/spikes/` (实验文档), `data/json_2.1.0/` (ALFRED 轨迹)

---

## Stats

- **Timeline**: 2026-06 → 2026-07-14
- **Session commits**: 2 (+ 5 prior)
- **Session 变更**: 13 files, +5,796 / -2,317
- **新文件**: 3 (critic_guard.py, curiosity_scorer.py, analyze_ablation.py)
- **Contributors**: Eric Zhang, Claude, laoma9604-design
