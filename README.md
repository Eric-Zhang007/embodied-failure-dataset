# Embodied Failure Dataset

具身智能体级联失败 + 反事实推理数据集。在 AI2-THOR 5.0.0 中运行 ALFRED 7 种任务，通过 Oracle Agent 注入陷阱制造失败场景，记录完整诊断、恢复尝试和反事实标注。

## 快速开始

### 环境要求

- **WSL2 Ubuntu 24.04**（AI2-THOR 无 Windows build）
- Python 3.10（通过 uv 管理）
- GitHub 账号 + 仓库访问权限

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/Eric-Zhang007/embodied-failure-dataset.git
cd embodied-failure-dataset

# 2. 安装 uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. 同步依赖（自动创建 venv + 安装 AI2-THOR 等）
uv sync

# 4. (可选) 安装 GSD — Claude Code 的项目管理插件
# GSD 提供 /gsd-spike, /gsd-pause-work, /gsd-resume-work 等命令
npx -y @opengsd/gsd-core@latest --claude --local
```

### 下载 ALFRED 数据

```bash
# 从 ALFRED 官网下载 json_2.1.0 并解压到 data/
# https://github.com/askforalfred/alfred/data/
```

### 运行

```bash
# WSL2 中执行
export PATH="$HOME/.local/bin:$PATH"

# 单任务测试
uv run python scripts/e2e_test.py \
  --api-key <your-api-key> \
  --api-base-url https://api.fullcupai.com/v1 \
  --task pick_and_place_simple --random --no-traps

# 消融实验（关闭特定 spike）
uv run python scripts/e2e_test.py \
  --api-key <your-api-key> \
  --api-base-url https://api.fullcupai.com/v1 \
  --task pick_and_place_simple --random --no-traps \
  --ablation 003 006

# 分析消融结果
uv run python scripts/analyze_ablation.py output_e2e_*/
```

---

## 给 Partner 的操作指南

### 同步最新代码

```bash
git pull origin main
uv sync  # 依赖有变化时重新同步
```

### 理解项目结构

```
src/
├── branch_runner.py      # 主循环：Phase 1-4 编排
├── eb_agent.py           # Planner：意图规划 + 审核 + 诊断
├── executor.py           # Executor：意图 → 动作序列
├── oracle_agent.py       # Oracle：注入决策 + 失败评估
├── vlm_client.py         # VLM API 调用（含 CoT 保存 + 5xx retry）
├── egocentric_memory.py  # 空间记忆 + StuckTracker + SearchTrail
├── context_builder.py    # 上下文渲染（含 Intent Tree）
├── task_conditions.py    # 任务完成检查 + criteria 文本
├── critic_guard.py       # Spike 002b：执行前验证
├── curiosity_scorer.py   # Spike 003：好奇心评分
└── ...

.planning/
├── spikes/               # 实验文档（每个 spike 一个 README）
│   └── MANIFEST.md       # ← 先读这个：实验全景
├── reports/              # 里程碑总结
├── .continue-here.md     # 断点续跑
└── PROJECT.md            # 项目定位
```

### 查看 API 调用日志（含 CoT 推理链）

```bash
cd output_e2e_*/
python3 -c "
import json
with open('api_calls.jsonl') as f:
    for line in f:
        r = json.loads(line)
        cot = len(r.get('reasoning_content', ''))
        print(f'status={r[\"status\"]} cot={cot}chars model={r[\"model\"]}')
"
```

### 使用 GSD 工作流

如果安装了 GSD (`npx -y @opengsd/gsd-core@latest --claude --local`)：

```bash
# Claude Code 中可用的命令：
/gsd-spike              # 创建新实验 spike
/gsd-spike --wrap-up    # 打包 spike 发现
/gsd-pause-work         # 保存当前状态
/gsd-resume-work        # 恢复上次状态
/gsd-plan-phase         # 创建正式开发计划
/gsd-milestone-summary  # 生成里程碑总结
```

### 审查 PR

```bash
# 在 PowerShell 中（gh 认证后）
gh pr view <number> --json title,body,files,additions,deletions
gh pr diff <number>
```

### 当前导航策略（6 层纵深防御）

读 `.planning/spikes/MANIFEST.md` 了解全貌：

| 层 | Spike | 机制 | 成本 |
|:--:|-------|------|:--:|
| 1 | 003 curiosity-scoreboard | 3D 主动评分引导探索 | 零 |
| 2 | 001 searched-markers | 实例级标记 + unmark | 零 |
| 2 | 006 search-trail-cost | 路径重复惩罚 | 零 |
| 3 | 005 progress-gating | 5 阶段动态提示词 | 零 |
| 4 | 002b critic-guard | 执行前验证 | 零/低 |
| 4 | 004 contrastive-planner | 双 Planner 辩论 | 1 VLM call |
| 5 | 002a intent-dedup | 硬阻断重复 | 零 |

所有 spike 默认开启。通过 `--ablation` CLI 做消融实验。

### 当前阻塞问题

1. **Executor 空 actions 死循环** — 已在目标旁时输出 `actions: []`
2. **Camera pitch 耗竭** — 反复 LookDown 不知道 LookUp
3. **语义先验表稀疏** — `curiosity_scorer.py` 只有 99 条
4. **部分 teleport 位置差** — ALFRED 已知问题

---

## API 配置

- **端点**: `https://api.fullcupai.com/v1`
- **Planner 模型**: gpt-5.5 (reasoning_effort=xhigh)
- **Executor 模型**: gpt-5.5 (reasoning_effort=medium)
- **Oracle 模型**: gpt-5.5 (reasoning_effort=xhigh)
- **gridSize**: 0.125m
- **visibilityDistance**: 100.0
