# PLAN.md — Phase 1: LookAround 泄漏修复

**日期**: 2026-06-10
**目标**: 修复 Executor 输出的 LookAround 动作被直接传给 AI2-THOR 导致崩溃的 bug

---

## Bug 分析

### 触发链路

```
Executor 输出 actions 含 LookAround
  → Planner review approved
  → proposed_action = "MoveSequence"
  → _execute_move_sequence(steps)
  → LookAround ∉ _MOVEMENT_ACTIONS && LookAround ∉ {"Done"}
  → LookAround ∈ _VALID_ACTIONS，但不该发给 AI2-THOR
  → resolve_object_ids(LookAround, ...) 误处理
  → env.step("LookAround") → ValueError
```

### 根因

`_execute_move_sequence` (line 877-878) 的 `_MOVEMENT_ACTIONS` 不含 LookAround，`_VALID_ACTIONS` (line 27) 含 LookAround。LookAround 在 `_VALID_ACTIONS` 里是合法的 Phase 1 单步动作，但在 MoveSequence 上下文中不该出现——它属于 branch_runner 层 meta-action（四向扫描+多图决策），不应下发给 AI2-THOR。

Executor prompt 没有明确禁止 LookAround，8B 模型在某些场景下（找目标、被阻挡后）会输出 LookAround 作为探索策略。

### 修复范围

两个文件，每个一行改动 + prompt 加一句话。

---

## Task 1: _execute_move_sequence 过滤 LookAround

**文件**: `src/branch_runner.py`

在 `_execute_move_sequence` 的步骤循环中，遇到 LookAround 时：
- 若前面已有成功执行的步骤 → 截断序列，返回 partial success（已执行部分保留）
- 若 LookAround 是第一步且什么都没有执行 → 返回错误，明确提示 "LookAround is not allowed in MoveSequence"

实现：在 `_MOVEMENT_ACTIONS` 集合之后增加一个 `_META_ACTIONS = {"Done", "LookAround"}` 集合，在步骤循环开头检查。遇到 meta-action 时根据是否有已执行步骤分别处理。

```
_MOVEMENT_ACTIONS = {...}
_META_ACTIONS = {"Done", "LookAround"}  # handled by branch_runner, not AI2-THOR

for i, step in enumerate(steps):
    action = step.get("action", "")
    
    # NEW: block meta-actions at MoveSequence level
    if action in _META_ACTIONS:
        if executed:
            # Truncate: keep what was executed, ignore the rest
            desc = " → ".join(executed)
            msg = f"MoveSequence completed (before {action}): {desc}. (Meta-action {action} ignored — handled separately.)"
            return ({"success": True, "all_succeeded": True, ...}, msg)
        else:
            msg = f"MoveSequence: {action} is not allowed as a MoveSequence step."
            return ({"success": False, "error": msg, ...}, msg)
    
    if action not in _MOVEMENT_ACTIONS:
        ...
```

同时移除原来的 `action == "Done"` 特殊处理 (line 917-919)，因为 Done 现在统一走 `_META_ACTIONS` 逻辑。

## Task 2: Executor prompt 明确禁止 LookAround

**文件**: `src/executor.py`

在 `EXECUTOR_SYSTEM` 的 RULES 部分加一行：

```
- Do NOT output LookAround or Done in the actions array. These are handled by the system separately.
```

放在第一个 rule 下方或 ACTIONS 列表之前。

## Task 3: 验证

单条 E2E 验证 bug 不再复现，然后重跑 --all。

```bash
uv run python scripts/e2e_test.py --api-key sk-xxx --task pick_and_place_simple --random --no-traps
```

检查点：
1. 不再出现 `ValueError: Invalid action: LookAround`
2. 若 Executor 输出 LookAround，MoveSequence 返回 truncated 结果（保留前面已执行步骤）
3. 截断后 Phase 1 正常继续下一轮

---

## 不做的事

- 不修改 Executor prompt 中的其他内容（范围最小化）
- 不在 Planner review 中拦截 LookAround（防御深度没必要，源头修复即可）
- 不修改 `_VALID_ACTIONS`（LookAround 在单步 Phase 1 中仍需合法）

## 预计改动量

| 文件 | 改动 |
|------|------|
| `branch_runner.py:_execute_move_sequence` | ~15 行（加 _META_ACTIONS 检查 + 移除旧 Done 特殊处理） |
| `executor.py:EXECUTOR_SYSTEM` | 1 行（加禁止说明） |
