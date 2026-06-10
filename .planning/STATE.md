# STATE.md — Project Memory

**Last updated**: 2026-06-10
**Current milestone**: M1 — 系统完善
**Current phase**: P1 (PickupObject 诊断) 和 P2 (测试) 准备开工

---

## E2E 现状（2026-06-10 分析结果）

- pick_and_place_simple: 近期 12 跑，3 成功 (25%)，但成功 case 都是 agent 碰巧站在目标旁边
- look_at_obj_in_light (examine): 3 跑，0 成功 — 捡起物体后找不到灯，rotate 死循环
- pick_and_place_with_movable_recep: 3 跑，0 成功 — 三阶段任务，最高 47 次失败
- pick_two_obj_and_place: 1 跑，0 成功
- heat/cool/clean: 从未 E2E 测试
- **所有任务类型的共同 blocker**: PickupObject 失败（metadata visible 但 AI2-THOR "not found"）

## Active Decisions

- P1 (PickupObject) 最优先 — 它阻塞了所有任务类型
- P2 (测试) 与 P1 并行
- 用户在上一轮确认：修系统优先，测试仅关键模块

## Recent Changes

- 2026-06-10: E2E 全量分析完成（37 个 output 目录，19 个有 final_outcome 的近期 run）
- 2026-06-10: REQUIREMENTS.md / ROADMAP.md 基于 E2E 分析重写优先级
- 2026-06-10: `/gsd-map-codebase` 完成，7 份代码库文档
- 2026-06-10: `/gsd-new-project` 初始化

## Known Blockers

- **#1: PickupObject 失败** — metadata.visible 和 AI2-THOR 实际交互范围不一致
- gsd-* subagent 类型与 deepseek-v4-pro 不兼容（需用 general-purpose 或 inline）
- Fork 从未端到端验证
- heat/cool/clean 任务类型从未测试

## Key Files

- `CLAUDE.md` — 项目技术文档（含已知问题列表、最近补丁）
- `.planning/codebase/` — 代码库地图（7 份文档）
- `.planning/REQUIREMENTS.md` — M1 需求（6 个需求，按优先级排列）
- `.planning/ROADMAP.md` — 阶段规划（P1-P6）
- `src/branch_runner.py` — 核心循环（944 行，最常修改的文件）
- `src/vlm_client.py` — VLM API 客户端
- `src/env_injector.py` — 注入方法实现

## Next Action

运行 `/gsd:plan-phase 1` 开始 P1（PickupObject 诊断与修复）。
