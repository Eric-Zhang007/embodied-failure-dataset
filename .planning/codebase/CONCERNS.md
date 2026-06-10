# CONCERNS.md — Issues, Risks & Technical Debt

## Risk Assessment Summary

| # | Concern | Severity | Impact | Likelihood | Effort to Fix |
|---|---------|----------|--------|------------|---------------|
| 1 | No automated tests | High | Changes break silently | Certain (every change) | Medium |
| 2 | API key in CLAUDE.md | High | Credential leak | Low (private repo) | Low |
| 3 | Fork mechanism untested | High | Core feature may not work | High | High |
| 4 | PickupObject persistent failure | Medium | Blocks task completion | Medium | Unknown |
| 5 | EB spatial reasoning instability | Medium | Wrong navigation decisions | High | High |
| 6 | Serial scheduler despite max_parallel param | Medium | Slow batch processing | Certain | Medium |
| 7 | Stage 0 never done | Medium | Failure types may be poor | Medium | High |
| 8 | Large monolithic function (BranchRunner.run) | Medium | Hard to modify/debug | Certain | High |
| 9 | No type checking | Low | Runtime type errors | Medium | Low |
| 10 | Module-level mutable state | Low | Test isolation impossible | Certain (when tests exist) | Low |

## 1. No Automated Tests (HIGH)

**Zero test files.** The entire verification process is manual E2E runs. Every code change risks breaking task completion logic, prompt formatting, or environment interaction without detection.

**Recommendation**: Add pytest, start with `task_conditions.py` tests (pure logic, no AI2-THOR dependency), then add prompt template validation tests.

## 2. API Key in CLAUDE.md (HIGH)

The SiliconFlow API key (`sk-umtqhy...`) is hardcoded in `CLAUDE.md` which is checked into git. If the repo becomes public or shared, the key is exposed.

**Recommendation**: Move to `.env` file (gitignored), reference via environment variable.

## 3. Fork Mechanism Untested (HIGH)

The entire fork/counterfactual branch system exists in code but:
- `enable_fork=False` is the default in all scripts
- No E2E test has ever run with forks enabled
- `fork_manager.py` reasoning rewrite has never been validated against real Oracle output
- Fork branch execution (replay + alternative action + continue) has unknown bugs

**Recommendation**: Run dedicated fork E2E tests before relying on fork for data generation.

## 4. PickupObject Persistent Failure (Medium)

Metadata reports objects as `visible: true` but `PickupObject` returns "not found." Possible causes:
- Distance > 0.5m (interaction range) despite appearing close
- Object occluded by transparent or thin geometry
- AI2-THOR internal state mismatch (object marked visible but not interactable)

**Impact**: Blocks task completion for any pick-and-place task.

## 5. EB Spatial Reasoning Instability (Medium)

Even with multi-image LookAround input and direction labels, the 32B VLM sometimes:
- Misidentifies which direction an object is in
- Chooses wrong movement direction
- Fails to navigate around obstacles despite clear visual feedback

**Impact**: Increases failure rate, reduces valid recovery data.

## 6. Serial Scheduler (Medium)

`SchedulerConfig.max_parallel` exists but the scheduler always processes branches one at a time via `deque.popleft()`. Fork branches are also processed serially after all main branches complete.

**Impact**: Slow batch processing. With ~60s per step and 20-step episodes, 50 episodes = ~16 hours.

## 7. Stage 0 Never Done (Medium)

`failure_type_library.json` contains 8 hand-crafted failure types. The original plan ("Stage 0") was to have Oracle read AI2-THOR documentation and auto-generate a comprehensive failure type library. The hand-crafted types may miss important failure modes.

## 8. Monolithic BranchRunner.run() (Medium)

At 550+ lines, `BranchRunner.run()` handles Phase 1-4 loop, LookAround, Done detection, injection, error recovery, fork creation, and step recording. Adding features or fixing bugs requires navigating the entire function.

**Recommendation**: Extract phases into separate methods or a state machine.

## 9. No Type Checking (Low)

Type hints are used inconsistently. No mypy/pyright configuration. Some functions have no type annotations at all.

## 10. Module-Level Mutable State (Low)

`EnvController._xvfb_proc` is a class variable that holds process state. This prevents running multiple isolated tests in the same process.

## TODO/FIXME/HACK Scan Results

No `TODO`, `FIXME`, `HACK`, `XXX`, or `BUG` comments found in the source code. Issues are tracked in `CLAUDE.md` under "已知待修复问题" (Known Issues to Fix).

## Known Issues from CLAUDE.md

1. **PickupObject 持续失败** — metadata visible but PickupObject "not found"
2. **EB 空间推理不稳定** — model may misread direction/distance even with multi-image
3. **Fork 机制未端到端测试** — code present, default disabled
4. **Phase 2 guard** — `cascade_level <= 1` may still be too strict
5. **max_parallel** — parameter exists, always serial
6. **Stage 0 未做** — failure_type_library.json hand-written, should be auto-generated

## Security Concerns

| Issue | Location | Risk |
|-------|----------|------|
| API key in CLAUDE.md | `CLAUDE.md:9` | Key exposure if repo shared |
| No input validation on API responses | `vlm_client.py` | Malformed VLM output could inject data |
| No rate limiting on local logs | `vlm_client.py` | `api_calls.jsonl` grows unbounded |

## Reliability Concerns

| Issue | Location | Impact |
|-------|----------|--------|
| VLM timeout escalation may not be enough | `vlm_client.py:351` | 32B model can take 60-90s for multi-image; 180s max may still timeout |
| JSON parse failure after 2 retries = crash | `vlm_client.py:249` | One malformed VLM response kills the episode |
| Replay mismatch = crash | `branch_runner.py:813` | Any environment state drift during resume kills the branch |
| No checkpoint/resume for scheduler | `scheduler.py` | If process dies, all progress lost (episode JSONs survive but queue state doesn't) |
