"""
分支执行器。运行单个分支的 Phase 1-4 交互循环直到终止。
成功步骤写入 episode JSON，失败事件写入独立 failure log。
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import OracleAgent
from src.env_controller import EnvController
from src.env_injector import inject
from src.step_recorder import StepRecorder
from src.episode_manager import EpisodeManager
from src.action_adapter import adapt, resolve_object_ids, _OBJECT_ACTIONS
from src.task_conditions import check_task_complete, check_unrecoverable, detect_dead_loop, get_completion_criteria_text
from src.context_builder import build_branch_history
from src.egocentric_memory import EgocentricMemory

_VALID_ACTIONS = {
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown",
    "PickupObject", "PutObject", "OpenObject", "CloseObject",
    "ToggleObjectOn", "ToggleObjectOff", "SliceObject", "BreakObject", "FillObjectWithLiquid", "EmptyLiquidFromObject", "DropHandObject",
    "Done", "LookAround", "MoveSequence",
}


@dataclass
class BranchConfig:
    episode_id: str
    branch_id: str
    parent_branch_id: Optional[str]
    shared_context_step_ids: list[str] = field(default_factory=list)
    diverges_at_step_id: Optional[str] = None
    fork_config: Optional[dict] = None


@dataclass
class BranchResult:
    branch_id: str
    termination_reason: str
    total_steps: int
    fork_tasks: list[dict]
    fork_source_step_ids: list[str] = field(default_factory=list)


class BranchRunner:
    def __init__(
        self,
        eb_agent: EBAgent,
        oracle_agent: OracleAgent,
        output_dir: str,
        step_limit_multiplier: int = 4,
        enable_fork: bool = True,
        enable_phase2: bool = True,
        executor_agent=None,
    ):
        self.eb_agent = eb_agent
        self.oracle_agent = oracle_agent
        self.executor_agent = executor_agent
        self.output_dir = output_dir
        self.enable_phase2 = enable_phase2
        self.step_limit_multiplier = step_limit_multiplier
        self.enable_fork = enable_fork
        self.recorder = StepRecorder()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(
        self,
        config: BranchConfig,
        env: EnvController,
        ep: EpisodeManager,
        start_step_index: int = 0,
    ) -> BranchResult:
        step_index = start_step_index
        cascade_level = 0
        fork_tasks: list[dict] = []
        fork_source_ids: list[str] = []
        last_error: Optional[str] = None

        failure_log_path = os.path.join(
            self.output_dir, ep.episode_id,
            f"failures_{config.branch_id}.jsonl",
        )
        os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)

        eb_history: list[dict] = build_branch_history(
            ep.data["steps"],
            branch_id=config.branch_id,
            parent_branch_id=config.parent_branch_id,
            shared_step_ids=config.shared_context_step_ids,
        )

        failed_object_ids: set[str] = set()
        memory = EgocentricMemory()
        phase2_injection_count = 0
        phase2_max_injections = 3
        nonexecuted_retry_count = 0
        max_nonexecuted_retries = 3

        # ==========================================================
        # Initial LookAround: 开局强制四向扫描
        # ==========================================================
        if step_index == 0:
            state0 = env.get_state_snapshot()
            image0 = state0["frame"]
            metadata0 = state0["metadata"]
            look_images = []
            for d in ["ahead", "left", "behind", "right"]:
                if d == "ahead":
                    snap = image0
                else:
                    r = env.step("RotateLeft")
                    if not r["success"]:
                        raise RuntimeError(f"Init LookAround RotateLeft failed: {r['error']}")
                    snap = r["frame"]
                look_images.append((d, snap))
            restore_result = env.step("RotateLeft")
            if not restore_result["success"]:
                raise RuntimeError(f"Init LookAround restore RotateLeft failed: {restore_result['error']}")
            look_step = self._build_step_entry(
                ep.episode_id, config.branch_id, 0, None,
                "LookAround", {},
                {"success": True, "error": None, "frame": image0, "metadata": metadata0},
                "initial full-room scan", None,
            )
            img_dir = os.path.join(self.output_dir, ep.episode_id)
            vps = []
            for label, frame in look_images:
                vp = os.path.join(img_dir, f"s0_look_{label}.png")
                StepRecorder.save_frame(frame, vp)
                vps.append({"label": label, "image_path": vp})
            look_step["lookaround_views"] = vps
            self._write_success_step_direct(ep, look_step)
            eb_history.append(look_step)
            tc0 = get_completion_criteria_text(
                ep.data.get("alfred_task_type") or ep.data.get("task_type", ""),
                ep.data.get("pddl_params", {}),
            )
            memory.update(metadata0, metadata0.get("objects", []), "LookAround", True, None, tc0)
            step_index = 1
            eb_phase1 = self.eb_agent.propose_action_lookaround(
                task_goal=ep.data["task_goal"],
                look_images=look_images,
                visible_objects=metadata0.get("objects", []),
                action_history=eb_history,
                last_error=None,
                failed_object_ids=set(),
                inventory_objects=[],
                task_criteria=tc0,
                memory_text=memory.render(),
            )
            proposed_action = eb_phase1["action"]
            proposed_params = eb_phase1.get("params", {})
            eb_reasoning = eb_phase1.get("reasoning", "")

        while True:
            if step_index >= 200:
                return self._make_result(config, "step_hard_limit", step_index, fork_tasks, fork_source_ids, ep)
            # ==========================================================
            # Phase 1: EB Agent 提议动作
            # ==========================================================
            state = env.get_state_snapshot()
            image = state["frame"]
            metadata = state["metadata"]
            task_state = state.get("task_state", {})

            inventory_objects = metadata.get("inventoryObjects") or []
            task_criteria = get_completion_criteria_text(
                ep.data.get("alfred_task_type") or ep.data.get("task_type", ""),
                ep.data.get("pddl_params", {}),
            )

            task_complete, _ = check_task_complete(metadata, ep.data, task_state)
            if task_complete:
                self._write_success_step(
                    ep, config.branch_id, step_index,
                    f"s{step_index - 1}" if step_index > 0 else None,
                    "Done", {},
                    {"success": True, "error": None, "frame": image, "metadata": metadata},
                    eb_reasoning="Task goal achieved (auto-detected).",
                )
                return self._make_result(config, "task_complete", step_index + 1,
                                         fork_tasks, fork_source_ids, ep)

            agent_pose = metadata.get("agent", {})
            # Skip Phase 1 VLM if initial LookAround already decided
            if step_index == 1 and proposed_action:
                pass
            elif self.executor_agent is not None:
                # ── Planner→Executor→Review cycle ──
                hand_status = "empty"
                if inventory_objects:
                    h = inventory_objects[0]
                    hand_status = f"holding {h.get('objectType', '?')}"

                # 1. Planner proposes intent
                planner_intent = self.eb_agent.plan_intent(
                    task_goal=ep.data["task_goal"],
                    image=image,
                    visible_objects=metadata.get("objects", []),
                    action_history=eb_history,
                    last_error=last_error,
                    agent_pos=agent_pose.get("position"),
                    agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                    inventory_objects=inventory_objects,
                    hand_status=hand_status,
                    task_criteria=task_criteria,
                    memory_text=memory.render(),
                )
                intent = planner_intent["intent"]
                intent_target = planner_intent.get("target", "")

                # 2. Executor proposes actions
                exec_result = self.executor_agent.execute_intent(
                    intent=intent,
                    target=intent_target,
                    image=image,
                    visible_objects=metadata.get("objects", []),
                    action_history=eb_history,
                    last_error=last_error,
                    agent_pos=agent_pose.get("position"),
                    agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                    hand_status=hand_status,
                    task_criteria=task_criteria,
                    memory_text=memory.render(),
                    failed_object_ids=failed_object_ids,
                )

                # 3. Planner reviews — if rejected, Planner provides corrected actions
                review = self.eb_agent.review_actions(
                    intent=intent,
                    target=intent_target,
                    proposed_actions=exec_result.get("actions", []),
                    executor_reasoning=exec_result.get("reasoning", ""),
                    image=image,
                    visible_objects=metadata.get("objects", []),
                    action_history=eb_history,
                    last_error=last_error,
                    memory_text=memory.render(),
                    agent_pos=agent_pose.get("position"),
                    agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                )

                if review.get("approved"):
                    proposed_action = "MoveSequence"
                    proposed_params = {"steps": exec_result.get("actions", [])}
                    eb_reasoning = exec_result.get("reasoning", "")
                else:
                    proposed_action = "MoveSequence"
                    proposed_params = {"steps": review.get("corrected_actions", exec_result.get("actions", []))}
                    eb_reasoning = review.get("reason", "")
            else:
                # ── Original single-agent path (no executor) ──
                eb_phase1 = self.eb_agent.propose_action(
                    task_goal=ep.data["task_goal"],
                    image=image,
                    visible_objects=metadata.get("objects", []),
                    action_history=eb_history,
                    last_error=last_error,
                    failed_object_ids=failed_object_ids,
                    agent_pos=agent_pose.get("position"),
                    agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                    inventory_objects=inventory_objects,
                    task_criteria=task_criteria,
                    memory_text=memory.render(),
                )
                proposed_action = eb_phase1["action"]
                proposed_params = eb_phase1.get("params", {})
                eb_reasoning = eb_phase1.get("reasoning", "")

            if proposed_action not in _VALID_ACTIONS:
                error_msg = self._invalid_action_message(proposed_action)
                nonexecuted_retry_count += 1
                self._write_failure_log(failure_log_path, {
                    "step_index": step_index,
                    "branch_id": config.branch_id,
                    "failure_type": "model_invalid_action",
                    "proposed_action": proposed_action,
                    "proposed_params": proposed_params,
                    "eb_reasoning": eb_reasoning,
                    "error_message": error_msg,
                    "retry_attempt": nonexecuted_retry_count,
                })
                if nonexecuted_retry_count >= max_nonexecuted_retries:
                    raise RuntimeError(error_msg)
                last_error = error_msg
                continue

            # Done 检测：仅在成功步之后检查，避免初始空手状态误判
            if proposed_action == "Done":
                task_complete_done, done_reason = check_task_complete(metadata, ep.data, task_state)
                if task_complete_done:
                    self._write_success_step(
                        ep, config.branch_id, step_index,
                        f"s{step_index - 1}" if step_index > 0 else None,
                        "Done", {},
                        {"success": True, "error": None, "frame": image, "metadata": metadata},
                        eb_reasoning=eb_reasoning,
                    )
                    return self._make_result(config, "task_complete", step_index + 1,
                                             fork_tasks, fork_source_ids, ep)
                # Task not actually complete — record as failed step with specific reason
                rejected_msg = f"Done rejected: {done_reason}. Fix the unmet criteria before calling Done again."
                result = {
                    "success": False,
                    "error": rejected_msg,
                    "frame": image,
                    "metadata": metadata,
                }
                step_entry = self._build_step_entry(
                    ep.episode_id, config.branch_id, step_index,
                    f"s{step_index - 1}" if step_index > 0 else None,
                    "Done", {}, result, eb_reasoning,
                )
                step_entry["error_type"] = "done_rejected"
                self._write_success_step_direct(ep, step_entry)
                eb_history.append(step_entry)
                self._write_failure_log(failure_log_path, {
                    "step_index": step_index,
                    "branch_id": config.branch_id,
                    "failure_type": "done_rejected",
                    "proposed_action": "Done",
                    "proposed_params": {},
                    "eb_reasoning": eb_reasoning,
                    "error_message": rejected_msg,
                })
                last_error = rejected_msg
                cascade_level = 0
                step_index += 1
                nonexecuted_retry_count = 0
                continue

            # ==========================================================
            # Phase 2: Oracle 注入决策
            # ==========================================================
            remaining = phase2_max_injections - phase2_injection_count
            injection_decision = None
            injection_attempt_errors: list[dict] = []
            if self.enable_phase2 and cascade_level <= 1 and remaining > 0:
                for injection_attempt in range(3):
                    oracle_phase2 = self.oracle_agent.decide_injection(
                        task_goal=ep.data["task_goal"],
                        image=image,
                        env_state=metadata,
                        proposed_action=proposed_action,
                        proposed_params=proposed_params,
                        eb_reasoning=eb_reasoning,
                        action_history=eb_history,
                        cascade_level=cascade_level,
                        is_fork=(config.parent_branch_id is not None),
                        remaining_injections=remaining,
                        injection_attempt_errors=injection_attempt_errors,
                    )
                    if not (oracle_phase2.get("inject") and oracle_phase2.get("injection")):
                        break

                    inj = oracle_phase2["injection"]
                    inj_result = inject(env.controller, method=inj["method"], **inj["params"])
                    if inj_result["success"]:
                        phase2_injection_count += 1
                        trap = {
                            "trap_id": f"trap_{config.branch_id}_{step_index}",
                            "branch_id": config.branch_id,
                            "created_at_step_id": f"s{step_index}",
                            "created_by": "oracle",
                            "injection": inj,
                            "modification_success": True,
                            "triggered_at_step_id": None,
                            "env_error": None,
                            "status": "active",
                        }
                        ep.add_runtime_trap(trap)
                        injection_decision = {
                            "decided_to_inject": True,
                            "injection": inj,
                            "modification_success": True,
                            "modification_error": None,
                        }
                        break

                    attempt_error = {
                        "method": inj.get("method"),
                        "params": inj.get("params", {}),
                        "error": inj_result.get("error"),
                    }
                    injection_attempt_errors.append(attempt_error)
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "injection_setup_failed",
                        "proposed_action": proposed_action,
                        "proposed_params": proposed_params,
                        "injection": inj,
                        "error_message": inj_result.get("error"),
                        "retry_attempt": injection_attempt + 1,
                    })
                else:
                    raise RuntimeError(
                        "Oracle injection setup failed 3 times for the same EB action. "
                        "See failures log for attempted injections."
                    )
            elif self.enable_phase2 and remaining <= 0:
                injection_decision = {
                    "decided_to_inject": False,
                    "reasoning": "LIMIT REACHED: all 3 successful injections have been used.",
                    "injection": None,
                }

            parent_id = f"s{step_index - 1}" if step_index > 0 else None

            # ==========================================================
            # LookAround: 站原地旋转 4 次，截 4 张图发给 EB 做多图综合分析。
            # 如果 EB 扫完还想再扫，记录为无效决策并回到普通 Phase 1。
            # ==========================================================
            if proposed_action == "LookAround":
                lookaround_reasoning = eb_reasoning
                look_images = []
                look_dirs = ["ahead", "left", "behind", "right"]
                for d in look_dirs:
                    if d == "ahead":
                        snap = image
                    else:
                        rotate_result = env.step("RotateLeft")
                        if not rotate_result["success"]:
                            raise RuntimeError(f"LookAround internal RotateLeft failed: {rotate_result['error']}")
                        snap = rotate_result["frame"]
                    look_images.append((d, snap))
                rotate_result = env.step("RotateLeft")  # back to original facing
                if not rotate_result["success"]:
                    raise RuntimeError(f"LookAround restore RotateLeft failed: {rotate_result['error']}")
                look_result = {
                    "success": True,
                    "error": None,
                    "frame": image,
                    "metadata": metadata,
                }
                look_step = self._build_step_entry(
                    ep.episode_id, config.branch_id, step_index, parent_id,
                    "LookAround", {}, look_result, lookaround_reasoning, injection_decision,
                )
                image_dir = os.path.join(self.output_dir, ep.episode_id)
                view_paths = []
                for label, frame in look_images:
                    view_path = os.path.join(image_dir, f"s{step_index}_look_{label}.png")
                    StepRecorder.save_frame(frame, view_path)
                    view_paths.append({"label": label, "image_path": view_path})
                look_step["lookaround_views"] = view_paths
                self._write_success_step_direct(ep, look_step)
                eb_history.append(look_step)
                parent_id = f"s{step_index}"
                step_index += 1
                nonexecuted_retry_count = 0
                memory.update(metadata, metadata.get("objects", []), "LookAround", True, None, task_criteria)

                eb_phase1 = self.eb_agent.propose_action_lookaround(
                    task_goal=ep.data["task_goal"],
                    look_images=look_images,
                    visible_objects=metadata.get("objects", []),
                    action_history=eb_history,
                    last_error=last_error,
                    failed_object_ids=failed_object_ids,
                    inventory_objects=inventory_objects,
                    task_criteria=task_criteria,
                    memory_text=memory.render(),
                )
                proposed_action = eb_phase1["action"]
                proposed_params = eb_phase1.get("params", {})
                eb_reasoning = eb_phase1.get("reasoning", "")

                if proposed_action == "LookAround":
                    error_msg = (
                        "LookAround already scanned the room in the previous step. "
                        "Choose a concrete action from the scan results instead of scanning again."
                    )
                    nonexecuted_retry_count += 1
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "model_repeated_lookaround",
                        "proposed_action": proposed_action,
                        "proposed_params": proposed_params,
                        "eb_reasoning": eb_reasoning,
                        "error_message": error_msg,
                        "retry_attempt": nonexecuted_retry_count,
                    })
                    if nonexecuted_retry_count >= max_nonexecuted_retries:
                        raise RuntimeError(error_msg)
                    last_error = error_msg
                    continue

                if proposed_action not in _VALID_ACTIONS:
                    error_msg = self._invalid_action_message(proposed_action)
                    nonexecuted_retry_count += 1
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "model_invalid_action",
                        "proposed_action": proposed_action,
                        "proposed_params": proposed_params,
                        "eb_reasoning": eb_reasoning,
                        "error_message": error_msg,
                        "retry_attempt": nonexecuted_retry_count,
                    })
                    if nonexecuted_retry_count >= max_nonexecuted_retries:
                        raise RuntimeError(error_msg)
                    last_error = error_msg
                    continue

            # ==========================================================
            # MoveSequence: execute a chain of movement steps sequentially.
            # Stops on first failure, reports what actually succeeded.
            # ==========================================================
            if proposed_action == "MoveSequence":
                seq_result, seq_msg = self._execute_move_sequence(
                    proposed_params, env, metadata, failure_log_path,
                    config.branch_id, step_index, ep.episode_id,
                    eb_reasoning, injection_decision,
                )
                if seq_result.get("all_succeeded"):
                    # All steps succeeded — record as one successful step
                    cascade_level = 0
                    last_error = None
                    step_entry = self._build_step_entry(
                        ep.episode_id, config.branch_id, step_index, parent_id,
                        "MoveSequence", proposed_params, seq_result, eb_reasoning, injection_decision,
                    )
                    self._write_success_step_direct(ep, step_entry)
                    eb_history.append(step_entry)
                    step_index += 1
                    nonexecuted_retry_count = 0
                    memory.update(metadata, metadata.get("objects", []),
                                  "MoveSequence", True, None, task_criteria)
                    # Restore image/metadata from final state
                    image = seq_result["frame"]
                    metadata = seq_result["metadata"]
                    continue
                elif seq_result.get("partial") or not seq_result.get("all_succeeded"):
                    # MoveSequence failed — build pending step, then Phase 3+4 below
                    cascade_level += 1
                    last_error = seq_msg
                    image = seq_result.get("frame", image)
                    metadata = seq_result.get("metadata", metadata)
                    result = {"success": False, "error": seq_msg,
                              "frame": image, "metadata": metadata}
                    step_index += 1
                    nonexecuted_retry_count = 0
                    memory.update(metadata, metadata.get("objects", []),
                                  "MoveSequence", False, seq_msg, task_criteria)
                    pending_step = self._build_step_entry(
                        ep.episode_id, config.branch_id, step_index - 1, parent_id,
                        "MoveSequence", proposed_params, result, eb_reasoning, injection_decision,
                    )
                    pending_step["error_type"] = "environment_failure"
                    diagnosis_history = eb_history + [pending_step]
                    # Track failed objectIds from the sequence
                    obj_id = proposed_params.get("steps", [{}])[-1].get("objectId") if proposed_params.get("steps") else None
                    if obj_id:
                        failed_object_ids.add(obj_id)
                    # Phase 3: Planner diagnoses
                    eb_phase3 = self.eb_agent.diagnose_failure(
                        task_goal=ep.data["task_goal"],
                        error_message=seq_msg,
                        image=image,
                        action_history=diagnosis_history,
                        cascade_level=cascade_level,
                        visible_objects=metadata.get("objects", []),
                        agent_pos=agent_pose.get("position"),
                        agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                        inventory_objects=inventory_objects,
                        task_criteria=task_criteria,
                        memory_text=memory.render(),
                    )
                    # Phase 4: Oracle evaluates
                    oracle_phase4 = self.oracle_agent.evaluate_failure(
                        task_goal=ep.data["task_goal"],
                        error_message=seq_msg,
                        image=image,
                        action_history=diagnosis_history,
                        eb_diagnosis=eb_phase3["diagnosis"],
                        eb_recovery_reasoning=eb_phase3["recovery_reasoning"],
                        eb_counterfactual=eb_phase3.get("counterfactual"),
                        eb_proposed_recovery=eb_phase3.get("proposed_recovery_action", {}),
                    )
                    step_entry = pending_step
                    step_entry["eb_diagnosis"] = eb_phase3.get("diagnosis")
                    step_entry["eb_recovery_reasoning"] = eb_phase3.get("recovery_reasoning")
                    step_entry["eb_counterfactual"] = eb_phase3.get("counterfactual")
                    step_entry["eb_proposed_recovery_action"] = eb_phase3.get("proposed_recovery_action")
                    step_entry["oracle_diagnosis_correct"] = oracle_phase4.get("diagnosis_correct")
                    step_entry["oracle_ground_truth"] = oracle_phase4.get("ground_truth")
                    step_entry["oracle_counterfactual_grade"] = oracle_phase4.get("counterfactual_grade")
                    step_entry["oracle_counterfactual_gold"] = oracle_phase4.get("counterfactual_gold")
                    step_entry["oracle_recovery_verdict"] = oracle_phase4.get("recovery_verdict")
                    self._write_success_step_direct(ep, step_entry)
                    eb_history.append(step_entry)
                    # Dead loop / unrecoverable checks
                    hard_unrec = check_unrecoverable(metadata, ep.data)
                    if hard_unrec:
                        result_br = BranchResult(branch_id=config.branch_id, termination_reason=f"unrecoverable_hard:{hard_unrec}", total_steps=step_index, fork_tasks=fork_tasks, fork_source_step_ids=fork_source_ids)
                        self._finalize(ep, config, result_br, fork_source_ids)
                        return result_br
                    if detect_dead_loop(eb_history):
                        result_br = BranchResult(branch_id=config.branch_id, termination_reason="dead_loop", total_steps=step_index, fork_tasks=fork_tasks, fork_source_step_ids=fork_source_ids)
                        self._finalize(ep, config, result_br, fork_source_ids)
                        return result_br
                    if oracle_phase4.get("recovery_verdict") == "unrecoverable":
                        result_br = BranchResult(branch_id=config.branch_id, termination_reason="unrecoverable", total_steps=step_index, fork_tasks=fork_tasks, fork_source_step_ids=fork_source_ids)
                        self._finalize(ep, config, result_br, fork_source_ids)
                        return result_br
                    continue

            # ==========================================================
            # 执行 EB 动作（objectType → objectId 解析 + 适配）
            # ==========================================================
            resolved, resolve_warning = resolve_object_ids(proposed_action, proposed_params,
                                                           metadata.get("objects", []))
            act, params = adapt(proposed_action, resolved)
            parent_id = f"s{step_index - 1}" if step_index > 0 else None

            # objectType not found in scene: log only, keep step_index unchanged, retry the EB request.
            if resolve_warning and act in _OBJECT_ACTIONS:
                last_error = (
                    "Your previous proposal was not executed. "
                    + resolve_warning
                    + " Re-think from the current image and output a valid JSON action."
                )
                nonexecuted_retry_count += 1
                self._write_failure_log(failure_log_path, {
                    "step_index": step_index,
                    "branch_id": config.branch_id,
                    "failure_type": "model_unresolved_object",
                    "proposed_action": proposed_action,
                    "proposed_params": proposed_params,
                    "eb_reasoning": eb_reasoning,
                    "error_message": last_error,
                    "retry_attempt": nonexecuted_retry_count,
                })
                if nonexecuted_retry_count >= max_nonexecuted_retries:
                    raise RuntimeError(last_error)
                continue

            try:
                result = env.step(act, **params)
            except Exception as exc:
                self._write_failure_log(failure_log_path, {
                    "step_index": step_index,
                    "branch_id": config.branch_id,
                    "failure_type": "system_env_step_exception",
                    "proposed_action": proposed_action,
                    "proposed_params": proposed_params,
                    "action": act,
                    "action_params": params,
                    "eb_reasoning": eb_reasoning,
                    "error_message": str(exc),
                })
                raise

            # --- 成功 → 写 JSON ---
            if result["success"]:
                cascade_level = 0
                last_error = None
                step_entry = self._build_step_entry(
                    ep.episode_id, config.branch_id, step_index, parent_id,
                    act, params, result, eb_reasoning, injection_decision,
                )
                self._write_success_step_direct(ep, step_entry)
                eb_history.append(step_entry)
                step_index += 1
                nonexecuted_retry_count = 0
                memory.update(metadata, metadata.get("objects", []), act, True, None, task_criteria)
                continue

            # --- 环境失败 ---
            step_index += 1
            nonexecuted_retry_count = 0

            # --- 环境失败 → Phase 3 + Phase 4 (写 JSON) ---
            cascade_level += 1
            last_error = result["error"]
            memory.update(metadata, metadata.get("objects", []), act, False, result["error"], task_criteria)

            # 追踪失败的 objectId
            obj_id = params.get("objectId")
            if obj_id:
                failed_object_ids.add(obj_id)

            pending_step = self._build_step_entry(
                ep.episode_id, config.branch_id, step_index - 1, parent_id,
                act, params, result, eb_reasoning, injection_decision,
            )
            pending_step["error_type"] = "environment_failure"
            diagnosis_history = eb_history + [pending_step]

            eb_phase3 = self.eb_agent.diagnose_failure(
                task_goal=ep.data["task_goal"],
                error_message=result["error"] or "Unknown error",
                image=result["frame"],
                action_history=diagnosis_history,
                cascade_level=cascade_level,
                visible_objects=metadata.get("objects", []),
                agent_pos=agent_pose.get("position"),
                agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                inventory_objects=inventory_objects,
                task_criteria=task_criteria,
                memory_text=memory.render(),
            )

            oracle_phase4 = self.oracle_agent.evaluate_failure(
                task_goal=ep.data["task_goal"],
                error_message=result["error"] or "Unknown error",
                image=result["frame"],
                action_history=diagnosis_history,
                eb_diagnosis=eb_phase3["diagnosis"],
                eb_recovery_reasoning=eb_phase3["recovery_reasoning"],
                eb_counterfactual=eb_phase3.get("counterfactual"),
                eb_proposed_recovery=eb_phase3.get("proposed_recovery_action", {}),
            )

            step_entry = pending_step
            step_entry["eb_diagnosis"] = eb_phase3.get("diagnosis")
            step_entry["eb_recovery_reasoning"] = eb_phase3.get("recovery_reasoning")
            step_entry["eb_counterfactual"] = eb_phase3.get("counterfactual")
            step_entry["eb_proposed_recovery_action"] = eb_phase3.get("proposed_recovery_action")
            step_entry["oracle_diagnosis_correct"] = oracle_phase4.get("diagnosis_correct")
            step_entry["oracle_ground_truth"] = oracle_phase4.get("ground_truth")
            step_entry["oracle_counterfactual_grade"] = oracle_phase4.get("counterfactual_grade")
            step_entry["oracle_counterfactual_gold"] = oracle_phase4.get("counterfactual_gold")
            step_entry["oracle_recovery_verdict"] = oracle_phase4.get("recovery_verdict")
            self._write_success_step_direct(ep, step_entry)
            eb_history.append(step_entry)

            # Hard-coded dead-loop / unrecoverable checks (does not depend on Oracle API)
            hard_unrec = check_unrecoverable(metadata, ep.data)
            if hard_unrec:
                result_br = BranchResult(
                    branch_id=config.branch_id, termination_reason=f"unrecoverable_hard:{hard_unrec}",
                    total_steps=step_index, fork_tasks=fork_tasks,
                    fork_source_step_ids=fork_source_ids,
                )
                self._finalize(ep, config, result_br, fork_source_ids)
                return result_br

            if detect_dead_loop(eb_history):
                result_br = BranchResult(
                    branch_id=config.branch_id, termination_reason="dead_loop",
                    total_steps=step_index, fork_tasks=fork_tasks,
                    fork_source_step_ids=fork_source_ids,
                )
                self._finalize(ep, config, result_br, fork_source_ids)
                return result_br

            if oracle_phase4.get("recovery_verdict") == "unrecoverable":
                result_br = BranchResult(
                    branch_id=config.branch_id, termination_reason="unrecoverable",
                    total_steps=step_index, fork_tasks=fork_tasks,
                    fork_source_step_ids=fork_source_ids,
                )
                self._finalize(ep, config, result_br, fork_source_ids)
                return result_br

            if self.enable_fork and oracle_phase4.get("should_fork") and eb_phase3.get("counterfactual"):
                grade = oracle_phase4.get("counterfactual_grade", "WA")
                if grade == "AC":
                    fork_task = self._build_fork_task(
                        config=config, current_step_idx=step_index - 1,
                        eb_phase3=eb_phase3, oracle_phase4=oracle_phase4,
                        history=eb_history, ep=ep,
                    )
                    if fork_task:
                        fork_tasks.append(fork_task)
                        fork_source_ids.append(f"s{step_index - 1}")

        return self._make_result(config, "unreachable", step_index,
                                 fork_tasks, fork_source_ids, ep)

    # ------------------------------------------------------------------
    # 续跑（resume）
    # ------------------------------------------------------------------

    @classmethod
    def resume(
        cls,
        episode_path: str,
        branch_id: str,
        eb_agent: EBAgent,
        oracle_agent: OracleAgent,
        output_dir: str,
        step_limit_multiplier: int = 4,
        enable_fork: bool = False,
    ) -> BranchResult:
        ep = EpisodeManager.load(episode_path)
        branch_steps = ep.get_steps_for_branch(branch_id)
        if not branch_steps:
            raise ValueError(f"Branch '{branch_id}' has no steps in {episode_path}")

        scene = ep.data["scene"]
        env = EnvController(scene=scene)
        env.reset_to_alfred_scene(_require_alfred_scene(ep.data))
        replay_steps(env, branch_steps, skip_failed=True)

        last_step = branch_steps[-1]
        start_idx = last_step["step_index_in_branch"] + 1
        config = BranchConfig(
            episode_id=ep.episode_id,
            branch_id=branch_id,
            parent_branch_id=None,
        )
        runner = cls(eb_agent, oracle_agent, output_dir,
                     step_limit_multiplier=step_limit_multiplier,
                     enable_fork=enable_fork)
        return runner.run(
            config=config, env=env, ep=ep,
            start_step_index=start_idx,
        )

    # ------------------------------------------------------------------
    # 内部：写入
    # ------------------------------------------------------------------

    def _build_step_entry(self, episode_id, branch_id, step_idx, parent_id,
                          action, params, result, eb_reasoning, injection_decision=None):
        image_dir = os.path.join(self.output_dir, episode_id)
        step = self.recorder.build_step(
            step_id=f"s{step_idx}", branch_id=branch_id, parent_step_id=parent_id,
            step_index=step_idx, action=action, action_params=params,
            result=result, image_dir=image_dir,
        )
        step["eb_reasoning"] = eb_reasoning
        step["oracle_injection_decision"] = injection_decision
        return step

    def _write_success_step(self, ep, branch_id, step_idx, parent_id,
                            action, params, result, eb_reasoning, injection_decision=None):
        step = self._build_step_entry(
            ep.episode_id, branch_id, step_idx, parent_id,
            action, params, result, eb_reasoning, injection_decision,
        )
        self._write_success_step_direct(ep, step)

    def _write_success_step_direct(self, ep, step_entry):
        ep.add_step(step_entry)

    def _write_failure_log(self, path: str, event: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _invalid_action_message(self, proposed_action: str) -> str:
        return (
            f'Your previous proposal was not executed. Reason: action "{proposed_action}" is invalid. '
            f"Allowed actions are: {', '.join(sorted(_VALID_ACTIONS))}. "
            "Re-think from the current image and output a valid JSON action."
        )

    def _execute_move_sequence(self, params, env, metadata, failure_log_path,
                                branch_id, step_index, episode_id,
                                eb_reasoning, injection_decision):
        """Execute a sequence of movement steps. Stop on first failure.

        params format: {"steps": [{"action": "MoveAhead", "repeat": 5}, ...]}
        Each step in the sequence is a movement action with optional repeat count (default 1).
        Returns (result_dict, message_string).
        """
        steps = params.get("steps", [])
        if not steps:
            return (
                {"success": False, "error": "MoveSequence has no steps"},
                "MoveSequence failed: empty steps list. Provide at least one movement step.",
            )

        _MOVEMENT_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
                             "RotateLeft", "RotateRight", "LookUp", "LookDown"}

        executed = []
        final_frame = None
        final_metadata = None
        all_succeeded = True

        for i, step in enumerate(steps):
            action = step.get("action", "")
            # Executor outputs {"action": "PickupObject", "params": {"objectType": "AlarmClock"}}
            if "params" in step and isinstance(step["params"], dict):
                step_params = dict(step["params"])
            else:
                step_params = {k: v for k, v in step.items() if k not in ("action", "repeat")}

            # Resolve objectType → objectId for object-interaction actions
            if action not in _MOVEMENT_ACTIONS and action != "Done":
                if action not in _VALID_ACTIONS:
                    desc = " → ".join(executed) if executed else "(nothing)"
                    msg = f"MoveSequence: {desc} succeeded, then invalid action '{action}' at step {i+1}."
                    return (
                        {"success": False, "error": msg, "partial": True,
                         "executed": executed, "frame": final_frame, "metadata": final_metadata},
                        msg,
                    )
                current_objs = (final_metadata or metadata).get("objects", [])
                resolved, warn = resolve_object_ids(action, step_params, current_objs)
                if warn and action in _OBJECT_ACTIONS:
                    desc = " → ".join(executed) if executed else "(nothing)"
                    msg = f"MoveSequence partially executed: {desc}, then {action} failed — {warn}"
                    return (
                        {"success": False, "error": msg, "partial": True,
                         "executed": executed, "frame": final_frame, "metadata": final_metadata},
                        msg,
                    )
                act, params = adapt(action, resolved)
            elif action in _MOVEMENT_ACTIONS:
                act = action
                params = {}
            elif action == "Done":
                act = "Done"
                params = {}
            else:
                desc = " → ".join(executed) if executed else "(nothing)"
                msg = f"MoveSequence: {desc} succeeded, then unknown action '{action}' at step {i+1}."
                return (
                    {"success": False, "error": msg, "partial": True,
                     "executed": executed, "frame": final_frame, "metadata": final_metadata},
                    msg,
                )

            repeat = max(1, int(step.get("repeat", 1)))
            repeat = min(repeat, 200)

            succeeded = 0
            for r in range(repeat):
                result = env.step(act, **params)
                if result["success"]:
                    succeeded += 1
                    final_frame = result["frame"]
                    final_metadata = result["metadata"]
                else:
                    all_succeeded = False
                    desc = self._format_seq(executed, action, succeeded, repeat)
                    if executed or succeeded > 0:
                        msg = (
                            f"MoveSequence partially executed: {desc}. "
                            f"Last failure: {result['error']}. "
                            f"The agent is now at a new position. Continue from here."
                        )
                        return (
                            {"success": False, "error": msg, "partial": True,
                             "executed": executed,
                             "failed_action": action,
                             "failed_at_repeat": succeeded,
                             "frame": (final_frame if final_frame is not None else result["frame"]),
                             "metadata": (final_metadata if final_metadata is not None else result["metadata"])},
                            msg,
                        )
                    else:
                        msg = (
                            f"MoveSequence: first step {action} failed immediately: {result['error']}. "
                            f"Nothing was executed."
                        )
                        return (
                            {"success": False, "error": msg, "partial": False,
                             "frame": result["frame"], "metadata": result["metadata"]},
                            msg,
                        )
                    break

            if not all_succeeded:
                break

            if succeeded > 0:
                executed.append(f"{action}×{succeeded}")
            # Continue to next step in sequence

        if all_succeeded:
            desc = " → ".join(executed) if executed else "(all steps executed)"
            msg = f"MoveSequence completed: {desc}."
            return (
                {"success": True, "error": None, "all_succeeded": True,
                 "executed": executed,
                 "frame": final_frame, "metadata": final_metadata},
                msg,
            )

        # Shouldn't reach here, but safe fallback
        return (
            {"success": False, "error": "MoveSequence unexpected state"},
            "MoveSequence failed unexpectedly.",
        )

    @staticmethod
    def _format_seq(executed, current_action, succeeded, total):
        parts = list(executed)
        if succeeded > 0:
            parts.append(f"{current_action}×{succeeded}/{total}")
        return " → ".join(parts) if parts else f"{current_action}×0/{total}"

    # ------------------------------------------------------------------
    # 内部：fork / finalize
    # ------------------------------------------------------------------

    def _make_result(self, config, reason, total_steps, fork_tasks, fork_source_ids, ep):
        result = BranchResult(
            branch_id=config.branch_id, termination_reason=reason,
            total_steps=total_steps, fork_tasks=fork_tasks,
            fork_source_step_ids=fork_source_ids,
        )
        self._finalize(ep, config, result, fork_source_ids)
        return result

    def _build_fork_task(self, config, current_step_idx, eb_phase3, oracle_phase4, history, ep):
        cf = eb_phase3.get("counterfactual")
        if not cf:
            return None

        # 结构化 counterfactual（新格式）
        if isinstance(cf, dict) and cf.get("target_step") is not None:
            target_idx = cf["target_step"]
            alt_action = cf.get("alternative_action", {})
            cf_text = cf.get("reasoning", "")
            # 在 history 中查找目标步
            target_step = None
            for s in history:
                if s.get("step_index_in_branch") == target_idx:
                    target_step = s
                    break
            if target_step is None:
                return None
            branches_from_id = target_step.get("parent_step_id")
            replaces_id = target_step.get("step_id")
        else:
            # 旧格式 string counterfactual — 用启发式方法
            cf_text = str(cf)
            target_step = None
            for s in reversed(history):
                if s.get("action") in ("PickupObject", "PutObject", "OpenObject", "CloseObject",
                                        "ToggleObjectOn", "ToggleObjectOff", "SliceObject"):
                    target_step = s
                    break
            if target_step is None and history:
                target_step = history[-1]
            if target_step is None:
                return None
            branches_from_id = target_step.get("parent_step_id")
            replaces_id = target_step.get("step_id")
            alt_action = eb_phase3.get("proposed_recovery_action", {})

        fork_branch_id = f"fork_s{target_step.get('step_index_in_branch', '?')}_{config.branch_id}"

        shared_ids = []
        for s in history:
            shared_ids.append(s.get("step_id", ""))
            if s.get("step_id") == branches_from_id:
                break

        return {
            "episode_id": config.episode_id,
            "branch_id": fork_branch_id,
            "parent_branch_id": config.branch_id,
            "shared_context_step_ids": shared_ids,
            "diverges_at_step_id": branches_from_id,
            "fork_config": {
                "replaces_step_id": replaces_id,
                "origin_step_id": f"s{current_step_idx}",
                "alternative_action": alt_action,
                "counterfactual_text": cf_text,
            },
        }

    def _finalize(self, ep, config, result, fork_source_ids):
        outcome = ep.data.get("final_outcome") or {"main_branch": None, "forks": []}
        branch_entry = {
            "branch_id": result.branch_id,
            "termination_reason": result.termination_reason,
            "total_steps": result.total_steps,
        }
        if result.branch_id == "main":
            outcome["main_branch"] = branch_entry
        else:
            branch_entry["fork_source_step_id"] = (
                fork_source_ids[0] if fork_source_ids else
                config.fork_config.get("origin_step_id", "?") if config.fork_config else "?"
            )
            branch_entry["counterfactual_verified"] = (
                result.termination_reason == "task_complete"
            )
            outcome["forks"].append(branch_entry)
        ep.set_final_outcome(outcome)


# ======================================================================
# 共享工具函数（供 scheduler 和 e2e_test 复用）
# ======================================================================

def _extract_task_target_types(traj: dict) -> set[str]:
    """从 ALFRED pddl_params 提取任务目标物体类型。"""
    pddl = traj.get("pddl_params", {}) or {}
    targets = set()
    for key in ("object_target", "parent_target", "mrecep_target", "toggle_target"):
        val = pddl.get(key, "")
        if val:
            targets.add(val)
    return targets


def replay_steps(env: EnvController, steps: list[dict], skip_failed: bool = True):
    """
    在环境中逐步 replay，恢复状态。
    用于 resume 和 fork 初始化。
    """
    for s in steps:
        action = s["action"]
        if action in ("Pass", "Done"):
            continue
        if skip_failed and not s.get("success", True):
            continue
        # Strict for now: a replay mismatch means the branch state is not trustworthy.
        # After the core system is stable, add an explicit repair policy here if needed.
        result = env.step(action, **s.get("action_params", {}))
        if not result["success"]:
            raise RuntimeError(
                f"Replay failed at {s.get('step_id')}: {action}({s.get('action_params', {})}) "
                f"-> {result['error']}"
            )


def _require_alfred_scene(data: dict) -> dict:
    scene_state = data.get("alfred_scene")
    if not scene_state or not scene_state.get("object_poses"):
        raise ValueError("Episode lacks ALFRED scene state; regenerate it from data/json_2.1.0")
    return scene_state


def _infer_pddl_params(task_goal: str, visible_objects: list[dict]) -> dict:
    """
    Match task_goal words against actual AI2-THOR objectType names from the state.
    task_goal: "move the red cloth from the cabinet to the tub"
    Checks: split goal phrase into words, match each word as substring of type names.
    Returns {"object_target": "Cloth", "parent_target": "Bathtub", ...}
    """
    goal = task_goal.lower()
    object_types = sorted(set(o["objectType"] for o in visible_objects))
    receptacle_types = sorted(set(o["objectType"] for o in visible_objects if o.get("receptacle")))

    def _match_word_in(phrase: str, candidates: list[str]) -> str:
        """For each word in phrase, check if any candidate type contains that word."""
        for word in phrase.split():
            for t in candidates:
                if word in t.lower():
                    return t
        return ""

    result = {}
    from_idx = goal.find(" from ")
    to_idx = goal.find(" to ") if from_idx >= 0 else -1
    with_idx = goal.find(" with ") if from_idx >= 0 else -1

    if from_idx >= 0:
        before_from = goal[:from_idx]  # "move the red cloth"
        result["object_target"] = _match_word_in(before_from, object_types)

    if to_idx >= 0:
        after_to_start = to_idx + 4
        after_to_end = with_idx if with_idx > to_idx else len(goal)
        after_to = goal[after_to_start:after_to_end]  # "the tub"
        result["parent_target"] = _match_word_in(after_to, receptacle_types)

    if with_idx >= 0:
        after_with = goal[with_idx + 6:]  # tool description
        result["toggle_target"] = _match_word_in(after_with, object_types)

    return result


def run_single_branch(
    traj_path: str,
    eb_agent: EBAgent,
    oracle_agent: OracleAgent,
    output_dir: str,
    trap_planner=None,
    enable_fork: bool = False,
    step_limit_multiplier: int = 4,
    executor_agent=None,
) -> BranchResult:
    """
    一个 episode 的完整生命周期：加载数据 → 初始化环境 → 陷阱 → 跑分支。
    scheduler 和 e2e_test 共用此入口。
    """
    from src.alfred_parser import load_traj, extract_metadata

    traj = load_traj(traj_path)
    meta = extract_metadata(traj)
    episode_id = os.path.basename(os.path.dirname(traj_path))

    env = EnvController(scene=meta["scene"])
    try:
        env.reset_to_alfred_scene(meta["alfred_scene"])

        # Infer pddl_params only if not already present (ALFRED data has them)
        if not meta.get("pddl_params"):
            state = env.get_state_snapshot()
            meta["pddl_params"] = _infer_pddl_params(meta.get("task_goal", ""),
                                                        state["metadata"].get("objects", []))

        ep = EpisodeManager(episode_id, output_dir, meta)

        if trap_planner:
            failure_log_path = os.path.join(output_dir, episode_id, "failures_main.jsonl")
            os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
            state = env.get_state_snapshot()
            task_targets = _extract_task_target_types(traj)
            traps = trap_planner.plan_traps(
                task_type=meta["task_type"],
                scene_objects=state["metadata"]["objects"],
                exclude_types=task_targets,
            )
            results = trap_planner.apply_traps(env.controller, traps)
            for t, r in zip(traps, results):
                if r["success"]:
                    ep.add_initial_trap({
                        "trap_id": t["trap_id"],
                        "failure_type": t["failure_type"],
                        "description": t["description"],
                        "injection": t["injection"],
                        "severity": t["severity"],
                    })
                else:
                    event = {
                        "step_index": None,
                        "branch_id": "main",
                        "failure_type": "initial_trap_setup_failed",
                        "trap": t,
                        "error_message": r.get("error"),
                    }
                    with open(failure_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")
                    raise RuntimeError(
                        f"Initial trap setup failed for {t.get('trap_id')}: {r.get('error')}"
                    )

        runner = BranchRunner(eb_agent, oracle_agent, output_dir,
                              step_limit_multiplier=step_limit_multiplier,
                              enable_fork=enable_fork,
                              enable_phase2=(trap_planner is not False and trap_planner is not None),
                              executor_agent=executor_agent)

        config = BranchConfig(
            episode_id=episode_id,
            branch_id="main",
            parent_branch_id=None,
        )
        return runner.run(config=config, env=env, ep=ep)
    finally:
        env.close()
