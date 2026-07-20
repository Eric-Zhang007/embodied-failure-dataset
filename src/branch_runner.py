"""
分支执行器。运行单个分支的 Phase 1-4 交互循环直到终止。
成功步骤写入 episode JSON，失败事件写入独立 failure log。
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import AgenticOracle, OracleAgent
from src.env_controller import EnvController
from src.env_injector import inject
from src.step_recorder import StepRecorder
from src.episode_manager import EpisodeManager
from src.action_adapter import adapt, resolve_object_ids, _OBJECT_ACTIONS
from src.task_conditions import check_task_complete, check_unrecoverable, detect_dead_loop, get_completion_criteria_text
from src.context_builder import build_branch_history
from src.egocentric_memory import EgocentricMemory, SearchTrail
from src.semantic_memory import SemanticMemory
from src.critic_guard import CriticGuard

_VALID_ACTIONS = {
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown",
    "PickupObject", "PutObject", "OpenObject", "CloseObject",
    "ToggleObjectOn", "ToggleObjectOff", "SliceObject", "BreakObject", "FillObjectWithLiquid", "EmptyLiquidFromObject", "DropHandObject",
    "Done", "LookAround", "MoveSequence",
}

# Meta-actions: handled by Planner / branch_runner, must never reach AI2-THOR.
# Executor and Phase 3 recovery are forbidden from outputting these.
_META_ACTIONS = {"Done", "LookAround"}
_EXECUTABLE_SEQUENCE_ACTIONS = _VALID_ACTIONS - _META_ACTIONS - {"MoveSequence"}
_REPEATABLE_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
_MAX_SEQUENCE_ACTIONS = 12
_MAX_ACTION_REPEAT = 200


def _sid(branch_id: str, idx: int) -> str:
    """Globally-unique step id: '<branch>__s<idx>'.

    Prefixing with branch_id makes step_id unique across branches (main +
    forks) within one episode file. The '__' separator is filesystem-safe on
    Windows, so it stays valid as a PNG filename ('<branch>__s<idx>.png').
    step_id is treated as an opaque key everywhere (never int-parsed), so the
    prefix is safe. Fork branches share one image dir, so this also prevents
    frame-filename collisions between branches.
    """
    return f"{branch_id}__s{idx}"


def _last_branch_step_id(ep, branch_id: str) -> Optional[str]:
    """Return the step_id of the most recently recorded step on this branch,
    or None if the branch has no steps yet.

    Used to compute parent_step_id by reading actual episode state rather than
    recomputing a string — correct for fresh runs, forks, and resume across the
    old-format/new-format seam (old episodes have bare 's<idx>' ids)."""
    last = None
    for s in ep.data.get("steps", []):
        if s.get("branch_id") == branch_id:
            last = s.get("step_id")
    return last

# AI2-THOR cameraHorizon uses positive degrees down from level: LookDown adds
# 30 degrees and LookUp subtracts 30. Controller.plan_horizons explicitly
# enumerates 330° (-30° up), 0°, 30°, and 60° as supported horizons.
_CAMERA_HORIZON_MIN = -30.0
_CAMERA_HORIZON_MAX = 60.0
_CAMERA_HORIZON_EPSILON = 0.01
_CAMERA_LOOK_STEP_DEGREES = 30.0


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
        enable_critic: bool = False,
        critic_llm: bool = False,
        enable_searched_markers: bool = True,
        enable_intent_dedup: bool = True,
        enable_critic_guard: bool = False,
        enable_curiosity_scoreboard: bool = True,
        enable_contrastive_planner: bool = True,
        enable_progress_gating: bool = True,
        enable_search_trail: bool = True,
        memory_mode: str = "semantic",  # "semantic" | "geometric"
        agentic_oracle: bool = False,
    ):
        self.eb_agent = eb_agent
        self.oracle_agent = oracle_agent
        self.agentic = AgenticOracle(oracle_agent.client) if agentic_oracle else None
        self.executor_agent = executor_agent
        self.output_dir = output_dir
        self.enable_phase2 = enable_phase2
        self.step_limit_multiplier = step_limit_multiplier
        self.enable_fork = enable_fork
        self.recorder = StepRecorder()
        self._last_scan_step = -100
        self.memory_mode = memory_mode
        self.enable_critic = enable_critic
        # CriticGuard (002b): validates Planner intents before Executor execution
        self.critic_guard = CriticGuard(
            use_llm=critic_llm,
            vlm_client=eb_agent.client if critic_llm else None,
            max_regen_attempts=1,
        )
        # ── Ablation experiment flags ──
        # Preserve the constructor values so CLI ablation selections reach both
        # the runtime behavior and the persisted experiment configuration.
        self.enable_searched_markers: bool = enable_searched_markers     # 001 — mark receptacles as SEARCHED
        self.enable_intent_dedup: bool = enable_intent_dedup              # 002a — block repeated failed intents
        self.enable_critic_guard: bool = enable_critic_guard              # 002b — pre-execution intent validation
        self.enable_curiosity_scoreboard: bool = enable_curiosity_scoreboard  # 003 — exploration priority table
        self.enable_contrastive_planner: bool = enable_contrastive_planner  # 004 — dual-intent selection
        self.enable_progress_gating: bool = enable_progress_gating        # 005 — phase-aware guardrail injection
        self.enable_search_trail: bool = enable_search_trail              # 006 — trail-based revisit scoring

    def _agentic_review(self, ep, config, step_index, step_entry, env, image, metadata, history, memory_text):
        if not self.agentic:
            return
        review = self.agentic.periodic_review(
            image=image,
            task_goal=ep.data["task_goal"],
            visible_objects=metadata.get("objects", []),
            action_history=history + [step_entry],
            memory_text=memory_text,
        )
        step_entry["agentic_oracle_review"] = review
        intervention = review.get("intervention")
        if review.get("action") != "intervene" or not isinstance(intervention, dict):
            return
        tool = intervention.get("tool")
        params = intervention.get("params", {})
        if not tool or not isinstance(params, dict):
            return
        result = inject(env.controller, method=tool, **params)
        step_entry["agentic_intervention_result"] = result
        if result.get("success"):
            ep.add_runtime_trap({
                "trap_id": f"agentic_trap_{config.branch_id}_{step_index}",
                "branch_id": config.branch_id,
                "created_at_step_id": _sid(config.branch_id, step_index),
                "created_by": "agentic_oracle",
                "injection": {"method": tool, "params": params},
                "modification_success": True,
                "triggered_at_step_id": None,
                "env_error": None,
                "status": "active",
            })

    # ------------------------------------------------------------------
    # Trail helper (Spike 006: search-trail-cost)
    # ------------------------------------------------------------------

    def _record_trail(self, trail: SearchTrail, metadata: dict) -> None:
        """Record agent position in the trail grid after a successful step.
        Guarded by enable_search_trail ablation flag (Spike 006)."""
        if not self.enable_search_trail:
            return
        agent = metadata.get("agent", {})
        pos = agent.get("position", {})
        trail.record(pos.get("x", 0.0), pos.get("z", 0.0))

    @staticmethod
    def _render_trail(trail: SearchTrail, metadata: dict) -> str:
        """Build trail text for Planner/Executor prompts."""
        objects = metadata.get("objects", [])
        receptacles = [o for o in objects if o.get("receptacle")]
        if not receptacles:
            return ""
        return trail.render_summary(receptacles)

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
        stuck_escalation_count = 0  # StuckTracker interventions without recovery
        fork_tasks: list[dict] = []
        fork_source_ids: list[str] = []
        last_error: Optional[str] = None

        # ── Propagate ablation flags to eb_agent ──
        self.eb_agent.enable_intent_dedup = self.enable_intent_dedup
        self.eb_agent.enable_curiosity_scoreboard = self.enable_curiosity_scoreboard
        self.eb_agent.enable_progress_gating = self.enable_progress_gating

        # ── Log ablation config to episode JSON for post-hoc analysis ──
        ep.data["ablation_config"] = {
            "001_searched_markers": self.enable_searched_markers,
            "002a_intent_dedup": self.enable_intent_dedup,
            "002b_critic_guard": self.enable_critic_guard,
            "003_curiosity_scoreboard": self.enable_curiosity_scoreboard,
            "004_contrastive_planner": self.enable_contrastive_planner,
            "005_progress_gating": self.enable_progress_gating,
            "006_search_trail": self.enable_search_trail,
        }
        ep._flush()

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
        if self.memory_mode == "geometric":
            memory = EgocentricMemory()
        else:
            memory = SemanticMemory()
        trail = SearchTrail(resolution=0.5)
        intent_history: list[dict] = []  # completed/failed intent summaries
        current_intent = {"intent": "", "target": "", "steps": [], "start_step": 0}
        phase2_injection_count = 0
        phase2_max_injections = 3
        nonexecuted_retry_count = 0
        max_nonexecuted_retries = 3
        zero_step_completion_steps: dict[tuple[str, str], int] = {}

        # ==========================================================
        # Initial LookAround: 开局强制四向扫描
        # ==========================================================
        if step_index == 0:
            state0 = env.get_state_snapshot()
            image0 = state0["frame"]
            metadata0 = state0["metadata"]
            look_images = []
            look_metas = []  # accumulate metadata from each direction for memory
            for d in ["ahead", "left", "behind", "right"]:
                if d == "ahead":
                    snap = image0
                    meta = metadata0
                else:
                    r = env.step("RotateLeft")
                    if not r["success"]:
                        raise RuntimeError(f"Init LookAround RotateLeft failed: {r['error']}")
                    snap = r["frame"]
                    meta = r["metadata"]
                look_images.append((d, snap))
                look_metas.append((d, meta))
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
            memory.update(metadata0, metadata0.get("objects", []), "LookAround", True, None, tc0, "initial scan")
            # Also feed metadata from left/behind/right directions into memory
            for _d_label, _d_meta in look_metas[1:]:  # skip ahead (already done)
                memory.update(_d_meta, _d_meta.get("objects", []), "LookAround", True, None, tc0, "initial scan")
            self._record_trail(trail, metadata0)
            step_index = 1
            # Use new scan analysis to get direction + intent (not old propose_action)
            scan_result = self.eb_agent.analyze_scan_room(
                task_goal=ep.data["task_goal"],
                look_images=look_images,
                visible_objects=metadata0.get("objects", []),
                action_history=eb_history,
                last_error=None,
                inventory_objects=[],
                task_criteria=tc0,
                memory_text=memory.render(),
            )
            # Rotate to face the determined direction
            face_dir = scan_result.get("face_direction", "ahead")
            dir_rotations = {"right": 1, "behind": 2, "left": 3, "ahead": 0}
            for _ in range(dir_rotations.get(face_dir, 0)):
                env.step("RotateLeft")
            # Let the Planner→Executor cycle handle the first intent naturally
            proposed_action = ""  # cleared, so step 1 enters Planner→Executor cycle
            proposed_params = {}
            eb_reasoning = ""

        while True:
            if step_index >= 200:
                return self._make_result(config, "step_hard_limit", step_index, fork_tasks, fork_source_ids, ep)
            parent_id = _last_branch_step_id(ep, config.branch_id)
            injection_decision = None
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
                    _last_branch_step_id(ep, config.branch_id),
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

                # 1. Planner proposes intent (sees intent history + recent raw steps)
                planner_intent = self.eb_agent.plan_intent(
                    task_goal=ep.data["task_goal"],
                    image=image,
                    visible_objects=metadata.get("objects", []),
                    action_history=eb_history[-3:] if len(eb_history) > 3 else eb_history,
                    last_error=last_error,
                    agent_pos=agent_pose.get("position"),
                    agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                    inventory_objects=inventory_objects,
                    hand_status=hand_status,
                    task_criteria=task_criteria,
                    memory_text=memory.render(),
                    intent_history=intent_history,
                    memory=memory,
                    trail_text=self._render_trail(trail, metadata) if self.enable_search_trail else "",
                )
                # ── Contrastive Planner (Spike 004): dual-intent selection ──
                # Only activate when the Planner appears stuck (last 3 intents
                # target the same location). Based on Dynamic Self-Consistency
                # (RASC, 2024): extra samples only needed when uncertain.
                if self.enable_contrastive_planner and EBAgent._should_activate_contrastive(intent_history):
                    planner_intent_b = self.eb_agent.plan_intent_explore(
                        task_goal=ep.data["task_goal"],
                        image=image,
                        visible_objects=metadata.get("objects", []),
                        action_history=eb_history[-3:] if len(eb_history) > 3 else eb_history,
                        last_error=last_error,
                        agent_pos=agent_pose.get("position"),
                        agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                        inventory_objects=inventory_objects,
                        hand_status=hand_status,
                        task_criteria=task_criteria,
                        memory_text=memory.render(),
                        intent_history=intent_history,
                        memory=memory,
                    )
                    # Select between A (exploit) and B (explore)
                    chosen, chosen_label, reason = EBAgent.select_intent(
                        intent_a=planner_intent,
                        intent_b=planner_intent_b,
                        intent_history=intent_history,
                    )
                    self.eb_agent.contrastive_stats["activations"] += 1
                    if chosen_label == "A":
                        self.eb_agent.contrastive_stats["a_selected"] += 1
                    else:
                        self.eb_agent.contrastive_stats["b_selected"] += 1
                    self.eb_agent.contrastive_stats["selections"].append({
                        "step": step_index,
                        "chosen": chosen_label,
                        "a_intent": f"{planner_intent.get('intent','')} ({planner_intent.get('target','')})",
                        "b_intent": f"{planner_intent_b.get('intent','')} ({planner_intent_b.get('target','')})",
                        "reason": reason,
                    })
                    # Log the selection to failure log for later analysis
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "contrastive_selection",
                        "chosen": chosen_label,
                        "reason": reason,
                        "intent_a": {
                            "intent": planner_intent.get("intent"),
                            "target": planner_intent.get("target"),
                            "reasoning": planner_intent.get("reasoning", "")[:200],
                        },
                        "intent_b": {
                            "intent": planner_intent_b.get("intent"),
                            "target": planner_intent_b.get("target"),
                            "reasoning": planner_intent_b.get("reasoning", "")[:200],
                        },
                    })
                    planner_intent = chosen

                intent = planner_intent["intent"]
                intent_target = planner_intent.get("target", "")

                # ── Meta-intent: UNMARK ──
                # The Planner can explicitly undo a SEARCHED mark on a receptacle
                # when it suspects the target was missed (occlusion, angle, etc.).
                if intent.startswith("unmark "):
                    receptacle_type = intent_target
                    if not receptacle_type:
                        receptacle_type = intent.split("unmark ", 1)[1].strip()
                    count = memory.unmark_searched(object_type=receptacle_type)
                    import logging
                    if count > 0:
                        logging.info(
                            "UNMARK intent: unmarked %d instance(s) of '%s'",
                            count, receptacle_type,
                        )
                    else:
                        logging.debug(
                            "UNMARK intent: no searched instances found for '%s'",
                            receptacle_type,
                        )
                    last_error = None
                    continue

                # Trail soft scoring (Spike 006): warn if target area visited 3+ times
                if self.enable_search_trail:
                    _RECEPTACLE_INTENT_KW = ("approach", "open", "close", "check", "search", "look")
                    if intent_target and any(kw in intent.lower() for kw in _RECEPTACLE_INTENT_KW):
                        trail_score = _check_trail_revisit(trail, intent_target, metadata.get("objects", []))
                        if trail_score >= 3:
                            self.eb_agent._pending_trail_warning = (
                                f"You have already visited the {intent_target} area "
                                f"{int(trail_score)} times. Consider whether this is "
                                f"productive -- are there unvisited areas to explore first?"
                            )

                # Track intent -- if intent changed, close previous and start new
                prev_intent_key = (current_intent.get("intent", ""), current_intent.get("target", ""))
                new_intent_key = (intent, intent_target)
                if new_intent_key != prev_intent_key:
                    if current_intent.get("steps"):
                        current_intent["completed"] = current_intent["steps"][-1].get("success", False)
                        current_intent["end_step"] = step_index
                        if not current_intent["completed"]:
                            last_step = current_intent["steps"][-1]
                            diag = last_step.get("eb_diagnosis", "")
                            recovery = last_step.get("eb_recovery_reasoning", "")
                            if diag:
                                current_intent["eb_diagnosis"] = diag
                            if recovery:
                                current_intent["eb_recovery_reasoning"] = recovery
                        intent_history.append(dict(current_intent))
                        # Mark receptacles as searched when agent moves on (Spike 001)
                        if self.enable_searched_markers:
                            _check_and_mark_searched(current_intent, intent_target, memory,
                                                     enable_curiosity=self.enable_curiosity_scoreboard)
                    current_intent = {"intent": intent, "target": intent_target,
                                      "steps": [], "start_step": step_index}

                # ── Special intent: Done → task_conditions hard check ──
                if intent == "Done":
                    task_complete_done, done_reason = check_task_complete(metadata, ep.data, task_state)
                    if task_complete_done:
                        done_step = self._build_step_entry(
                            ep.episode_id, config.branch_id, step_index, parent_id,
                            "Done", {}, {"success": True, "error": None, "frame": image, "metadata": metadata},
                            planner_intent.get("reasoning", ""), injection_decision,
                        )
                        self._write_success_step_direct(ep, done_step)
                        return self._make_result(config, "task_complete", step_index + 1,
                                                 fork_tasks, fork_source_ids, ep)
                    last_error = f"Done rejected: {done_reason}. Fix the unmet criteria before calling Done."
                    continue

                # ── Scan cooldown: block repeat scans within 3 steps ──
                if intent == "scan room" and (step_index - self._last_scan_step) <= 3:
                    last_error = (
                        "scan room rejected: last scan was only "
                        f"{step_index - self._last_scan_step} steps ago. "
                        "Use the SPATIAL MEMORY section to pick a concrete approach target."
                    )
                    continue

                # ── Special intent: scan room → 4-view capture → 32B analysis ──
                if intent == "scan room":
                    self._last_scan_step = step_index
                    look_images = []
                    look_dirs = ["ahead", "left", "behind", "right"]
                    scan_metas = []  # accumulate metadata from each direction
                    for d in look_dirs:
                        if d == "ahead":
                            snap = image
                            meta = metadata
                        else:
                            r = env.step("RotateLeft")
                            if not r["success"]:
                                raise RuntimeError(f"Scan room RotateLeft failed: {r['error']}")
                            snap = r["frame"]
                            meta = r["metadata"]
                        look_images.append((d, snap))
                        scan_metas.append((d, meta))
                    env.step("RotateLeft")  # restore original facing
                    # Record LookAround step
                    look_result = {"success": True, "error": None, "frame": image, "metadata": metadata}
                    look_step = self._build_step_entry(
                        ep.episode_id, config.branch_id, step_index, parent_id,
                        "LookAround", {}, look_result, planner_intent.get("reasoning", ""), injection_decision,
                    )
                    image_dir = os.path.join(self.output_dir, ep.episode_id)
                    view_paths = []
                    for label, frame in look_images:
                        vp = os.path.join(image_dir, f"{_sid(config.branch_id, step_index)}_look_{label}.png")
                        StepRecorder.save_frame(frame, vp)
                        view_paths.append({"label": label, "image_path": vp})
                    look_step["lookaround_views"] = view_paths
                    self._write_success_step_direct(ep, look_step)
                    eb_history.append(look_step)
                    parent_id = _sid(config.branch_id, step_index)
                    step_index += 1
                    memory.update(metadata, metadata.get("objects", []),
                                  "LookAround", True, None, task_criteria)
                    self._record_trail(trail, metadata)
                    # 32B multi-image analysis -> direction + intent
                    scan_result = self.eb_agent.analyze_scan_room(
                        task_goal=ep.data["task_goal"],
                        look_images=look_images,
                        visible_objects=metadata.get("objects", []),
                        action_history=eb_history,
                        last_error=last_error,
                        inventory_objects=inventory_objects,
                        task_criteria=task_criteria,
                        memory_text=memory.render(),
                    )
                    # Rotate to face the determined direction
                    face_dir = scan_result.get("face_direction", "ahead")
                    dir_rotations = {"right": 1, "behind": 2, "left": 3, "ahead": 0}
                    for _ in range(dir_rotations.get(face_dir, 0)):
                        env.step("RotateLeft")
                    # Refresh state after rotation
                    snap = env.step("Pass")
                    image = snap["frame"]
                    metadata = snap["metadata"]
                    memory.update(metadata, metadata.get("objects", []),
                                  "LookAround", True, None, task_criteria,
                                  f"{intent} {intent_target}".strip())
                    intent = scan_result.get("intent", intent)
                    intent_target = scan_result.get("target", intent_target)
                    # Falls through to Executor below with the new intent

                # -- CriticGuard (002b): validate intent before Executor --
                if (self.enable_critic_guard or self.enable_critic) and intent not in ("Done", "scan room"):
                    critic_result = self.critic_guard.check(
                        intent=intent,
                        target=intent_target,
                        intent_history=intent_history,
                        visible_objects=metadata.get("objects", []),
                        memory_text=memory.render(),
                        action_history=eb_history,
                    )
                    if not critic_result.get("approved"):
                        self.critic_guard.stats.regen_attempted += 1
                        feedback = self.critic_guard.feedback_for_planner(
                            critic_result.get("reason", ""),
                            critic_result.get("suggestion", ""),
                        )
                        regen_intent = self.eb_agent.plan_intent(
                            task_goal=ep.data["task_goal"],
                            image=image,
                            visible_objects=metadata.get("objects", []),
                            action_history=eb_history[-3:] if len(eb_history) > 3 else eb_history,
                            last_error=last_error,
                            agent_pos=agent_pose.get("position"),
                            agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                            inventory_objects=inventory_objects,
                            hand_status=hand_status,
                            task_criteria=task_criteria,
                            memory_text=memory.render(),
                            intent_history=intent_history,
                            critic_feedback=feedback,
                            memory=memory,
                        )
                        intent = regen_intent.get("intent", intent)
                        intent_target = regen_intent.get("target", intent_target)
                        regen_critic = self.critic_guard.check(
                            intent=intent,
                            target=intent_target,
                            intent_history=intent_history,
                            visible_objects=metadata.get("objects", []),
                            memory_text=memory.render(),
                            action_history=eb_history,
                        )
                        if not regen_critic.get("approved"):
                            self.critic_guard.stats.regen_rejected += 1
                            # B13: Force hardcoded exploration fallback instead of
                            # letting the second rejection through to the Executor.
                            # Avoids wasting a Planner re-gen call on a third attempt.
                            intent = "explore area"
                            intent_target = ""
                        else:
                            self.critic_guard.stats.regen_approved += 1

                # 2. Executor proposes actions (sees only current-intent history)
                exec_result = self.executor_agent.execute_intent(
                    intent=intent,
                    target=intent_target,
                    image=image,
                    visible_objects=metadata.get("objects", []),
                    action_history=current_intent.get("steps", []),
                    last_error=last_error,
                    agent_pos=agent_pose.get("position"),
                    agent_rot_y=agent_pose.get("rotation", {}).get("y", 0.0),
                    hand_status=hand_status,
                    task_criteria=task_criteria,
                    memory_text=memory.render(),
                    failed_object_ids=failed_object_ids,
                    camera_horizon=agent_pose.get("cameraHorizon", 0.0),
                    trail_text=self._render_trail(trail, metadata) if self.enable_search_trail else "",
                    planner_reasoning=planner_intent.get("reasoning", ""),
                )

                executor_actions = exec_result.get("actions")
                executor_status = exec_result.get("status")

                # An already-satisfied intent is the sole valid zero-action
                # contract. Complete it without inventing an environment failure.
                if executor_status == "done" and executor_actions == []:
                    completed_intent_key = (intent, intent_target)
                    if zero_step_completion_steps.get(completed_intent_key) == step_index:
                        repeat_error = (
                            f"Intent {intent!r} targeting {intent_target!r} was already completed "
                            "without actions on the previous Planner cycle. Choose the next distinct "
                            "intent instead of repeating it."
                        )
                        nonexecuted_retry_count += 1
                        self._write_failure_log(failure_log_path, {
                            "step_index": step_index,
                            "branch_id": config.branch_id,
                            "failure_type": "model_repeated_completed_intent",
                            "intent": intent,
                            "intent_target": intent_target,
                            "executor_status": executor_status,
                            "executor_status_reason": exec_result.get("status_reason", ""),
                            "error_message": repeat_error,
                            "retry_attempt": nonexecuted_retry_count,
                        })
                        if nonexecuted_retry_count >= max_nonexecuted_retries:
                            raise RuntimeError(repeat_error)
                        last_error = repeat_error
                        continue

                    current_intent["intent"] = intent
                    current_intent["target"] = intent_target
                    current_intent["completed"] = True
                    current_intent["end_step"] = step_index
                    current_intent["completion_reason"] = exec_result.get("status_reason", "")
                    intent_history.append(dict(current_intent))
                    current_intent = {
                        "intent": "",
                        "target": "",
                        "steps": [],
                        "start_step": step_index,
                    }
                    zero_step_completion_steps[completed_intent_key] = step_index
                    last_error = None
                    nonexecuted_retry_count = 0
                    continue

                executor_contract_error = self._validate_executor_output_shape(
                    executor_actions, "executor"
                )
                if executor_contract_error:
                    nonexecuted_retry_count += 1
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "model_invalid_action_sequence",
                        "proposal_source": "executor",
                        "executor_status": executor_status,
                        "proposed_steps": executor_actions,
                        "error_message": executor_contract_error,
                        "retry_attempt": nonexecuted_retry_count,
                    })
                    if nonexecuted_retry_count >= max_nonexecuted_retries:
                        raise RuntimeError(executor_contract_error)
                    last_error = executor_contract_error
                    continue

                executor_sequence_error = self._validate_action_sequence(
                    executor_actions, "executor action sequence"
                )
                if executor_sequence_error:
                    nonexecuted_retry_count += 1
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "model_invalid_action_sequence",
                        "proposal_source": "executor",
                        "executor_status": executor_status,
                        "proposed_steps": executor_actions,
                        "error_message": executor_sequence_error,
                        "retry_attempt": nonexecuted_retry_count,
                    })
                    if nonexecuted_retry_count >= max_nonexecuted_retries:
                        raise RuntimeError(executor_sequence_error)
                    last_error = executor_sequence_error
                    continue

                # 3. Planner reviews — if rejected, Planner provides corrected actions
                review = self.eb_agent.review_actions(
                    intent=intent,
                    target=intent_target,
                    proposed_actions=executor_actions,
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
                    selected_steps = exec_result.get("actions")
                    proposal_source = "executor"
                    eb_reasoning = exec_result.get("reasoning", "")
                else:
                    selected_steps = review.get("corrected_actions")
                    proposal_source = "reviewer correction"
                    eb_reasoning = review.get("reason", "")

                sequence_error = self._validate_executor_result(
                    selected_steps, exec_result.get("status"), proposal_source
                )
                if sequence_error:
                    nonexecuted_retry_count += 1
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "model_invalid_action_sequence",
                        "proposal_source": proposal_source,
                        "executor_status": exec_result.get("status"),
                        "proposed_steps": selected_steps,
                        "eb_reasoning": eb_reasoning,
                        "error_message": sequence_error,
                        "retry_attempt": nonexecuted_retry_count,
                    })
                    if nonexecuted_retry_count >= max_nonexecuted_retries:
                        raise RuntimeError(sequence_error)
                    last_error = sequence_error
                    continue

                proposed_action = "MoveSequence"
                proposed_params = {"steps": selected_steps}
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

            if proposed_action not in _META_ACTIONS and proposed_action != "MoveSequence":
                action_error = self._validate_standalone_action(
                    proposed_action,
                    proposed_params,
                    metadata.get("agent", {}).get("cameraHorizon", 0.0),
                )
                if action_error:
                    nonexecuted_retry_count += 1
                    failure_type = (
                        "model_camera_horizon_violation"
                        if proposed_action in {"LookUp", "LookDown"}
                        else "model_invalid_action_params"
                    )
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": failure_type,
                        "proposed_action": proposed_action,
                        "proposed_params": proposed_params,
                        "eb_reasoning": eb_reasoning,
                        "error_message": action_error,
                        "retry_attempt": nonexecuted_retry_count,
                    })
                    if nonexecuted_retry_count >= max_nonexecuted_retries:
                        raise RuntimeError(action_error)
                    last_error = action_error
                    continue

            # Validate MoveSequence before Phase 2, including the legacy path.
            if proposed_action == "MoveSequence":
                sequence_error = self._validate_action_sequence(
                    proposed_params.get("steps") if isinstance(proposed_params, dict) else None,
                    "action sequence",
                )
                if sequence_error:
                    nonexecuted_retry_count += 1
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "model_invalid_action_sequence",
                        "proposed_action": proposed_action,
                        "proposed_params": proposed_params,
                        "eb_reasoning": eb_reasoning,
                        "error_message": sequence_error,
                        "retry_attempt": nonexecuted_retry_count,
                    })
                    if nonexecuted_retry_count >= max_nonexecuted_retries:
                        raise RuntimeError(sequence_error)
                    last_error = sequence_error
                    continue
                horizon_error = self._validate_camera_horizon_sequence(
                    proposed_params["steps"], metadata.get("agent", {}).get("cameraHorizon", 0.0)
                )
                if horizon_error:
                    nonexecuted_retry_count += 1
                    self._write_failure_log(failure_log_path, {
                        "step_index": step_index,
                        "branch_id": config.branch_id,
                        "failure_type": "model_camera_horizon_violation",
                        "proposed_action": proposed_action,
                        "proposed_params": proposed_params,
                        "eb_reasoning": eb_reasoning,
                        "error_message": horizon_error,
                        "retry_attempt": nonexecuted_retry_count,
                    })
                    if nonexecuted_retry_count >= max_nonexecuted_retries:
                        raise RuntimeError(horizon_error)
                    last_error = horizon_error
                    continue

            # Done 检测：仅在成功步之后检查，避免初始空手状态误判
            if proposed_action == "Done":
                task_complete_done, done_reason = check_task_complete(metadata, ep.data, task_state)
                if task_complete_done:
                    self._write_success_step(
                        ep, config.branch_id, step_index,
                        _last_branch_step_id(ep, config.branch_id),
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
                    _last_branch_step_id(ep, config.branch_id),
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
            if not self.agentic and self.enable_phase2 and cascade_level <= 1 and remaining > 0:
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
                            "created_at_step_id": _sid(config.branch_id, step_index),
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
            elif not self.agentic and self.enable_phase2 and remaining <= 0:
                injection_decision = {
                    "decided_to_inject": False,
                    "reasoning": "LIMIT REACHED: all 3 successful injections have been used.",
                    "injection": None,
                }

            # ==========================================================
            # LookAround: 站原地旋转 4 次，截 4 张图发给 EB 做多图综合分析。
            # 如果 EB 扫完还想再扫，记录为无效决策并回到普通 Phase 1。
            # ==========================================================
            if proposed_action == "LookAround":
                lookaround_reasoning = eb_reasoning
                look_images = []
                look_metas_legacy = []
                look_dirs = ["ahead", "left", "behind", "right"]
                for d in look_dirs:
                    if d == "ahead":
                        snap = image
                        meta = metadata
                    else:
                        rotate_result = env.step("RotateLeft")
                        if not rotate_result["success"]:
                            raise RuntimeError(f"LookAround internal RotateLeft failed: {rotate_result['error']}")
                        snap = rotate_result["frame"]
                        meta = rotate_result["metadata"]
                    look_images.append((d, snap))
                    look_metas_legacy.append((d, meta))
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
                    view_path = os.path.join(image_dir, f"{_sid(config.branch_id, step_index)}_look_{label}.png")
                    StepRecorder.save_frame(frame, view_path)
                    view_paths.append({"label": label, "image_path": view_path})
                look_step["lookaround_views"] = view_paths
                self._write_success_step_direct(ep, look_step)
                eb_history.append(look_step)
                parent_id = _sid(config.branch_id, step_index)
                step_index += 1
                nonexecuted_retry_count = 0
                memory.update(metadata, metadata.get("objects", []),
                              "LookAround", True, None, task_criteria)
                for _d_label, _d_meta in look_metas_legacy[1:]:
                    memory.update(_d_meta, _d_meta.get("objects", []),
                                  "LookAround", True, None, task_criteria)
                self._record_trail(trail, metadata)
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

                if proposed_action == "Done":
                    task_complete_done, done_reason = check_task_complete(
                        metadata, ep.data, task_state
                    )
                    if task_complete_done:
                        self._write_success_step(
                            ep, config.branch_id, step_index, parent_id,
                            "Done", {},
                            {"success": True, "error": None, "frame": image, "metadata": metadata},
                            eb_reasoning=eb_reasoning,
                        )
                        return self._make_result(
                            config, "task_complete", step_index + 1,
                            fork_tasks, fork_source_ids, ep,
                        )
                    last_error = (
                        f"Done rejected: {done_reason}. Fix the unmet criteria before calling Done again."
                    )
                    continue

                if proposed_action != "MoveSequence":
                    standalone_error = self._validate_standalone_action(
                        proposed_action,
                        proposed_params,
                        metadata.get("agent", {}).get("cameraHorizon", 0.0),
                    )
                    if standalone_error:
                        nonexecuted_retry_count += 1
                        self._write_failure_log(failure_log_path, {
                            "step_index": step_index,
                            "branch_id": config.branch_id,
                            "failure_type": "model_invalid_action",
                            "proposed_action": proposed_action,
                            "proposed_params": proposed_params,
                            "eb_reasoning": eb_reasoning,
                            "error_message": standalone_error,
                            "retry_attempt": nonexecuted_retry_count,
                        })
                        if nonexecuted_retry_count >= max_nonexecuted_retries:
                            raise RuntimeError(standalone_error)
                        last_error = standalone_error
                        continue

            # ==========================================================
            # MoveSequence: execute a chain of movement steps sequentially.
            # Stops on first failure, reports what actually succeeded.
            # ==========================================================
            if proposed_action == "MoveSequence":
                # Post-LookAround output is produced after the pre-Phase-2
                # boundary, so defend this final dispatch boundary as well.
                sequence_error = self._validate_action_sequence(
                    proposed_params.get("steps") if isinstance(proposed_params, dict) else None,
                    "action sequence",
                )
                if sequence_error:
                    raise RuntimeError(sequence_error)
                horizon_error = self._validate_camera_horizon_sequence(
                    proposed_params["steps"],
                    metadata.get("agent", {}).get("cameraHorizon", 0.0),
                )
                if horizon_error:
                    raise RuntimeError(horizon_error)

                seq_result, seq_msg = self._execute_move_sequence(
                    proposed_params, env, metadata, failure_log_path,
                    config.branch_id, step_index, ep.episode_id,
                    eb_reasoning, injection_decision,
                )
                if seq_result.get("all_succeeded"):
                    # All steps succeeded — record as one successful step
                    cascade_level = 0
                    stuck_escalation_count = 0
                    last_error = None
                    step_entry = self._build_step_entry(
                        ep.episode_id, config.branch_id, step_index, parent_id,
                        "MoveSequence", proposed_params, seq_result, eb_reasoning, injection_decision,
                    )
                    memory.update(seq_result["metadata"], seq_result["metadata"].get("objects", []),
                                  "MoveSequence", True, None, task_criteria,
                                  f"{intent} {intent_target}".strip())
                    self._record_trail(trail, seq_result["metadata"])
                    self._track_searched_receptacles(proposed_params)
                    self._agentic_review(
                        ep, config, step_index, step_entry, env, seq_result["frame"],
                        seq_result["metadata"], eb_history, memory.render(),
                    )
                    self._write_success_step_direct(ep, step_entry)
                    eb_history.append(step_entry)
                    current_intent["steps"].append(step_entry)
                    step_index += 1
                    nonexecuted_retry_count = 0
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
                                  "MoveSequence", False, seq_msg, task_criteria,
                                  f"{intent} {intent_target}".strip())
                    pending_step = self._build_step_entry(
                        ep.episode_id, config.branch_id, step_index - 1, parent_id,
                        "MoveSequence", proposed_params, result, eb_reasoning, injection_decision,
                    )
                    pending_step["error_type"] = "environment_failure"
                    diagnosis_history = eb_history + [pending_step]
                    # Track failed objectIds from the sequence
                    obj_id = seq_result.get("failed_params", {}).get("objectId")
                    if obj_id:
                        failed_object_ids.add(obj_id)
                    if self.agentic:
                        agentic_decision = self.agentic.analyze_failure(
                            image=image,
                            task_goal=ep.data["task_goal"],
                            error_message=seq_msg,
                            action_history=diagnosis_history,
                            memory_text=memory.render(),
                        )
                        pending_step["agentic_oracle_failure"] = agentic_decision
                        self._write_success_step_direct(ep, pending_step)
                        eb_history.append(pending_step)
                        current_intent["steps"].append(pending_step)
                        hard_unrec = check_unrecoverable(metadata, ep.data)
                        if hard_unrec or detect_dead_loop(eb_history):
                            reason = f"unrecoverable_hard:{hard_unrec}" if hard_unrec else "dead_loop"
                            result_br = BranchResult(config.branch_id, reason, step_index, fork_tasks, fork_source_ids)
                            self._finalize(ep, config, result_br, fork_source_ids)
                            return result_br
                        if agentic_decision.get("action") == "terminate":
                            result_br = BranchResult(config.branch_id, "agentic_terminate", step_index, fork_tasks, fork_source_ids)
                            self._finalize(ep, config, result_br, fork_source_ids)
                            return result_br
                        if agentic_decision.get("action") == "fork" and self.enable_fork:
                            fork_task = self._build_fork_task(
                                config, step_index - 1, agentic_decision.get("fork_plan"),
                                {}, eb_history, ep,
                            )
                            if fork_task:
                                fork_tasks.append(fork_task)
                                fork_source_ids.append(_sid(config.branch_id, step_index - 1))
                        continue
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
                        current_intent=f"{current_intent.get('intent', '')} ({current_intent.get('target', '')})",
                    )
                    # C2: occlusion-aware unmark based on Phase-3 diagnosis
                    _check_occlusion_unmark(
                        eb_phase3.get("diagnosis", ""),
                        current_intent.get("target", ""),
                        memory,
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
                    current_intent["steps"].append(step_entry)

                    # ── StuckTracker escalation counter ──
                    stuck_summary = memory._stuck_tracker.render_summary(step_index) if hasattr(memory, '_stuck_tracker') else ""
                    if stuck_summary:
                        stuck_escalation_count += 1
                    if stuck_escalation_count >= 5:
                        result_br = BranchResult(branch_id=config.branch_id, termination_reason="permanent_stuck", total_steps=step_index, fork_tasks=fork_tasks, fork_source_step_ids=fork_source_ids)
                        self._finalize(ep, config, result_br, fork_source_ids)
                        return result_br

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

                    # ── Recovery execution ──
                    recovery_verdict = oracle_phase4.get("recovery_verdict", "")
                    if recovery_verdict == "recoverable":
                        rec_action = eb_phase3.get("proposed_recovery_action") or {}
                        rec_name = rec_action.get("action", "")
                        rec_params = rec_action.get("params", {})
                        recovery_error = self._validate_standalone_action(
                            rec_name,
                            rec_params,
                            metadata.get("agent", {}).get("cameraHorizon", 0.0),
                        )
                        if recovery_error:
                            self._write_failure_log(failure_log_path, {
                                "step_index": step_index,
                                "branch_id": config.branch_id,
                                "failure_type": "model_invalid_recovery_action",
                                "proposed_recovery": rec_action,
                                "error_message": recovery_error,
                            })
                            raise RuntimeError(recovery_error)

                        resolved, warn = resolve_object_ids(
                            rec_name, rec_params, metadata.get("objects", [])
                        )
                        if warn and rec_name in _OBJECT_ACTIONS:
                            recovery_error = "Your recovery action was not executed. " + warn
                            self._write_failure_log(failure_log_path, {
                                "step_index": step_index,
                                "branch_id": config.branch_id,
                                "failure_type": "model_unresolved_recovery_object",
                                "proposed_recovery": rec_action,
                                "error_message": recovery_error,
                            })
                            raise RuntimeError(recovery_error)

                        rec_act, rec_adapt_params = adapt(rec_name, resolved)
                        rec_result = env.step(rec_act, **rec_adapt_params)
                        if rec_result.get("frame") is None:
                            rec_result["frame"] = image
                        rec_metadata = rec_result.get("metadata", metadata)
                        memory.update(
                            rec_metadata, rec_metadata.get("objects", []), rec_act,
                            rec_result["success"], rec_result.get("error"), task_criteria,
                            f"{intent} {intent_target}".strip(),
                        )
                        rec_step = self._build_step_entry(
                            ep.episode_id, config.branch_id, step_index, parent_id,
                            rec_act, rec_adapt_params, rec_result,
                            f"[recovery] {eb_phase3.get('recovery_reasoning', '')}",
                            injection_decision,
                        )
                        rec_step["recovery_step"] = True
                        rec_step["eb_diagnosis"] = eb_phase3.get("diagnosis")
                        self._write_success_step_direct(ep, rec_step)
                        eb_history.append(rec_step)
                        current_intent["steps"].append(rec_step)
                        step_index += 1
                        if rec_result["success"]:
                            cascade_level = 0
                            last_error = None
                            image = rec_result["frame"]
                            metadata = rec_metadata
                            self._record_trail(trail, metadata)
                        else:
                            cascade_level += 1
                            last_error = f"Recovery {rec_name} failed: {rec_result.get('error', '')}"
                            image = rec_result.get("frame", image)
                            metadata = rec_metadata
                    continue

            # ==========================================================
            # 执行 EB 动作（objectType → objectId 解析 + 适配）
            # ==========================================================
            resolved, resolve_warning = resolve_object_ids(proposed_action, proposed_params,
                                                           metadata.get("objects", []))
            act, params = adapt(proposed_action, resolved)
            parent_id = _last_branch_step_id(ep, config.branch_id)

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
                stuck_escalation_count = 0
                last_error = None
                step_entry = self._build_step_entry(
                    ep.episode_id, config.branch_id, step_index, parent_id,
                    act, params, result, eb_reasoning, injection_decision,
                )
                memory.update(result["metadata"], result["metadata"].get("objects", []), act, True, None, task_criteria, f"{intent} {intent_target}".strip())
                self._record_trail(trail, result["metadata"])
                self._agentic_review(
                    ep, config, step_index, step_entry, env, result["frame"],
                    result["metadata"], eb_history, memory.render(),
                )
                self._write_success_step_direct(ep, step_entry)
                eb_history.append(step_entry)
                current_intent["steps"].append(step_entry)
                step_index += 1
                nonexecuted_retry_count = 0
                if act == "OpenObject" and params.get("objectId"):
                    for obj in result["metadata"].get("objects", []):
                        if obj.get("objectId") == params["objectId"]:
                            self.critic_guard.mark_receptacle_searched(obj.get("objectType", ""))
                            break
                continue

            # --- 环境失败 ---
            step_index += 1
            nonexecuted_retry_count = 0

            # --- 环境失败 → Phase 3 + Phase 4 (写 JSON) ---
            cascade_level += 1
            last_error = result["error"]
            memory.update(result.get("metadata", metadata), result.get("metadata", metadata).get("objects", []), act, False, result["error"], task_criteria, f"{intent} {intent_target}".strip())

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

            if self.agentic:
                agentic_decision = self.agentic.analyze_failure(
                    image=result["frame"],
                    task_goal=ep.data["task_goal"],
                    error_message=result["error"] or "Unknown error",
                    action_history=diagnosis_history,
                    memory_text=memory.render(),
                )
                pending_step["agentic_oracle_failure"] = agentic_decision
                self._write_success_step_direct(ep, pending_step)
                eb_history.append(pending_step)
                current_intent["steps"].append(pending_step)
                hard_unrec = check_unrecoverable(metadata, ep.data)
                if hard_unrec or detect_dead_loop(eb_history):
                    reason = f"unrecoverable_hard:{hard_unrec}" if hard_unrec else "dead_loop"
                    result_br = BranchResult(config.branch_id, reason, step_index, fork_tasks, fork_source_ids)
                    self._finalize(ep, config, result_br, fork_source_ids)
                    return result_br
                if agentic_decision.get("action") == "terminate":
                    result_br = BranchResult(config.branch_id, "agentic_terminate", step_index, fork_tasks, fork_source_ids)
                    self._finalize(ep, config, result_br, fork_source_ids)
                    return result_br
                if agentic_decision.get("action") == "fork" and self.enable_fork:
                    fork_task = self._build_fork_task(
                        config, step_index - 1, agentic_decision.get("fork_plan"),
                        {}, eb_history, ep,
                    )
                    if fork_task:
                        fork_tasks.append(fork_task)
                        fork_source_ids.append(_sid(config.branch_id, step_index - 1))
                continue

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
                current_intent=f"{current_intent.get('intent', '')} ({current_intent.get('target', '')})",
            )

            # C2: occlusion-aware unmark based on Phase-3 diagnosis
            _check_occlusion_unmark(
                eb_phase3.get("diagnosis", ""),
                current_intent.get("target", ""),
                memory,
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
            current_intent["steps"].append(step_entry)

            # ── StuckTracker escalation counter ──
            stuck_summary = memory._stuck_tracker.render_summary(step_index) if hasattr(memory, '_stuck_tracker') else ""
            if stuck_summary:
                stuck_escalation_count += 1
            if stuck_escalation_count >= 5:
                result_br = BranchResult(
                    branch_id=config.branch_id, termination_reason="permanent_stuck",
                    total_steps=step_index, fork_tasks=fork_tasks,
                    fork_source_step_ids=fork_source_ids,
                )
                self._finalize(ep, config, result_br, fork_source_ids)
                return result_br

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

            if self.enable_fork and oracle_phase4.get("should_fork"):
                grade = oracle_phase4.get("counterfactual_grade", "WA")
                gold = oracle_phase4.get("counterfactual_gold")
                # AC: the agent's own counterfactual was correct → fork it.
                # PA/WA: the agent was wrong/partial → fork the Oracle's corrected
                #        gold instead, so the branch tests a verified alternative.
                if grade == "AC" and eb_phase3.get("counterfactual"):
                    cf_source = eb_phase3.get("counterfactual")
                elif isinstance(gold, dict) and gold.get("target_step") is not None:
                    cf_source = gold
                else:
                    cf_source = None
                if cf_source is not None:
                    fork_task = self._build_fork_task(
                        config=config, current_step_idx=step_index - 1,
                        counterfactual=cf_source,
                        fallback_recovery=eb_phase3.get("proposed_recovery_action", {}),
                        history=eb_history, ep=ep,
                    )
                    if fork_task:
                        fork_tasks.append(fork_task)
                        fork_source_ids.append(_sid(config.branch_id, step_index - 1))

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
        executor_agent: "ExecutorAgent" = None,
        output_dir: str = "",
        step_limit_multiplier: int = 4,
        enable_fork: bool = False,
        enable_phase2: bool = True,
        memory_mode: str = "semantic",
        agentic_oracle: bool = False,
    ) -> BranchResult:
        ep = EpisodeManager.load(episode_path)
        branch_steps = ep.get_steps_for_branch(branch_id)
        if not branch_steps:
            raise ValueError(f"Branch '{branch_id}' has no steps in {episode_path}")

        scene = ep.data["scene"]
        env = EnvController(scene=scene)
        env.reset_to_alfred_scene(_require_alfred_scene(ep.data))
        replay_steps(env, branch_steps, skip_failed=True)

        # ── Restore inventory: if the episode says the agent was holding
        #     an object, try to pick it up in the resumed environment (replay
        #     may have silently skipped the PickupObject step).
        _s = env.get_state_snapshot()
        _current_held = (_s["metadata"].get("inventoryObjects") or [])
        if not _current_held:
            _should_hold: str | None = None
            _hold_actions = {"PickupObject"}
            _drop_actions = {"PutObject", "DropHandObject"}
            for _step in branch_steps:
                _act = _step.get("action", "")
                _ok = _step.get("success", False)
                if not _ok:
                    continue
                if _act == "MoveSequence":
                    for _sub in _step.get("action_params", {}).get("steps", []):
                        _sa = _sub.get("action", "")
                        if _sa in _hold_actions:
                            _should_hold = _sub.get("params", {}).get("objectType")
                        elif _sa in _drop_actions:
                            _should_hold = None
                elif _act in _hold_actions:
                    _should_hold = _step.get("action_params", {}).get("objectType")
                elif _act in _drop_actions:
                    _should_hold = None
            if _should_hold:
                from src.action_adapter import resolve_object_ids, adapt
                _objs = _s["metadata"].get("objects", [])
                _resolved, _w = resolve_object_ids("PickupObject", {"objectType": _should_hold}, _objs)
                if _w:
                    logging.warning(
                        "resume: cannot restore held object %s (not found in scene): %s",
                        _should_hold, _w,
                    )
                else:
                    _a, _p = adapt("PickupObject", _resolved)
                    _r = env.step(_a, **_p)
                    if _r["success"]:
                        logging.info("resume: restored held object %s", _should_hold)
                    else:
                        logging.warning(
                            "resume: failed to restore held object %s: %s",
                            _should_hold, _r.get("error", "unknown"),
                        )

        last_step = branch_steps[-1]
        start_idx = last_step["step_index_in_branch"] + 1
        config = BranchConfig(
            episode_id=ep.episode_id,
            branch_id=branch_id,
            parent_branch_id=None,
        )
        if not output_dir:
            output_dir = os.path.dirname(episode_path)
        runner = cls(eb_agent, oracle_agent, output_dir,
                     step_limit_multiplier=step_limit_multiplier,
                     enable_fork=enable_fork,
                     enable_phase2=enable_phase2,
                     memory_mode=memory_mode,
                     agentic_oracle=agentic_oracle)
        runner.executor_agent = executor_agent
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
            step_id=_sid(branch_id, step_idx), branch_id=branch_id, parent_step_id=parent_id,
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

    @classmethod
    def _validate_executor_result(cls, steps, status, source: str) -> str | None:
        """Validate Executor status and actions before they become MoveSequence."""
        if status not in {"done", "partial", "failed"}:
            return (
                f"Your Executor response was not executed because status {status!r} is invalid. "
                "Use one of: done, partial, failed."
            )
        return cls._validate_action_sequence(steps, source)

    @staticmethod
    def _validate_executor_output_shape(actions, source: str) -> str | None:
        """Reject malformed raw Executor output before Planner review."""
        if not isinstance(actions, list):
            return (
                f"Your {source} response was not executed because actions must be a JSON list. "
                "Re-think from the current image and provide concrete actions."
            )
        return None

    @staticmethod
    def _action_params(step: dict) -> dict:
        """Read nested action params while retaining legacy flat-field output."""
        nested = step.get("params")
        if nested is not None:
            return nested if isinstance(nested, dict) else nested
        return {k: v for k, v in step.items() if k not in ("action", "repeat")}

    @classmethod
    def _validate_action_contract(cls, action, params, source: str) -> str | None:
        """Validate one concrete action before resolution, adaptation, or stepping."""
        if action not in _EXECUTABLE_SEQUENCE_ACTIONS:
            return (
                f"Your {source} was not executed because action {action!r} is invalid. "
                "Use a concrete AI2-THOR action, not Done, LookAround, or MoveSequence."
            )
        if not isinstance(params, dict):
            return f"Your {source} was not executed because params must be a JSON object."
        if action in _OBJECT_ACTIONS:
            object_type = params.get("objectType")
            if not isinstance(object_type, str) or not object_type.strip():
                return (
                    f"Your {source} was not executed because {action} requires a non-empty "
                    "objectType."
                )
            if action == "PutObject":
                receptacle_type = params.get("receptacleType")
                if not isinstance(receptacle_type, str) or not receptacle_type.strip():
                    return (
                        f"Your {source} was not executed because PutObject requires a non-empty "
                        "receptacleType."
                    )
        return None

    @classmethod
    def _validate_standalone_action(
        cls, action, params, camera_horizon, *, allow_meta: bool = False
    ) -> str | None:
        """Validate a legacy, recovery, or post-scan standalone action."""
        if allow_meta and action in _META_ACTIONS:
            return None if isinstance(params, dict) else "Your action params must be a JSON object."
        error = cls._validate_action_contract(action, params, "action")
        if error:
            return error
        if action in {"LookUp", "LookDown"}:
            return cls._validate_camera_horizon_sequence(
                [{"action": action}], camera_horizon
            )
        return None

    @classmethod
    def _validate_action_sequence(cls, steps, source: str) -> str | None:
        """Return an actionable error when a proposed sequence cannot run.

        Invalid sequences are model-output errors, not environment failures, so
        callers must use the nonexecuted retry path rather than Phase 3.
        """
        if not isinstance(steps, list) or not steps:
            return (
                f"Your {source} action sequence was not executed because it must be "
                "a non-empty JSON list of concrete actions. Re-think from the current "
                "image and provide at least one executable action."
            )
        if len(steps) > _MAX_SEQUENCE_ACTIONS:
            return (
                f"Your {source} action sequence was not executed because it has {len(steps)} "
                f"actions; the maximum is {_MAX_SEQUENCE_ACTIONS}. Use repeat for movement."
            )
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                return (
                    f"Your {source} action sequence was not executed because step {index} "
                    "must be a JSON object with an action field."
                )
            action = step.get("action")
            params = cls._action_params(step)
            action_error = cls._validate_action_contract(
                action, params, f"{source} action sequence step {index}"
            )
            if action_error:
                return action_error
            if "repeat" in step:
                repeat = step["repeat"]
                if action not in _REPEATABLE_ACTIONS:
                    return (
                        f"Your {source} action sequence was not executed because step {index} "
                        f"uses repeat with unsupported action {action!r}. Repeat is only valid "
                        "for MoveAhead, MoveBack, MoveLeft, and MoveRight."
                    )
                if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
                    return (
                        f"Your {source} action sequence was not executed because step {index} "
                        "has repeat that is not a positive integer."
                    )
                if repeat > _MAX_ACTION_REPEAT:
                    return (
                        f"Your {source} action sequence was not executed because step {index} "
                        f"has repeat={repeat}; the maximum is {_MAX_ACTION_REPEAT}."
                    )
        return None

    @staticmethod
    def _normalize_camera_horizon(camera_horizon) -> float | None:
        """Normalize the 330° encoding AI2-THOR also uses for -30° up."""
        if isinstance(camera_horizon, bool):
            return None
        try:
            horizon = float(camera_horizon)
        except (TypeError, ValueError):
            return None
        return horizon - 360.0 if horizon > 180.0 else horizon

    @classmethod
    def _validate_camera_horizon_sequence(cls, steps: list[dict], camera_horizon) -> str | None:
        """Reject LookUp/LookDown steps outside AI2-THOR's physical limits."""
        horizon = cls._normalize_camera_horizon(camera_horizon)
        if horizon is None or not (_CAMERA_HORIZON_MIN - _CAMERA_HORIZON_EPSILON) <= horizon <= (_CAMERA_HORIZON_MAX + _CAMERA_HORIZON_EPSILON):
            return (
                "Your action sequence was not executed because the current camera horizon "
                f"{camera_horizon!r} is outside AI2-THOR's supported -30° to +60° range."
            )
        for index, step in enumerate(steps, start=1):
            action = step["action"]
            repeat = step.get("repeat", 1)
            if action == "LookUp":
                horizon -= _CAMERA_LOOK_STEP_DEGREES * repeat
            elif action == "LookDown":
                horizon += _CAMERA_LOOK_STEP_DEGREES * repeat
            if not (_CAMERA_HORIZON_MIN - _CAMERA_HORIZON_EPSILON) <= horizon <= (_CAMERA_HORIZON_MAX + _CAMERA_HORIZON_EPSILON):
                return (
                    f"Your action sequence was not executed because {action} at step {index} "
                    f"would move cameraHorizon to {horizon:.0f}°, outside AI2-THOR's "
                    "supported -30° (up) to +60° (down) range."
                )
        return None

    def _track_searched_receptacles(self, proposed_params: dict):
        """Track receptacles opened/searched in a MoveSequence for CriticGuard."""
        if not self.enable_critic:
            return
        steps = proposed_params.get("steps", [])
        for step in steps:
            action = step.get("action", "")
            if action == "OpenObject":
                sp = step.get("params", {})
                obj_type = sp.get("objectType", "")
                if obj_type:
                    self.critic_guard.mark_receptacle_searched(obj_type)

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
        _META_ACTIONS = {"Done", "LookAround"}  # handled by branch_runner / Planner, never AI2-THOR

        executed = []
        final_frame = None
        final_metadata = None
        all_succeeded = True

        for i, step in enumerate(steps):
            action = step.get("action", "")

            # Meta-actions (Done, LookAround) belong to Planner / branch_runner,
            # not Executor. Executor does NOT know these actions — if one appears,
            # it's a hallucination. Reject the entire MoveSequence as failure.
            if action in _META_ACTIONS:
                executed_desc = (" → ".join(executed)) if executed else "(nothing executed)"
                msg = (f"MoveSequence FAILED: hallucinated meta-action '{action}' "
                       f"at step {i+1}/{len(steps)}. Executed before error: {executed_desc}. "
                       f"Do NOT output {action} — it is not a valid action.")
                return (
                    {"success": False, "all_succeeded": False,
                     "error": msg,
                     "executed": executed, "frame": final_frame, "metadata": final_metadata,
                     "failed_params": step.get("params", {})},
                    msg,
                )

            # Executor outputs {"action": "PickupObject", "params": {"objectType": "AlarmClock"}}
            if "params" in step and isinstance(step["params"], dict):
                step_params = dict(step["params"])
            else:
                step_params = {k: v for k, v in step.items() if k not in ("action", "repeat")}

            # Resolve objectType → objectId for object-interaction actions
            if action not in _MOVEMENT_ACTIONS:
                if action not in _VALID_ACTIONS:
                    desc = " → ".join(executed) if executed else "(nothing)"
                    msg = f"MoveSequence: {desc} succeeded, then invalid action '{action}' at step {i+1}."
                    return (
                        {"success": False, "error": msg, "partial": True,
                         "executed": executed, "frame": final_frame, "metadata": final_metadata,
                         "failed_params": step_params},
                        msg,
                    )
                current_objs = (final_metadata or metadata).get("objects", [])
                resolved, warn = resolve_object_ids(action, step_params, current_objs)
                if warn and action in _OBJECT_ACTIONS:
                    desc = " → ".join(executed) if executed else "(nothing)"
                    msg = f"MoveSequence partially executed: {desc}, then {action} failed — {warn}"
                    return (
                        {"success": False, "error": msg, "partial": True,
                         "executed": executed, "frame": final_frame, "metadata": final_metadata,
                         "failed_params": resolved},
                        msg,
                    )
                act, params = adapt(action, resolved)
            elif action in _MOVEMENT_ACTIONS:
                act = action
                params = {}
            else:
                desc = " → ".join(executed) if executed else "(nothing)"
                msg = f"MoveSequence: {desc} succeeded, then unknown action '{action}' at step {i+1}."
                return (
                    {"success": False, "error": msg, "partial": True,
                     "executed": executed, "frame": final_frame, "metadata": final_metadata,
                     "failed_params": step_params},
                    msg,
                )

            repeat = step.get("repeat", 1)

            succeeded = 0
            for r in range(repeat):
                result = env.step(act, **params)
                if result["success"]:
                    succeeded += 1
                    final_frame = result["frame"]
                    final_metadata = result["metadata"]
                else:
                    all_succeeded = False
                    # Hard rule: if agent position is unchanged, it's definitively blocked.
                    pre_pos = (final_metadata or metadata).get("agent", {}).get("position")
                    post_pos = result.get("metadata", {}).get("agent", {}).get("position")
                    if pre_pos and post_pos:
                        dx = abs(pre_pos.get("x", 0) - post_pos.get("x", 0))
                        dz = abs(pre_pos.get("z", 0) - post_pos.get("z", 0))
                        if dx < 0.001 and dz < 0.001:
                            result["error"] = (result.get("error", "")
                                + " [CONFIRMED BLOCKED: position unchanged]")
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
                             "failed_params": params,
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
                             "failed_params": params,
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

        raise RuntimeError(
            "MoveSequence: reached unreachable code after step loop — logic bug"
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

    def _build_fork_task(self, config, current_step_idx, counterfactual, fallback_recovery, history, ep):
        cf = counterfactual
        if not cf:
            return None

        # 结构化 counterfactual（新格式）— EB Phase-3 或 Oracle gold 同构
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
            alt_action = fallback_recovery or {}

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
                "origin_step_id": _sid(config.branch_id, current_step_idx),
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
            # Include dedup statistics in episode outcome for analysis
            outcome["dedup_stats"] = dict(self.eb_agent.dedup_stats)
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
# Searched-markers / occlusion-unmark helpers
# ======================================================================


# C2: Phrases in Phase 3 diagnosis that suggest the target might be occluded
# inside a previously-searched receptacle. When detected, the searched flag
# on that receptacle is cleared so the agent can re-check it.
_OCCLUSION_DIAGNOSIS_PHRASES: tuple[str, ...] = (
    "might be occluded", "could be behind", "possibly missed",
    "may be hidden", "might be inside", "could be blocked",
    "not visible but might be", "possibly inside",
)


def _check_occlusion_unmark(
    diagnosis: str,
    target: str,
    memory: "EgocentricMemory",
) -> bool:
    """If the Phase-3 diagnosis suggests the target might be occluded inside a
    previously-searched receptacle, unmark that receptacle so it can be
    re-checked. Returns True if any entry was unmarked."""
    if not diagnosis or not target:
        return False
    diag_lower = diagnosis.lower()
    if any(phrase in diag_lower for phrase in _OCCLUSION_DIAGNOSIS_PHRASES):
        count = memory.unmark_searched(object_type=target)
        if count > 0:
            import logging
            logging.info(
                "C2 occlusion-unmark: unmarked %d instance(s) of '%s' "
                "based on diagnosis",
                count, target,
            )
            return True
    return False


def _check_trail_revisit(
    trail: "SearchTrail",
    target_type: str,
    visible_objects: list[dict],
) -> float:
    """Check how many times the grid cell around a receptacle type has been visited.

    Args:
        trail: SearchTrail instance.
        target_type: objectType to look up (e.g. "Fridge").
        visible_objects: current frame's objects with position data.

    Returns:
        Max revisit score among all objects of that type (0.0 = never visited).
    """
    max_score = 0.0
    for obj in visible_objects:
        if obj.get("objectType") == target_type:
            pos = obj.get("position", {})
            score = trail.revisit_score(pos.get("x", 0), pos.get("z", 0))
            if score > max_score:
                max_score = score
    return max_score


def _extract_object_id_from_intent(completed_intent: dict) -> str | None:
    """Try to extract a specific objectId from a completed intent's step history.

    For single-action steps (non-MoveSequence), action_params carries the
    resolved objectId.  For MoveSequence, sub-step params may carry objectType
    (original Executor output, not yet resolved) — in that case we return None
    so the caller can fall back to type-level marking.
    """
    for step in completed_intent.get("steps", []):
        action = step.get("action", "")
        if action == "MoveSequence":
            for sub in step.get("action_params", {}).get("steps", []):
                if sub.get("action") in ("OpenObject", "CloseObject"):
                    oid = sub.get("params", {}).get("objectId")
                    if oid:
                        return oid
        elif action in ("OpenObject", "CloseObject"):
            oid = step.get("action_params", {}).get("objectId")
            if oid:
                return oid
    return None


def _check_and_mark_searched(
    completed_intent: dict,
    new_target: str,
    memory: "EgocentricMemory",
    enable_curiosity: bool = True,
) -> None:
    """Mark receptacles as searched when the Planner moves on.

    Called at intent transition. Triggers when:
    - Old intent had a target (old_target != "").
    - New intent targets something different (old_target != new_target).
    - The intent involved receptacle interaction (OpenObject/CloseObject)
      OR the intent phrase suggests approach/check/open/close/search behaviour
      AND the last step succeeded (prevents marking on failures).

    C1 fix: resolves the specific objectId from the intent's step history
    before marking, so that only the interacted-with instance is marked,
    not every instance of that type in the scene.
    """
    import logging

    old_target = completed_intent.get("target", "")
    old_intent = completed_intent.get("intent", "")
    completed_ok = completed_intent.get("completed", False)

    if not old_target:
        return
    if old_target == new_target:
        return

    steps = completed_intent.get("steps", [])

    # ----- detect explicit receptacle interaction -----
    had_receptacle_interaction = False
    for step in steps:
        action = step.get("action", "")
        if action == "MoveSequence":
            for sub in step.get("action_params", {}).get("steps", []):
                if sub.get("action") in ("OpenObject", "CloseObject"):
                    had_receptacle_interaction = True
                    break
        elif action in ("OpenObject", "CloseObject"):
            had_receptacle_interaction = True
        if had_receptacle_interaction:
            break

    # ----- detect approach / already-open patterns -----
    _RECEPTACLE_INTENT_KW = ("approach", "open", "close", "check", "search", "look")
    is_receptacle_intent = any(
        kw in old_intent.lower() for kw in _RECEPTACLE_INTENT_KW
    )

    if had_receptacle_interaction or (is_receptacle_intent and completed_ok):
        # C1: resolve to specific objectId when possible
        specific_id = _extract_object_id_from_intent(completed_intent)
        if specific_id:
            count = memory.mark_searched(object_id=specific_id)
            if count > 0:
                logging.debug(
                    "marked objectId '%s' as SEARCHED "
                    "(intent='%s' completed=%s interaction=%s)",
                    specific_id, old_intent, completed_ok, had_receptacle_interaction,
                )
        else:
            count = memory.mark_searched(object_type=old_target)
            if count > 0:
                logging.warning(
                    "C1 fallback: no specific objectId found for intent '%s' "
                    "target='%s' — marked %d instance(s) by objectType only",
                    old_intent, old_target, count,
                )

        # ── Spike 003: Curiosity Scoreboard visitation tracking ──
        if enable_curiosity:
            memory.record_receptacle_visit(old_target)
            if had_receptacle_interaction:
                memory.record_receptacle_open(old_target)


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
    Replay steps to restore agent state for resume / fork init.
    Handles MoveSequence, LookAround, and single actions correctly.
    MoveSequence steps are expanded and replayed individually; execution stops
    on first failure (matching the original MoveSequence behavior).
    Failed steps are skipped when skip_failed=True.
    """
    from src.action_adapter import resolve_object_ids, adapt

    _MOVEMENT = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
                 "RotateLeft", "RotateRight", "LookUp", "LookDown"}

    for s in steps:
        action = s["action"]
        params = s.get("action_params", {})

        if action in ("Pass", "Done"):
            continue

        # LookAround: replay the 4 rotation sequence that captured views
        if action == "LookAround":
            for _ in range(4):
                r = env.step("RotateLeft")
                if not r["success"]:
                    raise RuntimeError(
                        f"Replay LookAround RotateLeft failed at {s.get('step_id')}: {r['error']}"
                    )
            continue

        # MoveSequence: expand and replay individual steps
        if action == "MoveSequence":
            objects = env.controller.last_event.metadata.get("objects", [])
            seq_steps = params.get("steps", [])
            for st in seq_steps:
                a = st.get("action", "")
                sp = dict(st.get("params", {}))
                if a in _META_ACTIONS:
                    break  # meta-action in sequence → stop, matches _execute_move_sequence
                if a not in _MOVEMENT:
                    resolved, warn = resolve_object_ids(a, sp, objects)
                    if warn:
                        logging.warning(
                            "replay: MoveSequence step %s (%s) objectId resolution failed at %s: %s",
                            a, sp.get("objectType", "?"), s.get("step_id"), warn,
                        )
                        break  # object not found → stop, matches original behavior
                    a, sp = adapt(a, resolved)
                r = env.step(a, **sp)
                if not r["success"]:
                    if skip_failed:
                        logging.warning(
                            "replay: MoveSequence step %s skipped at %s: %s",
                            a, s.get("step_id"), r.get("error", "unknown"),
                        )
                        break  # partial execution accepted, stop here
                    raise RuntimeError(
                        f"Replay MoveSequence step {a} failed at {s.get('step_id')}: {r['error']}"
                    )
            continue

        # Single actions (old single-agent path)
        if action in _MOVEMENT:
            act, p = action, {}
        elif action in _META_ACTIONS:
            continue
        else:
            objects = env.controller.last_event.metadata.get("objects", [])
            resolved, warn = resolve_object_ids(action, params, objects)
            if warn and skip_failed:
                logging.warning(
                    "replay: single action %s (%s) objectId resolution failed at %s: %s",
                    action, params.get("objectType", "?"), s.get("step_id"), warn,
                )
                continue
            act, p = adapt(action, resolved)

        r = env.step(act, **p)
        if not r["success"]:
            if skip_failed:
                logging.warning(
                    "replay: single action %s skipped at %s: %s",
                    action, s.get("step_id"), r.get("error", "unknown"),
                )
                continue
            raise RuntimeError(
                f"Replay failed at {s.get('step_id')}: {action}({s.get('action_params', {})}) "
                f"-> {r['error']}"
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
    enable_critic: bool = False,
    critic_llm: bool = False,
    enable_searched_markers: bool = True,
    enable_intent_dedup: bool = True,
    enable_critic_guard: bool = False,
    enable_curiosity_scoreboard: bool = True,
    enable_contrastive_planner: bool = True,
    enable_progress_gating: bool = True,
    enable_search_trail: bool = True,
    memory_mode: str = "semantic",  # "semantic" | "geometric"
    episode_status: str = "pending",
    agentic_oracle: bool = False,
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
        if episode_status != "pending":
            ep.set_status(episode_status, os.getpid() if episode_status == "running" else None)

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
                              executor_agent=executor_agent,
                              enable_critic=enable_critic,
                              critic_llm=critic_llm,
                              enable_searched_markers=enable_searched_markers,
                              enable_intent_dedup=enable_intent_dedup,
                              enable_critic_guard=enable_critic_guard,
                              enable_curiosity_scoreboard=enable_curiosity_scoreboard,
                              enable_contrastive_planner=enable_contrastive_planner,
                              enable_progress_gating=enable_progress_gating,
                              enable_search_trail=enable_search_trail,
                              memory_mode=memory_mode,
                              agentic_oracle=agentic_oracle)

        config = BranchConfig(
            episode_id=episode_id,
            branch_id="main",
            parent_branch_id=None,
        )
        return runner.run(config=config, env=env, ep=ep)
    finally:
        env.close()
