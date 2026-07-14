"""
全局调度器。维护跨 episode 分支队列，委托 BranchRunner 执行 Phase 1-4 循环。
"""

import os
import glob
import threading
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import OracleAgent
from src.executor import ExecutorAgent
from src.branch_runner import (
    BranchRunner, BranchConfig, BranchResult,
    run_single_branch, replay_steps,
)
from src.fork_manager import ForkManager
from src.env_controller import EnvController
from src.episode_manager import EpisodeManager
from src.alfred_parser import load_traj, extract_metadata, extract_low_actions
from src.step_recorder import StepRecorder


@dataclass
class SchedulerConfig:
    data_dir: str = "data/json_2.1.0"
    output_dir: str = "output"
    max_episodes: int = 0
    max_parallel: int = 1
    task_filter: str = ""
    splits: str = "train,valid_seen,valid_unseen"
    api_key: str = ""
    api_base_url: str = "https://api.fullcupai.com"
    planner_model: str = "gpt-5.5"
    executor_model: str = "gpt-5.5"
    oracle_model: str = "gpt-5.5"
    planner_reasoning_effort: str = "xhigh"
    executor_reasoning_effort: str = "medium"
    oracle_reasoning_effort: str = "xhigh"
    enable_fork: bool = True


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        planner_client = VLMClient.openai(
            config.planner_model, config.api_key,
            base_url=config.api_base_url,
            reasoning_effort=config.planner_reasoning_effort,
        )
        executor_client = VLMClient.openai(
            config.executor_model, config.api_key,
            base_url=config.api_base_url,
            reasoning_effort=config.executor_reasoning_effort,
        )
        oracle_client = VLMClient.openai(
            config.oracle_model, config.api_key,
            base_url=config.api_base_url,
            reasoning_effort=config.oracle_reasoning_effort,
        )
        self.eb_agent = EBAgent(planner_client)
        self.oracle_agent = OracleAgent(oracle_client)
        self.executor_agent = ExecutorAgent(executor_client)
        # Warmup: GPT-5.5 首次调用需加载，避免首次 Phase 超时
        import time as _time
        _t0 = _time.time()
        oracle_client.chat_text(system_prompt="Say OK.", user_text="OK", max_tokens=5)
        print(f"Oracle warmup: {_time.time() - _t0:.1f}s")
        self.branch_runner = BranchRunner(
            self.eb_agent, self.oracle_agent, config.output_dir,
            enable_fork=config.enable_fork,
            executor_agent=self.executor_agent,
        )
        self.fork_manager = ForkManager(
            self.eb_agent, self.oracle_agent, self.branch_runner, config.output_dir,
        )
        self.queue: deque[dict] = deque()
        self.fork_queue: deque[dict] = deque()
        self.stats = {"completed": 0, "skipped": 0, "failed": 0}
        self.recorder = StepRecorder()

    # ------------------------------------------------------------------
    # 任务加载
    # ------------------------------------------------------------------
    def load_tasks(self):
        splits = [s.strip() for s in self.config.splits.split(",")]
        all_files = []
        for split in splits:
            pattern = os.path.join(self.config.data_dir, split, "**", "traj_data.json")
            all_files.extend(sorted(glob.glob(pattern, recursive=True)))

        if self.config.task_filter:
            filtered = []
            for f in all_files:
                t = load_traj(f)
                if t.get("task_type") == self.config.task_filter:
                    filtered.append(f)
            all_files = filtered

        if self.config.max_episodes > 0:
            all_files = all_files[: self.config.max_episodes]

        for f in all_files:
            traj = load_traj(f)
            meta = extract_metadata(traj)
            episode_id = os.path.basename(os.path.dirname(f))
            low_actions = extract_low_actions(traj)
            base_step_count = len(low_actions)

            self.queue.append({
                "traj_path": f,
                "episode_id": episode_id,
                "meta": meta,
                "base_step_count": base_step_count,
                "branch_config": BranchConfig(
                    episode_id=episode_id,
                    branch_id="main",
                    parent_branch_id=None,
                ),
            })

        print(f"Loaded {len(self.queue)} main branch tasks.")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self):
        self.load_tasks()
        total = len(self.queue)
        max_workers = max(1, self.config.max_parallel)
        self._lock = threading.Lock()

        # Phase 1: parallel main branches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, task in enumerate(self.queue):
                futures[executor.submit(self._run_branch_worker, task, i + 1, total)] = task

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = BranchResult(
                        branch_id=task["branch_config"].branch_id,
                        termination_reason=f"worker_crash:{e}",
                        total_steps=0, fork_tasks=[],
                    )
                    with self._lock:
                        self.stats["failed"] += 1
                if result:
                    with self._lock:
                        self.stats["completed"] += 1
                        for ft in result.fork_tasks:
                            self.fork_queue.append(self._fork_entry(task, ft))
                        self._report(total)

        # Phase 2: serial fork branches (depends on parent completion)
        if self.fork_queue:
            print(f"\nProcessing {len(self.fork_queue)} fork branches...")
        while self.fork_queue:
            task = self.fork_queue.popleft()
            result = self._run_fork(task)
            self.stats["completed"] += 1
            for ft in result.fork_tasks:
                self.fork_queue.append(self._fork_entry(task, ft))
            self._report(total)

        print(f"\n\nDone. {self.stats['completed']} branches, {self.stats['failed']} failed.")

    def _run_branch_worker(self, task: dict, n: int, total: int) -> BranchResult:
        """Thread-safe wrapper around _run_branch."""
        ep_id = task["episode_id"]
        out_file = os.path.join(self.config.output_dir, f"{ep_id}.json")
        if os.path.exists(out_file):
            with self._lock:
                self.stats["skipped"] += 1
            return BranchResult(branch_id="main", termination_reason="skipped",
                                total_steps=0, fork_tasks=[], fork_source_step_ids=[])
        print(f"\n[{n}/{total}] Starting {ep_id}")
        result = run_single_branch(
            traj_path=task["traj_path"],
            eb_agent=self.eb_agent,
            oracle_agent=self.oracle_agent,
            output_dir=self.config.output_dir,
            enable_fork=self.config.enable_fork,
            executor_agent=self.executor_agent,
        )
        print(f"[{n}/{total}] Finished {ep_id}: {result.termination_reason} ({result.total_steps} steps)")
        return result

    # ------------------------------------------------------------------
    # Main 分支 → 委托 run_single_branch
    # ------------------------------------------------------------------
    def _run_branch(self, task: dict) -> BranchResult:
        ep_id = task["episode_id"]
        out_file = os.path.join(self.config.output_dir, f"{ep_id}.json")
        if os.path.exists(out_file):
            return BranchResult(branch_id="main", termination_reason="skipped",
                                total_steps=0, fork_tasks=[], fork_source_step_ids=[])

        return run_single_branch(
            traj_path=task["traj_path"],
            eb_agent=self.eb_agent,
            oracle_agent=self.oracle_agent,
            output_dir=self.config.output_dir,
            enable_fork=self.config.enable_fork,
        )

    # ------------------------------------------------------------------
    # Fork 分支 → replay + fork root + BranchRunner
    # ------------------------------------------------------------------
    def _run_fork(self, task: dict) -> BranchResult:
        config: BranchConfig = task["branch_config"]
        meta = task["meta"]
        ep_id = task["episode_id"]
        fc = config.fork_config or {}

        out_file = os.path.join(self.config.output_dir, f"{ep_id}.json")
        if not os.path.exists(out_file):
            raise FileNotFoundError(f"Episode not found: {out_file}")
        ep = EpisodeManager.load(out_file)

        # 收集共享 step
        shared_ids = set(config.shared_context_step_ids)
        shared_steps = [
            s for s in ep.data["steps"]
            if s.get("branch_id") == config.parent_branch_id and s["step_id"] in shared_ids
        ]
        shared_steps.sort(key=lambda x: x.get("step_index_in_branch", 0))

        env = EnvController(scene=meta["scene"])
        try:
            # 1. Replay 共享上下文
            env.reset_to_alfred_scene(ep.data["alfred_scene"])
            replay_steps(env, shared_steps, skip_failed=True)

            # 2. 执行 fork 替代动作
            alt_action = fc.get("alternative_action", {})
            alt_name = alt_action.get("action")
            alt_params = alt_action.get("params", {}) or {}
            if not alt_name:
                raise ValueError(f"Fork alternative_action lacks action: {alt_action}")

            alt_result = env.step(alt_name, **alt_params)

            if not alt_result.get("success"):
                image_dir = os.path.join(self.config.output_dir, ep_id)
                fork_root = self.recorder.build_step(
                    step_id="s0",
                    branch_id=config.branch_id,
                    parent_step_id=config.diverges_at_step_id,
                    step_index=0,
                    action=alt_name,
                    action_params=alt_params,
                    result=alt_result,
                    image_dir=image_dir,
                )
                fork_root["error_type"] = "environment_failure"
                fork_root["fork_metadata"] = {
                    "is_fork_root": True,
                    "fork_source_branch_id": config.parent_branch_id,
                    "fork_source_step_id": fc.get("origin_step_id"),
                    "replaces_step_id": fc.get("replaces_step_id"),
                    "reasoning_rewritten_by": "oracle",
                }
                ep.add_step(fork_root)
                return BranchResult(
                    branch_id=config.branch_id, termination_reason="fork_failed",
                    total_steps=1, fork_tasks=[], fork_source_step_ids=[],
                )

            # 3. 重写推理 + 记录 fork root step（★ 统一 step_id 格式: s0）
            rewritten = self.fork_manager._rewrite_reasoning(
                shared_steps=shared_steps,
                original_action="?",
                original_reasoning="",
                alternative_action=alt_action,
                counterfactual_text=fc.get("counterfactual_text", ""),
            )

            image_dir = os.path.join(self.config.output_dir, ep_id)
            fork_root = self.recorder.build_step(
                step_id="s0",
                branch_id=config.branch_id,
                parent_step_id=config.diverges_at_step_id,
                step_index=0,
                action=alt_name,
                action_params=alt_params,
                result=alt_result,
                image_dir=image_dir,
            )
            fork_root["eb_reasoning"] = rewritten
            fork_root["fork_metadata"] = {
                "is_fork_root": True,
                "fork_source_branch_id": config.parent_branch_id,
                "fork_source_step_id": fc.get("origin_step_id"),
                "replaces_step_id": fc.get("replaces_step_id"),
                "reasoning_rewritten_by": "oracle",
            }
            ep.add_step(fork_root)

            # 4. 委托 BranchRunner 从 step_index=1 继续 ★ 链: s0(parent) ← s1
            return self.branch_runner.run(
                config=config, env=env, ep=ep,
                start_step_index=1,
            )
        finally:
            env.close()

    def _fork_entry(self, parent_task: dict, fork_task: dict) -> dict:
        return {
            "traj_path": parent_task.get("traj_path", ""),
            "episode_id": parent_task["episode_id"],
            "meta": parent_task["meta"],
            "base_step_count": parent_task["base_step_count"],
            "branch_config": BranchConfig(**fork_task),
        }

    def _report(self, total: int):
        remaining = len(self.queue) + len(self.fork_queue)
        print(f"\r[{self.stats['completed']}/{total}] ok, {remaining} queued, "
              f"{self.stats['failed']} failed", end="", flush=True)
