"""
全局调度器。维护跨 episode 分支队列，委托 BranchRunner 执行 Phase 1-4 循环。
"""

import os
import glob
import logging
import queue
import threading
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import OracleAgent
from src.executor import ExecutorAgent
from src.branch_runner import (
    BranchRunner, BranchConfig, BranchResult,
    run_single_branch, replay_steps, _sid,
    execute_move_sequence_steps,
)
from src.fork_manager import ForkManager
from src.env_controller import EnvController
from src.episode_manager import EpisodeManager
from src.alfred_parser import TASK_TYPE_MAP, load_traj, extract_metadata, extract_low_actions
from src.step_recorder import StepRecorder
from src.action_adapter import resolve_object_ids, adapt


@dataclass
class SchedulerConfig:
    data_dir: str = "data/json_2.1.0"
    output_dir: str = "output"
    max_episodes: int = 0
    max_parallel: int = 4
    task_filter: str = ""
    splits: str = "train,valid_seen,valid_unseen"
    api_key: str = ""
    api_base_url: str = "https://cdn.9527code.com/v1"
    planner_model: str = "gpt-5.5"
    executor_model: str = "gpt-5.5"
    oracle_model: str = "gpt-5.5"
    planner_reasoning_effort: str = "medium"
    executor_reasoning_effort: str = "medium"
    oracle_reasoning_effort: str = "medium"
    enable_fork: bool = True
    memory_mode: str = "semantic"  # "semantic" | "geometric"
    no_traps: bool = False
    max_worker_retries: int = 2
    task_lanes: bool = False


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        # Warmup: GPT-5.5 首次调用需加载，避免首次 Phase 超时
        import time as _time
        warmup_client = VLMClient.openai(
            config.planner_model, config.api_key,
            base_url=config.api_base_url,
            reasoning_effort=config.planner_reasoning_effort,
            recover_api_outages=False,
        )
        _t0 = _time.time()
        try:
            warmup_client.chat_text(system_prompt="Say OK.", user_text="OK", max_tokens=5)
            print(f"API warmup: {_time.time() - _t0:.1f}s")
        except Exception as e:
            print(f"API warmup skipped (API error: {e})")
        self.queue: deque[dict] = deque()
        self.recorder = StepRecorder()

    def _create_agents(self):
        """Create fresh per-worker agents (VLMClient not thread-safe across models)."""
        planner_client = VLMClient.openai(
            self.config.planner_model, self.config.api_key,
            base_url=self.config.api_base_url,
            reasoning_effort=self.config.planner_reasoning_effort,
        )
        executor_client = VLMClient.openai(
            self.config.executor_model, self.config.api_key,
            base_url=self.config.api_base_url,
            reasoning_effort=self.config.executor_reasoning_effort,
        )
        oracle_client = VLMClient.openai(
            self.config.oracle_model, self.config.api_key,
            base_url=self.config.api_base_url,
            reasoning_effort=self.config.oracle_reasoning_effort,
        )
        eb_agent = EBAgent(planner_client)
        oracle_agent = OracleAgent(oracle_client)
        executor_agent = ExecutorAgent(executor_client)
        return eb_agent, oracle_agent, executor_agent

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
            out_file = os.path.join(self.config.output_dir, f"{episode_id}.json")
            if os.path.exists(out_file):
                episode = EpisodeManager.load(out_file)
                status = episode.data["status"]
                main_outcome = (episode.data.get("final_outcome") or {}).get("main_branch")
                if main_outcome and main_outcome.get("termination_reason"):
                    if status != "completed":
                        episode.set_status("completed")
                    continue
                if status == "completed":
                    continue
                if status == "running":
                    pid = episode.data.get("pid")
                    if pid and self._pid_is_alive(pid):
                        continue
                    episode.set_status("interrupted")
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
        if self.config.task_lanes:
            self._run_task_lanes()
            return
        total = len(self.queue)
        max_workers = max(1, self.config.max_parallel)
        self.stats = {
            "terminal": 0,
            "task_complete": 0,
            "completed": 0,  # Backward-compatible alias for terminal.
            "skipped": 0,
            "failed": 0,
            "fork_duplicates": 0,
        }
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_workers)
        _fork_queue = queue.Queue()
        admitted_branch_keys = {self._branch_key(task) for task in self.queue}
        for index, task in enumerate(self.queue, start=1):
            task["_fork_queue"] = _fork_queue
            task.setdefault("_display_index", index)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: deque[dict] = deque(self.queue)
            futures: dict = {}

            def submit_available() -> None:
                while pending and len(futures) < max_workers:
                    task = pending.popleft()
                    futures[executor.submit(
                        self._run_task_worker, task, task.get("_display_index", 0), total,
                    )] = task

            submit_available()
            while futures or pending:
                submit_available()
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as e:
                        result = BranchResult(
                            branch_id=task["branch_config"].branch_id,
                            termination_reason=f"worker_crash:{e}",
                            total_steps=0,
                            fork_tasks=[],
                            fork_source_step_ids=[],
                        )

                    if result.termination_reason.startswith("skipped"):
                        with self._lock:
                            self.stats["skipped"] += 1
                    elif result.termination_reason.startswith("worker_crash"):
                        retries = task.get("_worker_retries", 0)
                        is_main = task["branch_config"].branch_id == "main"
                        if is_main and retries < self.config.max_worker_retries:
                            retry_task = dict(task)
                            retry_task["_worker_retries"] = retries + 1
                            pending.appendleft(retry_task)
                            print(
                                f"\nRETRY {task['episode_id']}: "
                                f"attempt {retries + 1}/{self.config.max_worker_retries} "
                                f"after {result.termination_reason}"
                            )
                        else:
                            with self._lock:
                                self.stats["failed"] += 1
                            if is_main:
                                out_file = os.path.join(
                                    self.config.output_dir, f"{task['episode_id']}.json"
                                )
                                if os.path.exists(out_file):
                                    EpisodeManager.load(out_file).set_status("failed")
                            print(f"\nCRASH {task['episode_id']}: {result.termination_reason}")
                    else:
                        with self._lock:
                            self.stats["terminal"] += 1
                            self.stats["completed"] += 1
                            if result.termination_reason == "task_complete":
                                self.stats["task_complete"] += 1

                        for fork_task in result.fork_tasks:
                            entry = self._fork_entry(task, fork_task)
                            if self._admit_fork_entry(entry, admitted_branch_keys):
                                total += 1
                                pending.append(entry)

                    # Drain fork_queue for forks submitted mid-run via callback
                    while not _fork_queue.empty():
                        entry = _fork_queue.get_nowait()
                        if self._admit_fork_entry(entry, admitted_branch_keys):
                            total += 1
                            pending.append(entry)
                    self._report(total)

        self._print_summary()

    def _run_task_lanes(self):
        """Keep one worker active for each raw ALFRED task type."""
        lane_order = list(TASK_TYPE_MAP)
        lanes: dict[str, deque[dict]] = {task_type: deque() for task_type in lane_order}
        for task in self.queue:
            task_type = task["meta"].get("alfred_task_type")
            if task_type in lanes:
                lanes[task_type].append(task)

        empty_lanes = [task_type for task_type, tasks in lanes.items() if not tasks]
        if empty_lanes:
            raise RuntimeError(f"No trajectories for task lanes: {', '.join(empty_lanes)}")
        if self.config.max_parallel < len(lane_order):
            raise ValueError(
                f"task_lanes requires at least {len(lane_order)} workers; "
                f"got {self.config.max_parallel}"
            )

        total = len(self.queue)
        self.stats = {
            "terminal": 0,
            "task_complete": 0,
            "completed": 0,  # Backward-compatible alias for terminal.
            "skipped": 0,
            "failed": 0,
            "fork_duplicates": 0,
        }
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(len(lane_order))
        fork_queue = queue.Queue()
        admitted_branch_keys = {self._branch_key(task) for task in self.queue}
        for index, task in enumerate(self.queue, start=1):
            task["_fork_queue"] = fork_queue
            task.setdefault("_display_index", index)

        print(f"Running {len(lane_order)} task lanes: {', '.join(lane_order)}")
        with ThreadPoolExecutor(max_workers=len(lane_order)) as executor:
            futures: dict = {}
            active_lanes: set[str] = set()

            def submit_lane(task_type: str) -> None:
                if task_type in active_lanes or not lanes[task_type]:
                    return
                task = lanes[task_type].popleft()
                active_lanes.add(task_type)
                futures[executor.submit(
                    self._run_task_worker,
                    task,
                    task.get("_display_index", 0),
                    total,
                )] = (task_type, task)

            for task_type in lane_order:
                submit_lane(task_type)

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task_type, task = futures.pop(future)
                    active_lanes.remove(task_type)
                    try:
                        result = future.result()
                    except Exception as e:
                        result = BranchResult(
                            branch_id=task["branch_config"].branch_id,
                            termination_reason=f"worker_crash:{e}",
                            total_steps=0,
                            fork_tasks=[],
                            fork_source_step_ids=[],
                        )

                    if result.termination_reason.startswith("skipped"):
                        with self._lock:
                            self.stats["skipped"] += 1
                    elif result.termination_reason.startswith("worker_crash"):
                        retries = task.get("_worker_retries", 0)
                        is_main = task["branch_config"].branch_id == "main"
                        if is_main and retries < self.config.max_worker_retries:
                            retry_task = dict(task)
                            retry_task["_worker_retries"] = retries + 1
                            lanes[task_type].appendleft(retry_task)
                            print(
                                f"\nRETRY {task['episode_id']} ({task_type}): "
                                f"attempt {retries + 1}/{self.config.max_worker_retries}"
                            )
                        else:
                            with self._lock:
                                self.stats["failed"] += 1
                            if is_main:
                                out_file = os.path.join(
                                    self.config.output_dir, f"{task['episode_id']}.json"
                                )
                                if os.path.exists(out_file):
                                    EpisodeManager.load(out_file).set_status("failed")
                            print(f"\nCRASH {task['episode_id']} ({task_type}): {result.termination_reason}")
                    else:
                        with self._lock:
                            self.stats["terminal"] += 1
                            self.stats["completed"] += 1
                            if result.termination_reason == "task_complete":
                                self.stats["task_complete"] += 1
                        for fork_task in reversed(result.fork_tasks):
                            entry = self._fork_entry(task, fork_task)
                            entry["_lane_type"] = task_type
                            if self._admit_fork_entry(entry, admitted_branch_keys):
                                lanes[task_type].appendleft(entry)
                                total += 1

                    while not fork_queue.empty():
                        entry = fork_queue.get_nowait()
                        entry_task_type = entry.get("_lane_type") or entry["meta"].get("alfred_task_type")
                        if entry_task_type in lanes:
                            entry["_lane_type"] = entry_task_type
                            if self._admit_fork_entry(entry, admitted_branch_keys):
                                lanes[entry_task_type].appendleft(entry)
                                total += 1

                    submit_lane(task_type)
                    self._report(total)

                for lane in lane_order:
                    submit_lane(lane)

        self._print_summary()

    def _run_task_worker(self, task: dict, n: int, total: int) -> BranchResult:
        with self._semaphore:
            config: BranchConfig = task["branch_config"]
            if config.branch_id == "main":
                return self._run_branch_worker(task, n, total)
            agents = self._create_agents()
            return self._run_fork(task, agents)

    def _run_branch_worker(self, task: dict, n: int, total: int) -> BranchResult:
        """Run or resume a main branch with fresh per-worker agents."""
        ep_id = task["episode_id"]
        out_file = os.path.join(self.config.output_dir, f"{ep_id}.json")
        existing = None
        if os.path.exists(out_file):
            existing = EpisodeManager.load(out_file)
            status = existing.data["status"]
            if status == "completed":
                return BranchResult(branch_id="main", termination_reason="skipped",
                                    total_steps=0, fork_tasks=[], fork_source_step_ids=[])
            if status == "running":
                pid = existing.data.get("pid")
                if pid and self._pid_is_alive(pid):
                    return BranchResult(branch_id="main", termination_reason="skipped",
                                        total_steps=0, fork_tasks=[], fork_source_step_ids=[])
                existing.set_status("interrupted")

        print(f"\n[{n}/{total}] Starting {ep_id}")
        eb_agent, oracle_agent, executor_agent = self._create_agents()
        if existing:
            existing.set_status("running", os.getpid())

        _fork_queue = task.get("_fork_queue")
        if _fork_queue is not None:
            def _on_fork(fork_task):
                _fork_queue.put(self._fork_entry(task, fork_task))
        else:
            _on_fork = None

        try:
            if existing and existing.get_steps_for_branch("main"):
                result = BranchRunner.resume(
                    episode_path=out_file,
                    branch_id="main",
                    eb_agent=eb_agent,
                    oracle_agent=oracle_agent,
                    executor_agent=executor_agent,
                    output_dir=self.config.output_dir,
                    enable_fork=self.config.enable_fork,
                    memory_mode=self.config.memory_mode,
                    fork_callback=_on_fork,
                )
            else:
                result = run_single_branch(
                    traj_path=task["traj_path"],
                    eb_agent=eb_agent,
                    oracle_agent=oracle_agent,
                    output_dir=self.config.output_dir,
                    enable_phase2=not self.config.no_traps,
                    enable_fork=self.config.enable_fork,
                    executor_agent=executor_agent,
                    memory_mode=self.config.memory_mode,
                    episode_status="running",
                    fork_callback=_on_fork,
                )

            if os.path.exists(out_file):
                final_status = (
                    "failed" if result.termination_reason.startswith("worker_crash")
                    else "completed"
                )
                EpisodeManager.load(out_file).set_status(final_status)
            print(f"[{n}/{total}] Finished {ep_id}: {result.termination_reason} ({result.total_steps} steps)")
            return result
        except KeyboardInterrupt:
            if os.path.exists(out_file):
                EpisodeManager.load(out_file).set_status("interrupted")
            raise
        except Exception as e:
            logging.warning("Worker for %s crashed, marking interrupted: %s", ep_id, e)
            if os.path.exists(out_file):
                EpisodeManager.load(out_file).set_status("interrupted")
            return BranchResult(branch_id="main", termination_reason="worker_crash",
                                total_steps=0, fork_tasks=[], fork_source_step_ids=[])

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------
    # Fork 分支 → replay + fork root + BranchRunner
    # ------------------------------------------------------------------
    def _run_fork(self, task: dict, agents: tuple) -> BranchResult:
        config: BranchConfig = task["branch_config"]
        meta = task["meta"]
        ep_id = task["episode_id"]
        fc = config.fork_config or {}

        out_file = os.path.join(self.config.output_dir, f"{ep_id}.json")
        if not os.path.exists(out_file):
            raise FileNotFoundError(f"Episode not found: {out_file}")
        ep = EpisodeManager.load(out_file)
        if ep.get_steps_for_branch(config.branch_id):
            logging.warning(
                "DUPLICATE_FORK_SUPPRESSED episode=%s branch=%s persisted_steps=%d",
                ep_id, config.branch_id, len(ep.get_steps_for_branch(config.branch_id)),
            )
            return BranchResult(
                branch_id=config.branch_id,
                termination_reason="skipped_duplicate_fork",
                total_steps=0,
                fork_tasks=[],
                fork_source_step_ids=[],
            )
        print(f"\n[fork] Starting {config.branch_id} (parent={config.parent_branch_id}) on {ep_id}")

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
            replay_steps(
                env, shared_steps, skip_failed=True,
                pddl_params=ep.data.get("pddl_params", {}),
            )

            # 2. 执行 fork 替代动作
            alt_action = fc.get("alternative_action", {})
            alt_name = alt_action.get("action")
            alt_params = alt_action.get("params", {}) or {}
            if not alt_name:
                raise ValueError(f"Fork alternative_action lacks action: {alt_action}")

            if alt_name == "MoveSequence":
                steps = alt_params.get("steps", [])
                success, frame, metadata, error = execute_move_sequence_steps(
                    env, steps, env.get_state_snapshot()["metadata"],
                )
                alt_result = {
                    "success": success,
                    "error": error,
                    "frame": frame,
                    "metadata": metadata or env.get_state_snapshot()["metadata"],
                }
                exec_name = alt_name
                exec_params = alt_params
            else:
                fork_objects = env.get_state_snapshot()["metadata"].get("objects", [])
                resolved_params, resolve_warning = resolve_object_ids(
                    alt_name, alt_params, fork_objects
                )
                exec_name, exec_params = adapt(alt_name, resolved_params)
                if resolve_warning:
                    logging.warning(
                        "fork %s: alt-action %s objectId resolution failed: %s",
                        config.branch_id, alt_name, resolve_warning,
                    )
                alt_result = env.step(exec_name, **exec_params)

            if not alt_result.get("success"):
                image_dir = os.path.join(self.config.output_dir, ep_id)
                fork_root = self.recorder.build_step(
                    step_id=_sid(config.branch_id, 0),
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

            # 3. 重写推理 + 记录 fork root step
            eb_agent, oracle_agent, executor_agent = agents
            replaces_id = fc.get("replaces_step_id", "")
            replaced_step = next(
                (s for s in ep.data["steps"] if s.get("step_id") == replaces_id),
                {},
            )
            fm = ForkManager(eb_agent, oracle_agent, None, self.config.output_dir)
            rewritten = fm._rewrite_reasoning(
                shared_steps=shared_steps,
                original_action=replaced_step.get("action", "?"),
                original_reasoning=replaced_step.get("eb_reasoning", ""),
                alternative_action=alt_action,
                counterfactual_text=fc.get("counterfactual_text", ""),
            )

            image_dir = os.path.join(self.config.output_dir, ep_id)
            fork_root = self.recorder.build_step(
                step_id=_sid(config.branch_id, 0),
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

            # 4. 委托 BranchRunner 从 step_index=1 继续
            _fork_queue = task.get("_fork_queue")
            if _fork_queue is not None:
                def _on_fork(fork_task):
                    _fork_queue.put(self._fork_entry(task, fork_task))
            else:
                _on_fork = None
            branch_runner = BranchRunner(
                eb_agent, oracle_agent, self.config.output_dir,
                enable_fork=self.config.enable_fork,
                executor_agent=executor_agent,
                _fork_callback=_on_fork,
            )
            result = branch_runner.run(
                config=config, env=env, ep=ep,
                start_step_index=1,
            )
            print(f"[fork] Finished {config.branch_id}: {result.termination_reason} ({result.total_steps} steps)")
            return result
        except KeyboardInterrupt:
            raise
        except Exception:
            raise
        finally:
            env.close()

    def _fork_entry(self, parent_task: dict, fork_task: dict) -> dict:
        entry = {
            "traj_path": parent_task.get("traj_path", ""),
            "episode_id": parent_task["episode_id"],
            "meta": parent_task["meta"],
            "base_step_count": parent_task["base_step_count"],
            "branch_config": BranchConfig(**fork_task),
            "_fork_queue": parent_task.get("_fork_queue"),
        }
        return entry

    @staticmethod
    def _branch_key(task: dict) -> tuple[str, str]:
        config: BranchConfig = task["branch_config"]
        return config.episode_id, config.branch_id

    def _admit_fork_entry(self, entry: dict, admitted_branch_keys: set[tuple[str, str]]) -> bool:
        """Allow each persisted branch identity to execute exactly once per run.

        BranchRunner reports a new fork twice: once immediately via its callback
        and once in its final BranchResult.  Both paths are intentional, but they
        must converge before a worker starts writing the branch trajectory.
        """
        key = self._branch_key(entry)
        if key in admitted_branch_keys:
            self.stats["fork_duplicates"] += 1
            logging.warning(
                "DUPLICATE_FORK_SUPPRESSED episode=%s branch=%s",
                key[0], key[1],
            )
            return False
        admitted_branch_keys.add(key)
        return True

    def _report(self, total: int):
        done = self.stats["terminal"] + self.stats["failed"] + self.stats["skipped"]
        remaining = total - done
        non_success_terminal = self.stats["terminal"] - self.stats["task_complete"]
        print(
            f"\r[{done}/{total}] {self.stats['task_complete']} task_complete, "
            f"{non_success_terminal} terminal_non_success, {remaining} remaining, "
            f"{self.stats['failed']} worker_failed, {self.stats['skipped']} skipped, "
            f"{self.stats['fork_duplicates']} fork_duplicates",
            end="",
            flush=True,
        )

    def _print_summary(self):
        non_success_terminal = self.stats["terminal"] - self.stats["task_complete"]
        print(
            f"\n\nDone. {self.stats['task_complete']} task_complete, "
            f"{non_success_terminal} terminal_non_success, "
            f"{self.stats['failed']} worker_failed, {self.stats['skipped']} skipped, "
            f"{self.stats['fork_duplicates']} fork_duplicates."
        )
