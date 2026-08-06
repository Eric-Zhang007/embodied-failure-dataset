"""
端到端测试：加载 ALFRED 轨迹，跑 Phase 1-4 循环。
支持单任务 (--task) 或全任务并行 (--all)。

运行（WSL2 中）：
    uv run python scripts/e2e_test.py --api-key sk-xxx --task pick_and_place_simple
    uv run python scripts/e2e_test.py --api-key sk-xxx --all
    uv run python scripts/e2e_test.py --api-key sk-xxx --all --random --no-traps
"""
import argparse
import configparser
import glob
import os
import sys
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import OracleAgent
from src.executor import ExecutorAgent
from src.branch_runner import run_single_branch, BranchRunner
from src.episode_manager import EpisodeManager
from src.env_controller import EnvController
from src.branch_runner import BranchConfig

ALL_TASKS = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
    "pick_and_place_with_movable_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def _load_api_config(config_path: str) -> dict[str, str]:
    """Read the small TOML-compatible configuration subset used by collection."""
    parser = configparser.ConfigParser()
    if not parser.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"Could not read config file: {config_path}")
    if not parser.has_option("api", "api_key"):
        raise ValueError("Config file must define [api] api_key")
    def value(section: str, option: str, fallback: str = "") -> str:
        return parser.get(section, option, fallback=fallback).strip().strip('"')

    return {
        "api_key": value("api", "api_key"),
        "api_base_url": value("api", "base_url"),
        "planner_model": value("models", "planner"),
        "executor_model": value("models", "executor"),
        "oracle_model": value("models", "oracle"),
    }


def _persist_main_result_status(episode_path: str, termination_reason: str) -> None:
    """Persist a terminal main-branch status when an episode file was created."""
    if not os.path.exists(episode_path):
        return
    final_status = "failed" if termination_reason.startswith("worker_crash") else "completed"
    EpisodeManager.load(episode_path).set_status(final_status)


def run_one(args, task_type, traj_path, n, total):
    """Run a single episode in a worker thread."""
    ep_id = os.path.basename(os.path.dirname(traj_path))
    prefix = f"[{n}/{total}]" if total > 1 else ""

    eb_agent, oracle_agent, executor_agent = _make_agents(args)
    # Compute ablation flags: --ablation includes spikes to DISABLE
    ablation_set = set(args.ablation)
    result = run_single_branch(
        traj_path=traj_path,
        eb_agent=eb_agent,
        oracle_agent=oracle_agent,
        output_dir=args.output,
        enable_phase2=not args.no_traps,
        enable_fork=args.enable_fork,
        step_limit_multiplier=2,
        executor_agent=executor_agent,
        enable_searched_markers="001" not in ablation_set,
        enable_intent_dedup="002a" not in ablation_set,
        enable_critic_guard="002b" in ablation_set,  # 002b is off by default; --ablation 002b ENABLES it
        enable_curiosity_scoreboard="003" not in ablation_set,
        enable_contrastive_planner="004" not in ablation_set,
        enable_progress_gating="005" not in ablation_set,
        enable_search_trail="006" not in ablation_set,
        memory_mode=args.memory,
        episode_status="running",
    )
    _persist_main_result_status(os.path.join(args.output, f"{ep_id}.json"), result.termination_reason)
    print(f"{prefix} {task_type}: {ep_id} -> {result.termination_reason} ({result.total_steps} steps)")
    return task_type, ep_id, result


def _make_agents(args):
    """Create a fresh set of agents for one worker."""
    planner_client = VLMClient.openai(
        args.planner_model, args.api_key,
        base_url=args.api_base_url,
        reasoning_effort=args.planner_reasoning_effort,
    )
    executor_client = VLMClient.openai(
        args.executor_model, args.api_key,
        base_url=args.api_base_url,
        reasoning_effort=args.executor_reasoning_effort,
    )
    oracle_client = VLMClient.openai(
        args.oracle_model, args.api_key,
        base_url=args.api_base_url,
        reasoning_effort=args.oracle_reasoning_effort,
    )
    eb_agent = EBAgent(planner_client)
    oracle_agent = OracleAgent(oracle_client)
    executor_agent = ExecutorAgent(executor_client)
    return eb_agent, oracle_agent, executor_agent


def _run_fork_e2e(args, parent_task_type, episode_id, fork_task: dict, n: int, total: int):
    """Run a fork branch spawned from a completed main branch."""
    from src.branch_runner import replay_steps
    from src.fork_manager import ForkManager
    from src.step_recorder import StepRecorder
    from src.alfred_scene import _require_alfred_scene

    out_file = os.path.join(args.output, f"{episode_id}.json")
    if not os.path.exists(out_file):
        return parent_task_type, episode_id, BranchResult(
            branch_id=fork_task.get("branch_id", "fork"),
            termination_reason="fork_skipped_no_episode",
            total_steps=0, fork_tasks=[], fork_source_step_ids=[],
        )

    ep = EpisodeManager.load(out_file)
    config = BranchConfig(**fork_task)
    fc = config.fork_config or {}
    eb_agent, oracle_agent, executor_agent = _make_agents(args)
    recorder = StepRecorder()

    ep.set_status("running", os.getpid())
    print(f"\n[fork {n}/{total}] Starting {config.branch_id} (parent={config.parent_branch_id}) on {episode_id}")

    env = EnvController(scene=ep.data.get("scene", ""))
    try:
        # 1. Replay shared context steps from parent branch
        shared_ids = set(config.shared_context_step_ids)
        shared_steps = [
            s for s in ep.data["steps"]
            if s.get("branch_id") == config.parent_branch_id and s["step_id"] in shared_ids
        ]
        shared_steps.sort(key=lambda x: x.get("step_index_in_branch", 0))

        env.reset_to_alfred_scene(_require_alfred_scene(ep.data))
        replay_steps(
            env, shared_steps, skip_failed=True,
            pddl_params=ep.data.get("pddl_params", {}),
        )

        # 2. Execute fork alternative action
        alt_action = fc.get("alternative_action", {})
        alt_name = alt_action.get("action")
        alt_params = alt_action.get("params", {}) or {}
        if not alt_name:
            raise ValueError(f"Fork alternative_action lacks action: {alt_action}")

        alt_result = env.step(alt_name, **alt_params)
        if not alt_result.get("success"):
            image_dir = os.path.join(args.output, episode_id)
            fork_root = recorder.build_step(
                step_id="s0", branch_id=config.branch_id,
                parent_step_id=config.diverges_at_step_id,
                step_index=0, action=alt_name, action_params=alt_params,
                result=alt_result, image_dir=image_dir,
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
            ep2 = EpisodeManager.load(out_file)
            ep2.set_status("failed")
            return parent_task_type, episode_id, BranchResult(
                branch_id=config.branch_id, termination_reason="fork_failed",
                total_steps=1, fork_tasks=[], fork_source_step_ids=[],
            )

        # 3. Rewrite reasoning + record fork root
        fm = ForkManager(eb_agent, oracle_agent, None, args.output)
        rewritten = fm._rewrite_reasoning(
            shared_steps=shared_steps,
            original_action="?",
            original_reasoning="",
            alternative_action=alt_action,
            counterfactual_text=fc.get("counterfactual_text", ""),
        )

        image_dir = os.path.join(args.output, episode_id)
        fork_root = recorder.build_step(
            step_id="s0", branch_id=config.branch_id,
            parent_step_id=config.diverges_at_step_id,
            step_index=0, action=alt_name, action_params=alt_params,
            result=alt_result, image_dir=image_dir,
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

        # 4. Delegate to BranchRunner from step_index=1
        branch_runner = BranchRunner(
            eb_agent, oracle_agent, args.output,
            enable_fork=args.enable_fork,
            executor_agent=executor_agent,
        )
        result = branch_runner.run(
            config=config, env=env, ep=ep,
            start_step_index=1,
        )

        ep2 = EpisodeManager.load(out_file)
        final_status = "failed" if result.termination_reason.startswith("worker_crash") else "completed"
        ep2.set_status(final_status)
        print(f"  [fork] Finished {config.branch_id}: {result.termination_reason} ({result.total_steps} steps)")
        return parent_task_type, episode_id, result
    except KeyboardInterrupt:
        if os.path.exists(out_file):
            EpisodeManager.load(out_file).set_status("interrupted")
        raise
    except Exception:
        if os.path.exists(out_file):
            EpisodeManager.load(out_file).set_status("failed")
        raise
    finally:
        env.close()


def _run_main_with_sem(sem, args, task_type, traj_path, n, total):
    """Wrapper: acquire semaphore, then run main branch."""
    with sem:
        return run_one(args, task_type, traj_path, n, total)


def _run_fork_with_sem(sem, args, task_type, episode_id, fork_task, n, total):
    """Wrapper: acquire semaphore, then run fork branch."""
    with sem:
        return _run_fork_e2e(args, task_type, episode_id, fork_task, n, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default="", help="OpenAI API key")
    parser.add_argument("--config", default="", help="Optional config.toml containing [api] api_key")
    parser.add_argument("--api-base-url", default="https://cdn.9527code.com/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--planner-model", default="gpt-5.5")
    parser.add_argument("--oracle-model", default="gpt-5.5")
    parser.add_argument("--executor-model", default="gpt-5.5")
    parser.add_argument("--planner-reasoning-effort", default="medium",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--executor-reasoning-effort", default="medium",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--oracle-reasoning-effort", default="medium",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--task", default="", help="Single ALFRED task_type")
    parser.add_argument("--all", action="store_true", help="Run all 7 task types in parallel")
    parser.add_argument("--trajectory-index", type=int, default=0,
                        help="Deterministic zero-based trajectory index within each task type")
    parser.add_argument("--data-dir", default="data/json_2.1.0")
    parser.add_argument("--output", default="")
    parser.add_argument("--no-traps", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--parallel", type=int, default=3, help="Max parallel workers for --all")
    parser.add_argument("--ablation", nargs="*", default=[],
                        choices=["001", "002a", "002b", "003", "004", "005", "006"],
                        help="Disable specific spikes for ablation testing (e.g. --ablation 003 006)")
    parser.add_argument("--memory", default="semantic", choices=["semantic", "geometric"],
                        help="Memory mode: semantic (receptacle-grouped, freshness-aware) or geometric (flat list)")
    parser.add_argument("--enable-fork", action="store_true", help="Enable Oracle fork tasks on AC counterfactuals")
    args = parser.parse_args()

    if args.config:
        config = _load_api_config(args.config)
        args.api_key = args.api_key or config["api_key"]
    if not args.api_key:
        parser.error("provide --api-key or --config with [api] api_key")

    if not args.output:
        from datetime import datetime
        args.output = f"output_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Collect task→trajectory mapping
    tasks_to_run: list[tuple[str, str]] = []  # (task_type, traj_path)

    if args.all:
        from src.alfred_parser import load_traj
        import random
        pattern = os.path.join(args.data_dir, "train", "**", "traj_data.json")
        all_files = sorted(glob.glob(pattern, recursive=True))
        for task in ALL_TASKS:
            task_files = [f for f in all_files if load_traj(f).get("task_type") == task]
            if task_files:
                traj = random.choice(task_files) if args.random else task_files[args.trajectory_index % len(task_files)]
                tasks_to_run.append((task, traj))
            else:
                print(f"SKIP {task}: no trajectories found")
    elif args.task:
        from src.alfred_parser import load_traj
        import random
        pattern = os.path.join(args.data_dir, "train", "**", "traj_data.json")
        all_files = sorted(glob.glob(pattern, recursive=True))
        task_files = [f for f in all_files if load_traj(f).get("task_type") == args.task]
        if not task_files:
            print(f"No trajectories found for task_type={args.task}")
            sys.exit(1)
        traj = random.choice(task_files) if args.random else task_files[args.trajectory_index % len(task_files)]
        tasks_to_run.append((args.task, traj))
    else:
        print("Specify --task <type> or --all")
        sys.exit(1)

    # Warmup: send a single request to warm the API
    warmup_client = VLMClient.openai(
        args.planner_model, args.api_key,
        base_url=args.api_base_url,
        reasoning_effort=args.planner_reasoning_effort,
    )
    print("Warming up API...", end=" ", flush=True)
    _t0 = _time.time()
    try:
        warmup_client.chat_text(system_prompt="Say OK.", user_text="OK", max_tokens=5)
        print(f"done ({_time.time() - _t0:.1f}s)")
    except Exception as e:
        print(f"skipped (API error: {e})")

    total = len(tasks_to_run)
    VLMClient.set_api_log_dir(args.output)
    print(f"\nRunning {total} task(s) -> {args.output}/\n")

    if total == 1:
        # Single task — run inline
        task_type, traj_path = tasks_to_run[0]
        _, ep_id, result = run_one(args, task_type, traj_path, 1, 1)
        print(f"\n{'='*50}")
        print(f"Task: {task_type}")
        print(f"Branch: {result.branch_id}")
        print(f"Termination: {result.termination_reason}")
        print(f"Total steps: {result.total_steps}")
        print(f"Fork tasks: {len(result.fork_tasks)}")
        if args.enable_fork:
            for fork_index, fork_task in enumerate(result.fork_tasks, 1):
                _, _, fork_result = _run_fork_e2e(
                    args, task_type, ep_id, fork_task, fork_index, len(result.fork_tasks)
                )
                print(
                    f"Fork {fork_index}/{len(result.fork_tasks)}: "
                    f"{fork_result.branch_id} -> {fork_result.termination_reason} "
                    f"({fork_result.total_steps} steps)"
                )
    else:
        # Parallel execution with Semaphore + dynamic fork submission
        max_workers = min(args.parallel, total)
        sem = threading.Semaphore(max_workers)
        results: dict[str, BranchResult] = {}
        n_done = 0
        n_submitted = total

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict = {}
            for i, (task_type, traj_path) in enumerate(tasks_to_run, 1):
                f = executor.submit(_run_main_with_sem, sem, args, task_type, traj_path, i, total)
                futures[f] = ("main", task_type)

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    kind, task_type = futures.pop(future)
                    task_type, ep_id, result = future.result()
                    n_done += 1

                    if kind == "main":
                        results[task_type] = result
                        # Submit fork tasks back to the same pool
                        for ft in result.fork_tasks:
                            n_submitted += 1
                            f = executor.submit(
                                _run_fork_with_sem, sem, args, task_type, ep_id, ft, n_submitted, n_submitted,
                            )
                            futures[f] = ("fork", task_type)

                    print(f"[{n_done}/{n_submitted}] {task_type}: {result.termination_reason} "
                          f"({result.total_steps} steps){' [fork]' if kind == 'fork' else ''}")

        # Summary
        print(f"\n{'='*60}")
        print(f"RESULTS ({args.output})")
        print(f"{'='*60}")
        for task in ALL_TASKS:
            if task in results:
                r = results[task]
                status = "OK" if r.termination_reason == "task_complete" else r.termination_reason
                print(f"  {task:40s} {status:30s} {r.total_steps:3d} steps")
            else:
                print(f"  {task:40s} SKIPPED")


if __name__ == "__main__":
    main()
