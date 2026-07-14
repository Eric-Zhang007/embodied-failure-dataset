"""
端到端测试：加载 ALFRED 轨迹，跑 Phase 1-4 循环。
支持单任务 (--task) 或全任务并行 (--all)。

运行（WSL2 中）：
    uv run python scripts/e2e_test.py --api-key sk-xxx --task pick_and_place_simple
    uv run python scripts/e2e_test.py --api-key sk-xxx --all
    uv run python scripts/e2e_test.py --api-key sk-xxx --all --random --no-traps
"""
import argparse
import glob
import os
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import OracleAgent
from src.executor import ExecutorAgent
from src.branch_runner import run_single_branch
from src.trap_planner import TrapPlanner

ALL_TASKS = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
    "pick_and_place_with_movable_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def run_one(args, task_type, traj_path, n, total):
    """Run a single episode in a worker thread."""
    ep_id = os.path.basename(os.path.dirname(traj_path))
    prefix = f"[{n}/{total}]" if total > 1 else ""

    # Each worker creates its own agents (VLMClient is not thread-safe across models)
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

    trap_planner = None if args.no_traps else TrapPlanner()

    # Compute ablation flags: --ablation includes spikes to DISABLE
    ablation_set = set(args.ablation)
    result = run_single_branch(
        traj_path=traj_path,
        eb_agent=eb_agent,
        oracle_agent=oracle_agent,
        output_dir=args.output,
        trap_planner=trap_planner,
        enable_fork=False,
        step_limit_multiplier=2,
        executor_agent=executor_agent,
        enable_searched_markers="001" not in ablation_set,
        enable_intent_dedup="002a" not in ablation_set,
        enable_critic_guard="002b" in ablation_set,  # 002b is off by default; --ablation 002b ENABLES it
        enable_curiosity_scoreboard="003" not in ablation_set,
        enable_contrastive_planner="004" not in ablation_set,
        enable_progress_gating="005" not in ablation_set,
        enable_search_trail="006" not in ablation_set,
    )
    print(f"{prefix} {task_type}: {ep_id} -> {result.termination_reason} ({result.total_steps} steps)")
    return task_type, ep_id, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--api-base-url", default="https://api.fullcupai.com", help="OpenAI-compatible API base URL")
    parser.add_argument("--planner-model", default="gpt-5.5")
    parser.add_argument("--oracle-model", default="gpt-5.5")
    parser.add_argument("--executor-model", default="gpt-5.5")
    parser.add_argument("--planner-reasoning-effort", default="xhigh",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--executor-reasoning-effort", default="medium",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--oracle-reasoning-effort", default="xhigh",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--task", default="", help="Single ALFRED task_type")
    parser.add_argument("--all", action="store_true", help="Run all 7 task types in parallel")
    parser.add_argument("--data-dir", default="data/json_2.1.0")
    parser.add_argument("--output", default="")
    parser.add_argument("--no-traps", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--parallel", type=int, default=3, help="Max parallel workers for --all")
    parser.add_argument("--ablation", nargs="*", default=[],
                        choices=["001", "002a", "002b", "003", "004", "005", "006"],
                        help="Disable specific spikes for ablation testing (e.g. --ablation 003 006)")
    args = parser.parse_args()

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
                traj = random.choice(task_files) if args.random else task_files[0]
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
        traj = random.choice(task_files) if args.random else task_files[0]
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
    else:
        # Parallel execution
        max_workers = min(args.parallel, total)
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, (task_type, traj_path) in enumerate(tasks_to_run, 1):
                f = executor.submit(run_one, args, task_type, traj_path, i, total)
                futures[f] = task_type
            for future in as_completed(futures):
                task_type, ep_id, result = future.result()
                results[task_type] = result

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
