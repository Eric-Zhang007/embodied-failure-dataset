"""
端到端测试：加载一条 ALFRED 轨迹，跑完一个 main 分支的 Phase 1-4 循环。
调用 run_single_branch()，不重复 scheduler 逻辑。

运行（WSL2 中）：
    cd ~/embodied-failure-dataset
    uv run python scripts/e2e_test.py --api-key sk-xxx
    uv run python scripts/e2e_test.py --api-key sk-xxx --task pick_heat_then_place_in_recep
"""

import argparse
import glob
import os
import sys

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import OracleAgent
from src.executor import ExecutorAgent
from src.branch_runner import run_single_branch
from src.trap_planner import TrapPlanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--eb-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--oracle-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--executor-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--task", default="", help="ALFRED task_type filter")
    parser.add_argument("--data-dir", default="data/json_2.1.0")
    parser.add_argument("--output", default="output_e2e")
    parser.add_argument("--no-traps", action="store_true", help="Disable initial traps")
    parser.add_argument("--random", action="store_true", help="Pick a random episode instead of the first")
    args = parser.parse_args()

    # 1. 找一条轨迹
    pattern = os.path.join(args.data_dir, "train", "**", "traj_data.json")
    all_files = sorted(glob.glob(pattern, recursive=True))
    if args.task:
        from src.alfred_parser import load_traj
        all_files = [f for f in all_files if load_traj(f).get("task_type") == args.task]

    if not all_files:
        print(f"No trajectories found for task_type={args.task}")
        sys.exit(1)

    if args.random:
        import random
        traj_path = random.choice(all_files)
    else:
        traj_path = all_files[0]
    print(f"Episode: {os.path.basename(os.path.dirname(traj_path))}")
    print(f"Traj: {traj_path}\n")

    # 2. 创建 Agent
    eb_client = VLMClient.siliconflow(args.eb_model, args.api_key)
    oracle_client = VLMClient.siliconflow(args.oracle_model, args.api_key)
    executor_client = VLMClient.siliconflow(args.executor_model, args.api_key)
    eb_agent = EBAgent(eb_client)
    oracle_agent = OracleAgent(oracle_client)
    executor_agent = ExecutorAgent(executor_client)

    # 2.5 Warmup: 32B 首次调用需要加载模型，提前发一个请求
    import time as _time
    print("Warming up Oracle model...", end=" ", flush=True)
    _t0 = _time.time()
    try:
        oracle_client.chat_text(system_prompt="Say OK.", user_text="OK", max_tokens=5)
        print(f"done ({_time.time() - _t0:.1f}s)")
    except Exception as e:
        print(f"skipped (API error: {e})")

    # 3. 跑
    trap_planner = None if args.no_traps else TrapPlanner()
    result = run_single_branch(
        traj_path=traj_path,
        eb_agent=eb_agent,
        oracle_agent=oracle_agent,
        output_dir=args.output,
        trap_planner=trap_planner,
        enable_fork=False,
        step_limit_multiplier=2,
        executor_agent=executor_agent,
    )

    # 4. 结果
    print(f"\n{'='*50}")
    print(f"Branch: {result.branch_id}")
    print(f"Termination: {result.termination_reason}")
    print(f"Total steps: {result.total_steps}")
    print(f"Fork tasks: {len(result.fork_tasks)}")


if __name__ == "__main__":
    main()
