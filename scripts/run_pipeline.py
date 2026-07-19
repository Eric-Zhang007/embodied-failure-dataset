"""
主入口脚本：运行完整的 Phase 1-4 数据生成流水线。

运行（WSL2 中）：
    cd ~/embodied-failure-dataset
    uv run python scripts/run_pipeline.py --max 10                     # 跑 10 条 main
    uv run python scripts/run_pipeline.py --task pick_and_place_simple # 按类型筛选
    uv run python scripts/run_pipeline.py --max 50 --eb-model qwen2.5-vl:7b
"""

import argparse

from src.scheduler import Scheduler, SchedulerConfig
from src.vlm_client import VLMClient


def main():
    parser = argparse.ArgumentParser(description="Embodied Failure Dataset Pipeline")
    parser.add_argument("--data-dir", default="data/json_2.1.0")
    parser.add_argument("--output", default="")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--splits", default="train,valid_seen,valid_unseen")
    parser.add_argument("--api-key", default="", help="OpenAI API key")
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
    parser.add_argument("--no-fork", action="store_true", help="Disable counterfactual forks")
    parser.add_argument("--no-traps", action="store_true", help="Disable initial trap injection and Phase 2")
    parser.add_argument("--memory", default="semantic", choices=["semantic", "geometric"],
                        help="Memory mode: semantic (receptacle-grouped) or geometric (flat list)")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    if not args.output:
        from datetime import datetime
        args.output = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    VLMClient.set_api_log_dir(args.output)

    config = SchedulerConfig(
        data_dir=args.data_dir,
        output_dir=args.output,
        max_episodes=args.max,
        task_filter=args.task,
        splits=args.splits,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        planner_model=args.planner_model,
        oracle_model=args.oracle_model,
        executor_model=args.executor_model,
        planner_reasoning_effort=args.planner_reasoning_effort,
        executor_reasoning_effort=args.executor_reasoning_effort,
        oracle_reasoning_effort=args.oracle_reasoning_effort,
        enable_fork=not args.no_fork,
        max_parallel=args.parallel,
        memory_mode=args.memory,
        no_traps=args.no_traps,
    )

    scheduler = Scheduler(config)
    scheduler.run()


if __name__ == "__main__":
    main()
