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


def main():
    parser = argparse.ArgumentParser(description="Embodied Failure Dataset Pipeline")
    parser.add_argument("--data-dir", default="data/json_2.1.0")
    parser.add_argument("--output", default="output")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--splits", default="train,valid_seen,valid_unseen")
    parser.add_argument("--eb-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--oracle-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--api-key", default="", help="SiliconFlow API key")
    parser.add_argument("--no-fork", action="store_true", help="Disable counterfactual forks")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    config = SchedulerConfig(
        data_dir=args.data_dir,
        output_dir=args.output,
        max_episodes=args.max,
        task_filter=args.task,
        splits=args.splits,
        eb_model=args.eb_model,
        oracle_model=args.oracle_model,
        siliconflow_key=args.api_key,
        enable_fork=not args.no_fork,
        max_parallel=args.parallel,
    )

    scheduler = Scheduler(config)
    scheduler.run()


if __name__ == "__main__":
    main()
