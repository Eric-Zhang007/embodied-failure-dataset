"""
ALFRED 轨迹重放引擎。

读取 data/json_2.1.0/ 下的所有 traj_data.json，
逐个在 AI2-THOR 中重放，截图并写入我们的 episode 格式。

运行（WSL2 中）：
    cd ~/embodied-failure-dataset
    uv run python scripts/alfred_replay.py                    # 全量
    uv run python scripts/alfred_replay.py --max 10           # 只跑 10 条
    uv run python scripts/alfred_replay.py --task pick_and_place_simple  # 按任务类型筛选
"""

import argparse
import glob
import os
import sys

from src.env_controller import EnvController
from src.step_recorder import StepRecorder
from src.episode_manager import EpisodeManager
from src.alfred_parser import load_traj, extract_metadata, extract_low_actions
from src.action_adapter import adapt


def replay_one(traj_path: str, output_dir: str) -> bool:
    """
    重放单条 ALFRED 轨迹，写入我们的 episode JSON。
    返回 True 表示重放完成。异常直接抛出，不静默跳过。
    """
    traj = load_traj(traj_path)
    meta = extract_metadata(traj)
    # 用目录名作为 episode_id
    episode_id = os.path.basename(os.path.dirname(traj_path))

    # 跳过已完成
    out_file = os.path.join(output_dir, f"{episode_id}.json")
    if os.path.exists(out_file) and os.path.isdir(os.path.join(output_dir, episode_id)):
        return True

    env = EnvController(scene=meta["scene"])
    ep = EpisodeManager(episode_id, output_dir, meta)
    image_dir = os.path.join(output_dir, episode_id)

    try:
        env.reset_to_alfred_scene(meta["alfred_scene"])

        # 执行 low_actions，每步通过适配层
        low_actions = extract_low_actions(traj)
        for i, la in enumerate(low_actions):
            act, params = adapt(la["action"], la["params"])
            result = env.step(act, **params)
            step = StepRecorder.build_step(
                step_id=f"s{i}",
                branch_id="main",
                parent_step_id=f"s{i-1}" if i > 0 else None,
                step_index=i,
                action=act,
                action_params=params,
                result=result,
                image_dir=image_dir,
            )
            ep.add_step(step)

        ep.set_final_outcome({
            "main_branch": {"result": "complete", "total_steps": len(low_actions)},
            "forks": [],
        })
        return True

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="最多重放 N 条")
    parser.add_argument("--task", type=str, default="", help="按 ALFRED task_type 筛选")
    parser.add_argument("--data-dir", type=str, default="data/json_2.1.0")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--split", type=str, default="train,valid_seen,valid_unseen",
                        help="逗号分隔的分片名，默认 train,valid_seen,valid_unseen")
    args = parser.parse_args()

    # 收集所有轨迹文件
    splits = [s.strip() for s in args.split.split(",")]
    all_files = []
    for split in splits:
        pattern = os.path.join(args.data_dir, split, "**", "traj_data.json")
        found = sorted(glob.glob(pattern, recursive=True))
        print(f"  {split}: {len(found)} trajectories")
        all_files.extend(found)

    if not all_files:
        print(f"No trajectories found at {pattern}")
        print("Run scripts/download_alfred.py first.")
        sys.exit(1)

    # 筛选
    if args.task:
        filtered = []
        for f in all_files:
            t = load_traj(f)
            if t.get("task_type") == args.task:
                filtered.append(f)
        all_files = filtered
        print(f"Filtered by task_type={args.task}: {len(all_files)} trajectories")

    if args.max > 0:
        all_files = all_files[: args.max]

    print(f"Replaying {len(all_files)} trajectories into {args.output}/\n")

    ok = 0
    skip = 0
    for idx, f in enumerate(all_files):
        task_id = os.path.basename(os.path.dirname(f))
        print(f"[{idx+1}/{len(all_files)}] {task_id} ... ", end="", flush=True)

        if replay_one(f, args.output):
            ok += 1
            print("OK")
        else:
            skip += 1
            print("SKIP")

    print(f"\nDone. OK={ok}  SKIP={skip}")


if __name__ == "__main__":
    main()
