"""
Replay demo：简单验证 env_controller + step_recorder + episode_manager 全链路。

在 AI2-THOR FloorPlan1 中：前进 → 旋转 → 拿物体 → 放下 → 再旋转 → 截图全链路记录。

运行（WSL2 中）：
    cd ~/embodied-failure-dataset
    uv run python scripts/replay_demo.py
"""

import os

from src.env_controller import EnvController
from src.step_recorder import StepRecorder
from src.episode_manager import EpisodeManager


def main():
    output_dir = "output"
    episode_id = "demo_001"
    scene = "FloorPlan10"

    env = EnvController(scene=scene)
    ep = EpisodeManager(episode_id, output_dir, {
        "task_goal": "Navigate the kitchen, pick up an object, and put it down.",
        "scene": scene,
        "task_type": "pick_and_place",
        "alfred_task_id": None,
        "initial_traps": [],
    })

    image_dir = os.path.join(output_dir, episode_id)

    # 获取视野内第一个可抓取的物体
    state = env.get_state_snapshot()
    target = None
    for obj in state["objects"]:
        if obj["visible"] and obj["pickupable"] and obj["isPickedUp"] is False:
            target = obj
            break

    if not target:
        print("No visible pickupable object found. Abort.")
        env.close()
        return

    target_id = target["objectId"]
    target_type = target["objectType"]
    print(f"Target: {target_type} ({target_id})\n")

    # 动作序列：前进接近物体 → 抓 → 放下 → 转一圈
    actions = [
        ("MoveAhead", {}),
        ("PickupObject", {"objectId": target_id}),
        ("PutObject", {"objectId": target_id}),
        ("RotateLeft", {}),
        ("RotateLeft", {}),
        ("RotateLeft", {}),
        ("RotateRight", {}),
        ("MoveAhead", {}),
    ]

    step_index = 0
    for action, params in actions:
        result = env.step(action, **params)
        step = StepRecorder.build_step(
            step_id=f"s{step_index}",
            branch_id="main",
            parent_step_id=f"s{step_index-1}" if step_index > 0 else None,
            step_index=step_index,
            action=action,
            action_params=params,
            result=result,
            image_dir=image_dir,
        )
        ep.add_step(step)

        if result["success"]:
            print(f"[{step_index}] {action} -> OK")
        else:
            print(f"[{step_index}] {action} -> FAIL: {result['error']}")

        step_index += 1

    ep.set_final_outcome({
        "main_branch": {"result": "complete", "total_steps": step_index},
        "forks": [],
    })

    env.close()
    print(f"\nDone. {step_index} steps -> {output_dir}/{episode_id}.json")


if __name__ == "__main__":
    main()
