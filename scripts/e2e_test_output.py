#!/usr/bin/env python3
"""
E2E test using local fine-tuned 8B + SiliconFlow 32B Oracle.
Reads episode data directly from output/ JSONs (no ALFRED data needed).
"""
import argparse, glob, json, os, sys

from src.vlm_client import VLMClient
from src.eb_agent import EBAgent
from src.oracle_agent import OracleAgent
from src.branch_runner import run_single_branch
from src.trap_planner import TrapPlanner

LOCAL_EB_URL = "http://localhost:8000/v1"


def ep_to_traj(ep):
    """Convert output episode JSON to ALFRED-like trajectory dict."""
    scene_state = ep.get("alfred_scene")
    if not scene_state:
        raise ValueError("Episode JSON lacks alfred_scene; regenerate it from ALFRED data first")
    return {
        "task_type": ep["task_type"],
        "task_id": ep.get("alfred_task_id", ep["episode_id"]),
        "pddl_params": ep.get("pddl_params", {}),
        "scene": scene_state,
        "turk_annotations": {"anns": [{"task_desc": ep["task_goal"]}]},
        "plan": {"low_actions": [{"api_action": {"action": "TeleportFull", "params": {"x": 0, "y": 0.9, "z": 0, "rotation": 0}}, "high_idx": 0}]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--eb-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--oracle-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--task", default="pick_and_place_simple")
    parser.add_argument("--output", default="output_e2e")
    parser.add_argument("--no-traps", action="store_true")
    args = parser.parse_args()

    # Load from output/ JSONs directly
    ep_files = sorted(glob.glob("output/trial_*.json"))
    if args.task:
        filtered = []
        for f in ep_files:
            with open(f) as fp: ep = json.load(fp)
            if ep.get("task_type") == args.task:
                filtered.append(f)
        ep_files = filtered

    if not ep_files:
        print(f"No episodes for task_type={args.task}")
        sys.exit(1)

    ep_file = ep_files[0]
    with open(ep_file) as f:
        ep = json.load(f)

    print(f"Episode: {ep['episode_id']}")
    print(f"Task: {ep['task_goal']}")
    print(f"Scene: {ep['scene']}\n")

    # Create fake traj_path pointing to the JSON
    traj_path = os.path.abspath(ep_file)

    # Patch alfred_parser to load from our JSON
    import src.alfred_parser as ap
    _orig_load = ap.load_traj
    ap.load_traj = lambda p: ep_to_traj(json.load(open(p)))

    # EB agent: support local model
    if args.eb_model == "local":
        print(f"Using local EB model at {LOCAL_EB_URL}")
        eb_client = VLMClient("openai", model="local", base_url=LOCAL_EB_URL, api_key="none")
    else:
        eb_client = VLMClient.siliconflow(args.eb_model, args.api_key)

    oracle_client = VLMClient.siliconflow(args.oracle_model, args.api_key)
    eb_agent = EBAgent(eb_client)
    oracle_agent = OracleAgent(oracle_client)

    import time as _time
    print("Warming up Oracle model...", end=" ", flush=True)
    _t0 = _time.time()
    oracle_client.chat_text(system_prompt="Say OK.", user_text="OK", max_tokens=5)
    print(f"done ({_time.time() - _t0:.1f}s)\n")

    trap_planner = None if args.no_traps else TrapPlanner()
    result = run_single_branch(
        traj_path=traj_path, eb_agent=eb_agent, oracle_agent=oracle_agent,
        output_dir=args.output, trap_planner=trap_planner,
        enable_fork=False, step_limit_multiplier=2,
    )

    print(f"\n{'='*50}")
    print(f"Branch: {result.branch_id}")
    print(f"Termination: {result.termination_reason}")
    print(f"Total steps: {result.total_steps}")
    print(f"Fork tasks: {len(result.fork_tasks)}")


if __name__ == "__main__":
    main()
