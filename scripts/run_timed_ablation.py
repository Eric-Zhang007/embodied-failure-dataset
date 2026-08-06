"""Run matched embodied-agent ablations for a fixed wall-clock budget."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "full": [],
    "geometric_memory": ["--memory", "geometric"],
    "no_intent_dedup": ["--ablation", "002a"],
    "no_curiosity": ["--ablation", "003"],
    "no_progress_gate": ["--ablation", "005"],
    "no_search_trail": ["--ablation", "006"],
}
TASKS = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
]
INITIAL_TRAJECTORY_INDEX = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 0,
    "pick_clean_then_place_in_recep": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=int, default=3 * 60 * 60)
    parser.add_argument("--task-budget-seconds", type=int, default=55 * 60)
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def api_base_url(config_path: Path) -> str:
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    value = config.get("api", "base_url").strip("\"'").rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def stop_processes(processes: dict[str, tuple[subprocess.Popen, object]]) -> None:
    live = [process for process, _ in processes.values() if process.poll() is None]
    for process in live:
        os.killpg(process.pid, signal.SIGINT)

    for timeout, final_signal in [(60, signal.SIGTERM), (30, signal.SIGKILL)]:
        deadline = time.monotonic() + timeout
        while any(process.poll() is None for process in live) and time.monotonic() < deadline:
            time.sleep(1)
        for process in live:
            if process.poll() is None:
                os.killpg(process.pid, final_signal)

    for process in live:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    output_root = args.output_root or ROOT / f"output_ablation_pilot_{started_at:%Y%m%d_%H%M%S}"
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_command = [
        sys.executable,
        str(ROOT / "scripts" / "e2e_test.py"),
        "--config",
        str(args.config.resolve()),
        "--api-base-url",
        api_base_url(args.config),
        "--enable-fork",
    ]
    manifest = {
        "started_at_utc": started_at.isoformat(),
        "duration_seconds": args.duration_seconds,
        "task_budget_seconds": args.task_budget_seconds,
        "conditions": list(CONDITIONS),
        "task_plan": TASKS,
        "initial_trajectory_index": INITIAL_TRAJECTORY_INDEX,
        "task_selection": "matched deterministic trajectory index across conditions",
        "parallelism": "six conditions in parallel; task types scheduled sequentially",
        "forks_enabled": True,
        "stages_started": 0,
        "status": "dry_run" if args.dry_run else "running",
    }
    manifest_path = output_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        for task in TASKS:
            for condition, extra_args in CONDITIONS.items():
                print(task, condition, " ".join(base_command + ["--task", task] + extra_args))
        return 0

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    deadline = time.monotonic() + args.duration_seconds
    stage_number = 0
    task_trajectory_indices = dict(INITIAL_TRAJECTORY_INDEX)

    try:
        while not stop_requested and time.monotonic() < deadline:
            task = TASKS[stage_number % len(TASKS)]
            trajectory_index = task_trajectory_indices[task]
            task_trajectory_indices[task] += 1
            stage_number += 1
            manifest["stages_started"] = stage_number
            manifest["current_task"] = task
            manifest["current_trajectory_index"] = trajectory_index
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            processes: dict[str, tuple[subprocess.Popen, object]] = {}

            for condition, extra_args in CONDITIONS.items():
                stage_name = f"stage_{stage_number:02d}_{task}_traj_{trajectory_index:03d}"
                run_dir = output_root / stage_name / condition
                run_dir.mkdir(parents=True, exist_ok=True)
                log_handle = (run_dir / "run.log").open("w", encoding="utf-8")
                command = base_command + [
                    "--task",
                    task,
                    "--trajectory-index",
                    str(trajectory_index),
                    "--output",
                    str(run_dir),
                ] + extra_args
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                processes[condition] = (process, log_handle)

            stage_deadline = min(deadline, time.monotonic() + args.task_budget_seconds)
            while not stop_requested and time.monotonic() < stage_deadline:
                if all(process.poll() is not None for process, _ in processes.values()):
                    break
                time.sleep(5)

            if stop_requested or time.monotonic() >= stage_deadline:
                stop_processes(processes)

            for process, log_handle in processes.values():
                if process.poll() is None:
                    stop_processes({"remaining": (process, log_handle)})
                log_handle.close()

    finally:
        manifest["status"] = "stopped" if stop_requested else "time_budget_complete"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
