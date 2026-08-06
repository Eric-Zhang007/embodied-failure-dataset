"""Audit a stopped EF-Bench collector directory.

The report deliberately separates raw stress-run observations from records that
meet the depth-capped benchmark contract. It uses only persisted episode JSON;
no simulator or model API is required.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SEMANTIC_ACTIONS = {
    "CleanObject",
    "CloseObject",
    "CoolObject",
    "EmptyLiquidFromObject",
    "FillObjectWithLiquid",
    "HeatObject",
    "OpenObject",
    "PickupObject",
    "PutObject",
    "SliceObject",
    "ToggleObjectOff",
    "ToggleObjectOn",
}


def _plain(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def _effective_failed_action(step: dict[str, Any] | None) -> str | None:
    if not step:
        return None
    if step.get("action") == "MoveSequence":
        trace = step.get("execution_trace")
        if isinstance(trace, list):
            for micro_action in trace:
                if isinstance(micro_action, dict) and micro_action.get("success") is False:
                    return micro_action.get("action")
    return step.get("action")


def _branch_index(steps: list[dict[str, Any]]) -> tuple[dict[str, dict], dict[str, int]]:
    roots: dict[str, dict] = {}
    for step in steps:
        metadata = step.get("fork_metadata")
        if not isinstance(metadata, dict) or not metadata.get("is_fork_root"):
            continue
        branch_id = step.get("branch_id")
        if not branch_id:
            raise ValueError("fork root is missing branch_id")
        if branch_id in roots:
            raise ValueError(f"multiple fork roots for branch {branch_id!r}")
        roots[branch_id] = step

    depths: dict[str, int] = {"main": 0}
    visiting: set[str] = set()

    def resolve(branch_id: str) -> int:
        if branch_id in depths:
            return depths[branch_id]
        if branch_id in visiting:
            raise ValueError(f"cycle in fork lineage at {branch_id!r}")
        root = roots.get(branch_id)
        if root is None:
            raise ValueError(f"fork branch {branch_id!r} has no root step")
        metadata = root["fork_metadata"]
        parent = metadata.get("fork_source_branch_id")
        if not parent:
            raise ValueError(f"fork branch {branch_id!r} has no parent branch")
        visiting.add(branch_id)
        depths[branch_id] = resolve(parent) + 1
        visiting.remove(branch_id)
        return depths[branch_id]

    for branch_id in roots:
        resolve(branch_id)
    return roots, depths


def _branch_completed_task(episode: dict[str, Any], branch_id: str) -> bool:
    outcome = episode.get("final_outcome")
    if not isinstance(outcome, dict):
        return False
    if branch_id == "main":
        main = outcome.get("main_branch")
        return isinstance(main, dict) and main.get("termination_reason") == "task_complete"
    forks = outcome.get("forks")
    if not isinstance(forks, list):
        return False
    return any(
        isinstance(fork, dict)
        and fork.get("branch_id") == branch_id
        and fork.get("termination_reason") == "task_complete"
        for fork in forks
    )


def _load_episodes(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    episodes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    episode_ids: set[str] = set()
    for path in sorted(output_dir.glob("trial_*.json")):
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"file": path.name, "error": str(exc)})
            continue
        episode_id = episode.get("episode_id") or path.stem
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id {episode_id!r}")
        episode_ids.add(episode_id)
        episode["_snapshot_file"] = path.name
        episodes.append(episode)
    return episodes, errors


def analyze_collection(output_dir: Path, max_depth: int = 3) -> dict[str, Any]:
    """Return raw, capped, strict, and integrity statistics for a snapshot."""
    output_dir = Path(output_dir)
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)

    episodes, parse_errors = _load_episodes(output_dir)
    statuses: Counter = Counter()
    task_types: Counter = Counter()
    main_terminations: Counter = Counter()
    fork_terminations: Counter = Counter()
    pending_states: Counter = Counter()
    failure_log_types: Counter = Counter()
    capped_depths: Counter = Counter()
    capped_methods: Counter = Counter()
    capped_actions: Counter = Counter()
    strict_actions: Counter = Counter()
    scenes: set[str] = set()

    raw = Counter()
    integrity = Counter()
    strict_candidates: list[dict[str, Any]] = []
    capped_candidates: list[dict[str, Any]] = []
    max_observed_depth = 0
    max_persisted_depth = 0
    failure_log_parse_errors: list[dict[str, Any]] = []

    for episode in episodes:
        episode_id = episode.get("episode_id") or episode["_snapshot_file"][:-5]
        statuses[str(episode.get("status"))] += 1
        task_type = episode.get("alfred_task_type") or episode.get("task_type") or "unknown"
        task_types[str(task_type)] += 1
        if episode.get("scene") is not None:
            scenes.add(str(episode["scene"]))

        steps = episode.get("steps") or []
        if not isinstance(steps, list):
            raise ValueError(f"{episode_id}: steps is not a list")
        step_ids = [step.get("step_id") for step in steps if step.get("step_id")]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"{episode_id}: duplicate step_id")
        by_step_id = {step.get("step_id"): step for step in steps if step.get("step_id")}
        roots, depths = _branch_index(steps)
        if depths:
            max_observed_depth = max(max_observed_depth, max(depths.values()))

        raw["steps"] += len(steps)
        raw["successful_steps"] += sum(step.get("success") is True for step in steps)
        raw["failed_steps"] += sum(step.get("success") is False for step in steps)
        raw["fork_roots_executed"] += len(roots)
        raw["nested_fork_roots_executed"] += sum(
            root["fork_metadata"].get("fork_source_branch_id") != "main"
            for root in roots.values()
        )

        for step in steps:
            image_path = step.get("image_path")
            if not image_path:
                continue
            integrity["referenced_step_images"] += 1
            resolved_image = Path(image_path)
            if not resolved_image.is_absolute():
                resolved_image = output_dir.parent / resolved_image
            if not resolved_image.exists():
                integrity["missing_step_images"] += 1

        outcome = episode.get("final_outcome")
        outcome = outcome if isinstance(outcome, dict) else {}
        main = outcome.get("main_branch")
        if isinstance(main, dict):
            reason = str(main.get("termination_reason"))
            main_terminations[reason] += 1
            raw["main_task_completions"] += reason == "task_complete"
        else:
            integrity["episodes_without_final_main"] += 1

        final_forks = outcome.get("forks")
        final_forks = final_forks if isinstance(final_forks, list) else []
        raw["finished_forks"] += len(final_forks)
        for fork in final_forks:
            if not isinstance(fork, dict):
                integrity["malformed_final_forks"] += 1
                continue
            fork_terminations[str(fork.get("termination_reason"))] += 1
            verified = fork.get("counterfactual_verified") is True
            raw["verified_forks"] += verified
            if not verified:
                continue
            branch_id = fork.get("branch_id")
            root = roots.get(branch_id)
            if root is None:
                integrity["verified_forks_without_root"] += 1
                continue
            metadata = root["fork_metadata"]
            source_id = metadata.get("replaces_step_id") or metadata.get("fork_source_step_id")
            source = by_step_id.get(source_id)
            if source is None:
                integrity["verified_forks_without_source"] += 1
                continue
            depth = depths[branch_id]
            linked_traps = [
                trap
                for trap in (episode.get("runtime_traps") or [])
                if isinstance(trap, dict) and trap.get("triggered_at_step_id") == source_id
            ]
            strict = (
                depth <= max_depth
                and source.get("success") is False
                and fork.get("counterfactual_root_feasible") is True
                and fork.get("termination_reason") == "task_complete"
            )
            if not strict:
                continue
            action = _effective_failed_action(source) or "unknown"
            semantic = action in SEMANTIC_ACTIONS
            trap_linked = bool(linked_traps)
            raw["strict_ecv_units"] += 1
            raw["strict_semantic_ecv_units"] += semantic
            raw["strict_trap_linked_semantic_ecv_units"] += semantic and trap_linked
            strict_actions[action] += 1
            strict_candidates.append(
                {
                    "episode_id": episode_id,
                    "branch_id": branch_id,
                    "depth": depth,
                    "source_step_id": source_id,
                    "source_action": source.get("action"),
                    "effective_failed_action": action,
                    "semantic": semantic,
                    "trap_linked": trap_linked,
                }
            )

        pending = episode.get("pending_forks") or []
        if isinstance(pending, list):
            raw["pending_fork_records"] += len(pending)
            for record in pending:
                if isinstance(record, dict):
                    pending_states[str(record.get("state"))] += 1
                    task = record.get("task")
                    persisted_depth = task.get("fork_depth") if isinstance(task, dict) else None
                    if isinstance(persisted_depth, int):
                        max_persisted_depth = max(max_persisted_depth, persisted_depth)

        traps = episode.get("runtime_traps") or []
        for trap in traps if isinstance(traps, list) else []:
            if not isinstance(trap, dict):
                integrity["malformed_runtime_traps"] += 1
                continue
            raw["runtime_traps"] += 1
            raw["runtime_traps_applied"] += trap.get("modification_success") is True
            triggered = bool(trap.get("triggered_at_step_id")) or trap.get("status") in {
                "triggered",
                "recovered",
            }
            raw["runtime_traps_triggered"] += triggered
            recovered = trap.get("status") == "recovered"
            raw["runtime_traps_recovered"] += recovered
            if not recovered:
                continue
            branch_id = trap.get("branch_id") or "main"
            depth = depths.get(branch_id)
            if depth is None:
                integrity["recovered_traps_without_lineage"] += 1
                continue
            if depth > max_depth:
                continue
            trigger_step = by_step_id.get(trap.get("triggered_at_step_id"))
            action = _effective_failed_action(trigger_step) or "unknown"
            semantic = action in SEMANTIC_ACTIONS
            task_complete = _branch_completed_task(episode, branch_id)
            raw["depth_capped_recovered_traps"] += 1
            raw["depth_capped_semantic_recovered_traps"] += semantic
            raw["depth_capped_task_completing_recovered_traps"] += task_complete
            raw["depth_capped_task_completing_semantic_recovered_traps"] += (
                semantic and task_complete
            )
            capped_depths[str(depth)] += 1
            method = (trap.get("injection") or {}).get("method") or "unknown"
            capped_methods[str(method)] += 1
            capped_actions[action] += 1
            capped_candidates.append(
                {
                    "episode_id": episode_id,
                    "branch_id": branch_id,
                    "depth": depth,
                    "trigger_step_id": trap.get("triggered_at_step_id"),
                    "effective_failed_action": action,
                    "mutation_method": method,
                    "semantic": semantic,
                    "task_complete": task_complete,
                }
            )

        episode_dir = output_dir / episode_id
        if episode_dir.is_dir():
            for log_path in sorted(episode_dir.glob("failures_*.jsonl")):
                for line_number, line in enumerate(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if not line.strip():
                        continue
                    raw["failure_log_events"] += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        failure_log_parse_errors.append(
                            {
                                "file": str(log_path.relative_to(output_dir)),
                                "line": line_number,
                                "error": str(exc),
                            }
                        )
                        continue
                    failure_log_types[str(record.get("failure_type"))] += 1

    png_files = list(output_dir.rglob("*.png"))
    all_files = [path for path in output_dir.rglob("*") if path.is_file()]
    total_labeled_steps = raw["successful_steps"] + raw["failed_steps"]

    return {
        "schema_version": 1,
        "source_dir": output_dir.name,
        "max_scored_depth": max_depth,
        "raw": {
            "episode_files": len(list(output_dir.glob("trial_*.json"))),
            "parsed_episodes": len(episodes),
            "statuses": _plain(statuses),
            "task_family_count": len(task_types),
            "task_families": _plain(task_types),
            "scene_count": len(scenes),
            "steps": raw["steps"],
            "successful_steps": raw["successful_steps"],
            "failed_steps": raw["failed_steps"],
            "step_success_rate": (
                raw["successful_steps"] / total_labeled_steps if total_labeled_steps else None
            ),
            "main_task_completions": raw["main_task_completions"],
            "main_terminations": _plain(main_terminations),
            "finished_forks": raw["finished_forks"],
            "verified_forks": raw["verified_forks"],
            "fork_terminations": _plain(fork_terminations),
            "fork_roots_executed": raw["fork_roots_executed"],
            "nested_fork_roots_executed": raw["nested_fork_roots_executed"],
            "max_observed_branch_depth": max_observed_depth,
            "max_persisted_fork_depth": max_persisted_depth,
            "pending_fork_records": raw["pending_fork_records"],
            "pending_states": _plain(pending_states),
            "runtime_traps": raw["runtime_traps"],
            "runtime_traps_applied": raw["runtime_traps_applied"],
            "runtime_traps_triggered": raw["runtime_traps_triggered"],
            "runtime_traps_recovered": raw["runtime_traps_recovered"],
            "failure_log_events": raw["failure_log_events"],
            "failure_log_types": _plain(failure_log_types),
            "png_files": len(png_files),
            "all_files": len(all_files),
            "bytes": sum(path.stat().st_size for path in all_files),
        },
        "depth_capped": {
            "max_depth": max_depth,
            "recovered_traps": raw["depth_capped_recovered_traps"],
            "semantic_recovered_traps": raw["depth_capped_semantic_recovered_traps"],
            "task_completing_recovered_traps": raw[
                "depth_capped_task_completing_recovered_traps"
            ],
            "task_completing_semantic_recovered_traps": raw[
                "depth_capped_task_completing_semantic_recovered_traps"
            ],
            "by_depth": _plain(capped_depths),
            "by_mutation_method": _plain(capped_methods),
            "by_effective_failed_action": _plain(capped_actions),
            "candidates": capped_candidates,
        },
        "strict_ecv": {
            "definition": (
                "verified and root-feasible fork that replaces a failed source step, "
                "finishes the task, and has inferred depth at most max_scored_depth"
            ),
            "units": raw["strict_ecv_units"],
            "semantic_units": raw["strict_semantic_ecv_units"],
            "trap_linked_semantic_units": raw[
                "strict_trap_linked_semantic_ecv_units"
            ],
            "by_effective_failed_action": _plain(strict_actions),
            "candidates": strict_candidates,
        },
        "integrity": {
            "episode_parse_errors": parse_errors,
            "failure_log_parse_errors": failure_log_parse_errors,
            "episodes_without_final_main": integrity["episodes_without_final_main"],
            "verified_forks_without_root": integrity["verified_forks_without_root"],
            "verified_forks_without_source": integrity["verified_forks_without_source"],
            "recovered_traps_without_lineage": integrity[
                "recovered_traps_without_lineage"
            ],
            "referenced_step_images": integrity["referenced_step_images"],
            "missing_step_images": integrity["missing_step_images"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = analyze_collection(args.output_dir, max_depth=args.max_depth)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
