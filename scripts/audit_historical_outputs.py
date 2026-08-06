"""Inventory and deduplicate historical output directories without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path


_ANALYZER_PATH = Path(__file__).with_name("analyze_collection_snapshot.py")
_ANALYZER_SPEC = importlib.util.spec_from_file_location("collection_snapshot_analyzer", _ANALYZER_PATH)
_ANALYZER = importlib.util.module_from_spec(_ANALYZER_SPEC)
assert _ANALYZER_SPEC.loader is not None
_ANALYZER_SPEC.loader.exec_module(_ANALYZER)


_NAVIGATION_ACTIONS = {
    "LookDown",
    "LookUp",
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "Pass",
    "RotateLeft",
    "RotateRight",
    "TeleportFull",
}


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _trace_hash(episode_id: str, branch_id: str, steps: list[dict]) -> str:
    trace = {
        "episode_id": episode_id,
        "branch_id": branch_id,
        "steps": [
            {
                "action": step.get("action"),
                "params": step.get("action_params") or {},
                "success": bool(step.get("success")),
                "error": step.get("error_message") or step.get("error") or "",
            }
            for step in steps
        ],
    }
    encoded = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hash_bytes(encoded)


def _counter(counter: Counter) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def audit_historical_outputs(root: Path) -> dict:
    root = Path(root)
    runs: dict[str, dict] = {}
    global_json_hashes: set[str] = set()
    global_episode_ids: set[str] = set()
    global_trace_hashes: set[str] = set()
    global_capped_recovery_traces: set[str] = set()
    global_strict_ecv_traces: set[str] = set()
    episode_occurrences: Counter = Counter()
    totals = Counter()

    for run_dir in sorted(path for path in root.glob("output*") if path.is_dir()):
        episode_files = sorted(run_dir.glob("trial_*.json"))
        if not episode_files:
            continue

        statuses: Counter = Counter()
        legacy_results: Counter = Counter()
        main_terminations: Counter = Counter()
        task_types: Counter = Counter()
        trap_statuses: Counter = Counter()
        scenes: set[str] = set()
        run_hashes: set[str] = set()
        run_trace_hashes: set[str] = set()
        parse_errors: list[str] = []
        steps_total = 0
        failed_steps = 0
        failed_interactions = 0
        runtime_traps = 0
        finished_forks = 0
        verified_forks = 0
        schema_signals: set[str] = set()
        trace_by_branch: dict[tuple[str, str], str] = {}

        for path in episode_files:
            payload = path.read_bytes()
            digest = _hash_bytes(payload)
            run_hashes.add(digest)
            global_json_hashes.add(digest)
            try:
                episode = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                parse_errors.append(f"{path.name}: {error}")
                continue

            episode_id = str(episode.get("episode_id") or path.stem)
            global_episode_ids.add(episode_id)
            episode_occurrences[episode_id] += 1
            task_types[episode.get("task_type") or episode.get("alfred_task_type") or "missing"] += 1
            if episode.get("scene"):
                scenes.add(str(episode["scene"]))

            if "status" in episode:
                schema_signals.add("collector")
                statuses[episode.get("status") or "missing"] += 1
            else:
                schema_signals.add("legacy_replay")

            outcome = episode.get("final_outcome") or {}
            main = outcome.get("main_branch") or {}
            if "result" in main:
                legacy_results[main.get("result") or "missing"] += 1
            if "termination_reason" in main:
                main_terminations[main.get("termination_reason") or "missing"] += 1

            forks = outcome.get("forks") or []
            finished_forks += len(forks)
            verified_forks += sum(
                bool(fork.get("counterfactual_verified"))
                for fork in forks
            )

            traps = episode.get("runtime_traps") or []
            runtime_traps += len(traps)
            trap_statuses.update(trap.get("status") or "missing" for trap in traps)

            branches: dict[str, list[dict]] = defaultdict(list)
            steps = episode.get("steps") or []
            steps_total += len(steps)
            for step in steps:
                branches[str(step.get("branch_id") or "main")].append(step)
                if not step.get("success"):
                    failed_steps += 1
                    if step.get("action") not in _NAVIGATION_ACTIONS:
                        failed_interactions += 1
            for branch_id, branch_steps in branches.items():
                digest = _trace_hash(episode_id, branch_id, branch_steps)
                run_trace_hashes.add(digest)
                global_trace_hashes.add(digest)
                trace_by_branch[(episode_id, branch_id)] = digest

        schema = "mixed" if len(schema_signals) > 1 else next(iter(schema_signals), "unknown")
        parsed = len(episode_files) - len(parse_errors)
        collection_audit = None
        collection_audit_error = None
        if schema in {"collector", "mixed"} and not parse_errors:
            try:
                collection_audit = _ANALYZER.analyze_collection(run_dir, max_depth=3)
            except (OSError, TypeError, ValueError) as error:
                collection_audit_error = str(error)

        capped_recovered = 0
        capped_semantic = 0
        strict_ecv = 0
        strict_semantic = 0
        if collection_audit is not None:
            capped = collection_audit["depth_capped"]
            strict = collection_audit["strict_ecv"]
            capped_recovered = capped["recovered_traps"]
            capped_semantic = capped["semantic_recovered_traps"]
            strict_ecv = strict["units"]
            strict_semantic = strict["semantic_units"]
            for candidate in capped["candidates"]:
                digest = trace_by_branch.get((candidate["episode_id"], candidate["branch_id"]))
                if digest:
                    global_capped_recovery_traces.add(digest)
            for candidate in strict["candidates"]:
                digest = trace_by_branch.get((candidate["episode_id"], candidate["branch_id"]))
                if digest:
                    global_strict_ecv_traces.add(digest)

        runs[run_dir.name] = {
            "schema": schema,
            "episode_records": len(episode_files),
            "parsed_records": parsed,
            "parse_errors": parse_errors,
            "unique_json_records": len(run_hashes),
            "unique_branch_traces": len(run_trace_hashes),
            "task_types": _counter(task_types),
            "scene_count": len(scenes),
            "statuses": _counter(statuses),
            "legacy_results": _counter(legacy_results),
            "main_terminations": _counter(main_terminations),
            "steps": steps_total,
            "failed_steps": failed_steps,
            "failed_interaction_steps": failed_interactions,
            "runtime_traps": runtime_traps,
            "trap_statuses": _counter(trap_statuses),
            "finished_forks": finished_forks,
            "verified_forks": verified_forks,
            "depth_capped_recovered": capped_recovered,
            "depth_capped_semantic": capped_semantic,
            "strict_ecv": strict_ecv,
            "strict_ecv_semantic": strict_semantic,
            "collection_audit_error": collection_audit_error,
        }
        totals.update({
            "runs": 1,
            "episode_records": len(episode_files),
            "parsed_records": parsed,
            "parse_errors": len(parse_errors),
            "steps": steps_total,
            "failed_steps": failed_steps,
            "failed_interaction_steps": failed_interactions,
            "runtime_traps": runtime_traps,
            "finished_forks": finished_forks,
            "verified_forks": verified_forks,
            "depth_capped_recovered": capped_recovered,
            "depth_capped_semantic": capped_semantic,
            "strict_ecv": strict_ecv,
            "strict_ecv_semantic": strict_semantic,
        })

    return {
        "schema_version": 1,
        "root": str(root),
        "totals": dict(totals),
        "deduplicated": {
            "unique_json_records": len(global_json_hashes),
            "unique_episode_ids": len(global_episode_ids),
            "unique_branch_traces": len(global_trace_hashes),
            "episode_ids_repeated_across_runs": sum(
                count > 1 for count in episode_occurrences.values()
            ),
            "exact_duplicate_json_records": totals["parsed_records"] - len(global_json_hashes),
            "unique_depth_capped_recovery_traces": len(global_capped_recovery_traces),
            "unique_strict_ecv_traces": len(global_strict_ecv_traces),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_historical_outputs(args.root)
    encoded = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
