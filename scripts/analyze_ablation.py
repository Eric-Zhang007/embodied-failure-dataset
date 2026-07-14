"""
Ablation analysis helper.

Reads multiple output directories and compares agent behaviour across
different spike configurations. The primary data source is the episode JSON
(which records ablation_config and steps) and api_calls.jsonl (which records
every VLM call with timing).

Usage:
    uv run python scripts/analyze_ablation.py output_e2e_20260101_120000 \\
        output_e2e_20260101_130000

Output:
    - Per-config metrics table (intent repetitions, step efficiency,
      Phase 3 triggers, dedup activations, contrastive selections, etc.)
    - Side-by-side comparison of key metrics.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any


ABLATION_LABELS = {
    "001_searched_markers": "Searched Markers (001)",
    "002a_intent_dedup": "Intent Dedup (002a)",
    "002b_critic_guard": "Critic Guard (002b)",
    "003_curiosity_scoreboard": "Curiosity Scoreboard (003)",
    "004_contrastive_planner": "Contrastive Planner (004)",
    "005_progress_gating": "Progress Gating (005)",
    "006_search_trail": "Search Trail (006)",
}

SHORT_LABELS = {
    "001_searched_markers": "001",
    "002a_intent_dedup": "002a",
    "002b_critic_guard": "002b",
    "003_curiosity_scoreboard": "003",
    "004_contrastive_planner": "004",
    "005_progress_gating": "005",
    "006_search_trail": "006",
}


def load_episodes_from_dir(output_dir: str) -> list[dict]:
    """Load all episode JSON files from an output directory.

    Returns a list of (episode_data, dir_name) tuples.
    """
    episodes = []
    if not os.path.isdir(output_dir):
        print(f"WARNING: {output_dir} is not a directory, skipping")
        return episodes
    for name in os.listdir(output_dir):
        if name.endswith(".json") and name != "api_calls.jsonl":
            path = os.path.join(output_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "ablation_config" in data:
                    data["_source_dir"] = output_dir
                    episodes.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  SKIP {name}: {e}")
    return episodes


def config_signature(ablation_config: dict) -> str:
    """Build a compact signature string from an ablation config.

    Example: "001+002a+004+005+006" (spikes that are ENABLED).
    If all are enabled, returns "ALL".
    """
    enabled = []
    for key, short in SHORT_LABELS.items():
        if ablation_config.get(key, True):
            enabled.append(short)
    if len(enabled) == len(SHORT_LABELS):
        return "ALL"
    if not enabled:
        return "NONE"
    return "+".join(enabled)


def disabled_spikes(config: dict) -> list[str]:
    """Return list of disabled spike short names."""
    return [
        SHORT_LABELS[k] for k in SHORT_LABELS
        if not config.get(k, True)
    ]


def compute_episode_metrics(ep: dict) -> dict[str, Any]:
    """Extract behavioural metrics from a single episode."""
    steps = ep.get("steps", [])
    config = ep.get("ablation_config", {})

    metrics = {
        "episode_id": ep.get("episode_id", "?"),
        "task_type": ep.get("task_type", "?"),
        "termination": (ep.get("final_outcome", {})
                        .get("main_branch", {})
                        .get("termination_reason", "?")),
        "total_steps": len(steps),
        "successful_steps": sum(1 for s in steps if s.get("success", False)),
        "failed_steps": sum(1 for s in steps if not s.get("success", True)),
        "move_sequence_steps": sum(1 for s in steps if s.get("action") == "MoveSequence"),
        # Phase 3 triggers — failure diagnosis events
        "phase3_triggers": sum(
            1 for s in steps
            if s.get("eb_diagnosis") or s.get("error_type") == "environment_failure"
        ),
        # Done rejected (agent thought it was done but conditions not met)
        "done_rejected": sum(
            1 for s in steps if s.get("error_type") == "done_rejected"
        ),
        # Intent counts (from intent_history in final_outcome is not directly available;
        # approximate from intent transition heuristics)
        "distinct_actions": len(set(s.get("action", "") for s in steps)),
        # Dedup statistics from outcome
        "dedup_warnings": (ep.get("final_outcome", {})
                           .get("dedup_stats", {})
                           .get("warnings", 0)),
        "dedup_forced": (ep.get("final_outcome", {})
                          .get("dedup_stats", {})
                          .get("forced", 0)),
        "dedup_blocked_count": len((ep.get("final_outcome", {})
                                     .get("dedup_stats", {})
                                     .get("blocked_intents", []))),
    }

    # Count step-level dedup/curiosity overrides
    dedup_overrides = 0
    curiosity_overrides = 0
    for s in steps:
        if s.get("dedup_blocked"):
            dedup_overrides += 1
        if s.get("curiosity_blocked"):
            curiosity_overrides += 1
    metrics["dedup_overrides"] = dedup_overrides
    metrics["curiosity_overrides"] = curiosity_overrides

    # Step efficiency: how many steps to first successful interaction?
    first_success = None
    for i, s in enumerate(steps):
        if s.get("success") and s.get("action") not in ("LookAround", "Pass", "Done"):
            first_success = i + 1
            break
    metrics["steps_to_first_success"] = first_success

    return metrics


def aggregate_metrics(episodes: list[dict]) -> dict[str, Any]:
    """Aggregate metrics across episodes sharing the same ablation config."""
    if not episodes:
        return {}

    all_metrics = [compute_episode_metrics(ep) for ep in episodes]
    n = len(all_metrics)

    def _avg(key):
        vals = [m[key] for m in all_metrics if m[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    def _sum(key):
        return sum(m.get(key, 0) for m in all_metrics)

    return {
        "num_episodes": n,
        "total_steps_avg": _avg("total_steps"),
        "successful_steps_avg": _avg("successful_steps"),
        "failed_steps_avg": _avg("failed_steps"),
        "move_sequence_avg": _avg("move_sequence_steps"),
        "phase3_triggers_avg": _avg("phase3_triggers"),
        "phase3_triggers_total": _sum("phase3_triggers"),
        "done_rejected_avg": _avg("done_rejected"),
        "done_rejected_total": _sum("done_rejected"),
        "dedup_warnings_total": _sum("dedup_warnings"),
        "dedup_forced_total": _sum("dedup_forced"),
        "dedup_overrides_total": _sum("dedup_overrides"),
        "curiosity_overrides_total": _sum("curiosity_overrides"),
        "steps_to_first_success_avg": _avg("steps_to_first_success"),
        "terminations": [m["termination"] for m in all_metrics],
        "task_types": sorted(set(m["task_type"] for m in all_metrics)),
    }


def count_api_calls(api_calls_path: str, output_dir: str) -> dict:
    """Count VLM API calls from api_calls.jsonl, filtered to this output dir."""
    path = os.path.join(output_dir, "api_calls.jsonl")
    if not os.path.exists(path):
        # Fall back to the path directly
        path = api_calls_path
        if not os.path.exists(path):
            return {"total_calls": 0, "total_latency_s": 0.0}

    calls = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    calls.append(json.loads(line))
    except (json.JSONDecodeError, IOError):
        return {"total_calls": 0, "total_latency_s": 0.0}

    total_latency = sum(
        c.get("latency_ms", 0) for c in calls
        if isinstance(c, dict)
    )
    return {
        "total_calls": len(calls),
        "total_latency_s": total_latency / 1000.0,
        "avg_latency_s": (total_latency / len(calls) / 1000.0) if calls else 0.0,
    }


def print_comparison_table(groups: list[tuple[str, list[dict], dict]]):
    """Print a side-by-side comparison table for all ablation configs."""
    if len(groups) < 2:
        return

    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 100)

    headers = ["Metric"] + [label[:25] for label, _, _ in groups]
    rows = [
        ("Num Episodes", [str(m["num_episodes"]) for _, _, m in groups]),
        ("Avg Total Steps", [f"{m['total_steps_avg']:.1f}" for _, _, m in groups]),
        ("Avg Successful Steps", [f"{m['successful_steps_avg']:.1f}" for _, _, m in groups]),
        ("Avg Failed Steps", [f"{m['failed_steps_avg']:.1f}" for _, _, m in groups]),
        ("Avg MoveSequence", [f"{m['move_sequence_avg']:.1f}" for _, _, m in groups]),
        ("Avg Phase 3 Triggers", [f"{m['phase3_triggers_avg']:.1f}" for _, _, m in groups]),
        ("Total Phase 3 Triggers", [str(m['phase3_triggers_total']) for _, _, m in groups]),
        ("Avg Done Rejected", [f"{m['done_rejected_avg']:.1f}" for _, _, m in groups]),
        ("Total Done Rejected", [str(m['done_rejected_total']) for _, _, m in groups]),
        ("Dedup Warnings", [str(m['dedup_warnings_total']) for _, _, m in groups]),
        ("Dedup Forced", [str(m['dedup_forced_total']) for _, _, m in groups]),
        ("Dedup Overrides", [str(m['dedup_overrides_total']) for _, _, m in groups]),
        ("Curiosity Overrides", [str(m['curiosity_overrides_total']) for _, _, m in groups]),
        ("Avg Steps to First Success", [f"{m['steps_to_first_success_avg']:.1f}" for _, _, m in groups]),
    ]

    # Compute column widths
    col_widths = [max(len(r[0]) for r in rows) + 2]
    for i in range(len(groups)):
        col_widths.append(max(len(headers[i + 1]), max(len(r[1][i]) for r in rows)) + 2)

    # Print header
    header_line = "".join(
        h.ljust(col_widths[j]) if j == 0 else h.rjust(col_widths[j])
        for j, h in enumerate(headers)
    )
    print(header_line)
    print("-" * sum(col_widths))

    # Print rows
    for name, vals in rows:
        line = name.ljust(col_widths[0])
        for j, v in enumerate(vals):
            line += v.rjust(col_widths[j + 1])
        print(line)

    # Termination breakdown
    print("\n── Termination Reasons ──")
    for label, _, m in groups:
        terms = m["terminations"]
        counts = defaultdict(int)
        for t in terms:
            counts[t] += 1
        term_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        print(f"  {label[:25]}: {term_str}")

    print("\n── Task Types ──")
    for label, _, m in groups:
        print(f"  {label[:25]}: {', '.join(m['task_types'])}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    output_dirs = sys.argv[1:]

    # Load all episodes grouped by ablation config
    # key: config_signature -> list of episodes
    all_episodes: list[tuple[str, dict]] = []
    for d in output_dirs:
        episodes = load_episodes_from_dir(d)
        for ep in episodes:
            sig = config_signature(ep.get("ablation_config", {}))
            all_episodes.append((sig, ep))
        print(f"Loaded {len(episodes)} episode(s) from {d}")

    if not all_episodes:
        print("No episodes with ablation_config found.")
        sys.exit(1)

    # Group by config signature
    groups: dict[str, list[dict]] = defaultdict(list)
    for sig, ep in all_episodes:
        groups[sig].append(ep)

    # Print per-config summary
    print("\n" + "=" * 100)
    print("ABLATION ANALYSIS — Per-Config Metrics")
    print("=" * 100)

    comparison_data = []
    for sig in sorted(groups.keys()):
        eps = groups[sig]
        agg = aggregate_metrics(eps)

        # Get first config to show disabled spikes
        first_config = eps[0].get("ablation_config", {})
        disabled = disabled_spikes(first_config)
        api = count_api_calls("", eps[0].get("_source_dir", ""))

        label = f"[{sig}]"
        if disabled:
            label += f" (disabled: {','.join(disabled)})"

        print(f"\n{label}  ({agg['num_episodes']} episodes)")
        print(f"  Task types:         {', '.join(agg['task_types'])}")
        print(f"  Avg total steps:    {agg['total_steps_avg']:.1f}")
        print(f"  Avg successful:     {agg['successful_steps_avg']:.1f}")
        print(f"  Avg failed:         {agg['failed_steps_avg']:.1f}")
        print(f"  Avg MoveSequence:   {agg['move_sequence_avg']:.1f}")
        print(f"  Phase 3 triggers:   {agg['phase3_triggers_avg']:.1f} avg ({agg['phase3_triggers_total']} total)")
        print(f"  Done rejected:      {agg['done_rejected_avg']:.1f} avg ({agg['done_rejected_total']} total)")
        print(f"  Dedup warnings:     {agg['dedup_warnings_total']}")
        print(f"  Dedup forced:       {agg['dedup_forced_total']}")
        print(f"  Dedup overrides:    {agg['dedup_overrides_total']}")
        print(f"  Curiosity overrides:{agg['curiosity_overrides_total']}")
        print(f"  Steps to 1st succ:  {agg['steps_to_first_success_avg']:.1f}")
        print(f"  API calls:          {api['total_calls']} ({api['total_latency_s']:.1f}s total)")
        terms = agg["terminations"]
        term_counts = defaultdict(int)
        for t in terms:
            term_counts[t] += 1
        print(f"  Terminations:       {dict(term_counts)}")

        comparison_data.append((label, eps, agg))

    # Print side-by-side comparison
    print_comparison_table(comparison_data)

    # Print qualitative guidance
    print("\n" + "=" * 100)
    print("QUALITATIVE GUIDANCE")
    print("=" * 100)
    print("""
Key metrics to watch when comparing ablation configs:

  Phase 3 triggers:     Higher = more environment failures. Spikes that reduce
                        this are preventing the agent from making mistakes.
  Dedup overrides:      Shows how often intent dedup (002a) blocked repetition.
                        Zero when 002a is disabled.
  Curiosity overrides:  Shows how often the scoreboard (003) redirected away
                        from low-value locations. Zero when 003 is disabled.
  Steps to first succ:  Lower = faster initial progress. A spike that reduces
                        this helps the agent orient quickly.
  Done rejected:        Agent called Done prematurely. High values suggest the
                        agent is confused about task completion criteria.
  MoveSequence count:   Higher = more efficient multi-step movement. Lower may
                        indicate the agent is taking single steps.

Interpretation:
  - A spike with HIGH dedup/curiosity overrides that also shows LOWER Phase 3
    triggers and HIGHER successful steps is likely beneficial.
  - A spike that shows no overrides (i.e., never activates) may be neutral or
    the trigger conditions are too strict.
  - A spike that INCREASES Phase 3 triggers may be causing harmful overrides.

Compare side-by-side rows — larger differences suggest stronger effects.
""")


if __name__ == "__main__":
    main()
