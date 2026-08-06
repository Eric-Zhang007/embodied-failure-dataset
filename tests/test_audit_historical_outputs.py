import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_historical_outputs.py"
SPEC = importlib.util.spec_from_file_location("audit_historical_outputs", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _episode(action: str, *, status=None, result=None) -> dict:
    document = {
        "episode_id": "trial_shared",
        "task_type": "pick_and_place",
        "scene": "FloorPlan1",
        "steps": [
            {
                "step_id": "main__s0",
                "branch_id": "main",
                "action": action,
                "action_params": {},
                "success": True,
            }
        ],
        "runtime_traps": [],
        "final_outcome": {"main_branch": {}, "forks": []},
    }
    if status is not None:
        document["status"] = status
    if result is not None:
        document["final_outcome"]["main_branch"]["result"] = result
    return document


def _write(root: Path, run: str, document: dict) -> None:
    path = root / run / "trial_shared.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_audit_deduplicates_copies_and_tracks_distinct_traces(tmp_path: Path) -> None:
    legacy = _episode("MoveAhead", result="complete")
    _write(tmp_path, "output", legacy)
    _write(tmp_path, "output copy", legacy)
    _write(tmp_path, "output_collector", _episode("PickupObject", status="completed"))

    report = MODULE.audit_historical_outputs(tmp_path)

    assert report["totals"]["episode_records"] == 3
    assert report["deduplicated"]["unique_json_records"] == 2
    assert report["deduplicated"]["unique_episode_ids"] == 1
    assert report["deduplicated"]["unique_branch_traces"] == 2
    assert report["runs"]["output"]["schema"] == "legacy_replay"
    assert report["runs"]["output_collector"]["schema"] == "collector"
