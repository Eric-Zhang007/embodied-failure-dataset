import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_collection_snapshot import analyze_collection


def _root_step(branch_id, parent_branch_id, source_step_id, *, success=True):
    return {
        "step_id": f"{branch_id}__s0",
        "branch_id": branch_id,
        "action": "MoveAhead",
        "success": success,
        "error_message": None if success else "blocked",
        "fork_metadata": {
            "is_fork_root": True,
            "fork_source_branch_id": parent_branch_id,
            "fork_source_step_id": source_step_id,
            "replaces_step_id": source_step_id,
        },
    }


class CollectionSnapshotAnalysisTest(unittest.TestCase):
    def test_separates_raw_depth_from_capped_strict_semantic_units(self):
        source = {
            "step_id": "main__s0",
            "branch_id": "main",
            "action": "MoveSequence",
            "success": False,
            "error_message": "Target object not found within visibility",
            "execution_trace": [
                {"action": "MoveAhead", "success": True},
                {"action": "PickupObject", "success": False},
            ],
        }
        depth_1 = _root_step("fork_s0_main", "main", "main__s0")
        depth_2 = _root_step(
            "fork_s0_fork_s0_main", "fork_s0_main", "fork_s0_main__s0"
        )
        depth_3 = _root_step(
            "fork_s0_fork_s0_fork_s0_main",
            "fork_s0_fork_s0_main",
            "fork_s0_fork_s0_main__s0",
            success=False,
        )
        depth_4 = _root_step(
            "fork_s0_fork_s0_fork_s0_fork_s0_main",
            "fork_s0_fork_s0_fork_s0_main",
            "fork_s0_fork_s0_fork_s0_main__s0",
        )
        episode = {
            "episode_id": "trial_demo",
            "scene": "FloorPlan1",
            "alfred_task_type": "pick_and_place_simple",
            "status": "completed",
            "steps": [source, depth_1, depth_2, depth_3, depth_4],
            "runtime_traps": [
                {
                    "branch_id": "main",
                    "status": "recovered",
                    "modification_success": True,
                    "triggered_at_step_id": "main__s0",
                    "injection": {"method": "hide_object"},
                }
            ],
            "pending_forks": [
                {
                    "branch_id": "fork_s0_fork_s0_fork_s0_fork_s0_main",
                    "state": "finished",
                    "task": {"fork_depth": 4},
                }
            ],
            "final_outcome": {
                "main_branch": {"termination_reason": "task_complete"},
                "forks": [
                    {
                        "branch_id": "fork_s0_main",
                        "termination_reason": "task_complete",
                        "counterfactual_root_feasible": True,
                        "counterfactual_verified": True,
                    },
                    {
                        "branch_id": "fork_s0_fork_s0_fork_s0_fork_s0_main",
                        "termination_reason": "task_complete",
                        "counterfactual_root_feasible": True,
                        "counterfactual_verified": True,
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trial_demo.json"
            path.write_text(json.dumps(episode), encoding="utf-8")
            report = analyze_collection(Path(tmp), max_depth=3)

        self.assertEqual(4, report["raw"]["max_observed_branch_depth"])
        self.assertEqual(4, report["raw"]["max_persisted_fork_depth"])
        self.assertEqual(2, report["raw"]["verified_forks"])
        self.assertEqual(1, report["depth_capped"]["recovered_traps"])
        self.assertEqual(1, report["depth_capped"]["semantic_recovered_traps"])
        self.assertEqual(1, report["strict_ecv"]["units"])
        self.assertEqual(1, report["strict_ecv"]["semantic_units"])
        self.assertEqual(1, report["strict_ecv"]["trap_linked_semantic_units"])
        self.assertEqual(
            {"PickupObject": 1},
            report["strict_ecv"]["by_effective_failed_action"],
        )


if __name__ == "__main__":
    unittest.main()
