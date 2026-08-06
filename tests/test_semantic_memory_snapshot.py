import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.branch_runner import (
    BranchRunner,
    _check_and_mark_searched,
    _sync_memory_after_injection,
    replay_steps,
)
from src.episode_manager import EpisodeManager
from src.geometric_memory import StuckTracker
from src.semantic_memory import SemanticMemory


def _metadata(inventory=None):
    return {
        "agent": {"position": {"x": 0.0, "z": 0.0}, "rotation": {"y": 90.0}},
        "inventoryObjects": inventory or [],
        "objects": [
            {
                "objectId": "Cabinet|1", "objectType": "Cabinet",
                "receptacle": True, "visibleBounds2D": [0, 0, 10, 10],
                "position": {"x": 0.0, "z": 1.0},
            },
            {
                "objectId": "Apple|1", "objectType": "Apple", "pickupable": True,
                "visibleBounds2D": [0, 0, 5, 5], "position": {"x": 0.0, "z": 1.0},
            },
        ],
    }


class SemanticMemorySnapshotTest(unittest.TestCase):
    def test_approaching_a_receptacle_does_not_mark_all_instances_searched(self):
        metadata = _metadata()
        metadata["objects"] = [
            {
                "objectId": "Drawer|1", "objectType": "Drawer",
                "receptacle": True, "visibleBounds2D": [0, 0, 10, 10],
                "position": {"x": 0.0, "z": 1.0},
            },
            {
                "objectId": "Drawer|2", "objectType": "Drawer",
                "receptacle": True, "visibleBounds2D": [10, 0, 20, 10],
                "position": {"x": 1.0, "z": 1.0},
            },
        ]
        memory = SemanticMemory()
        memory.update(metadata, metadata["objects"], "LookAround", True, None, "")

        _check_and_mark_searched(
            {
                "intent": "approach Drawer",
                "target": "Drawer",
                "completed": True,
                "steps": [{
                    "action": "MoveSequence",
                    "action_params": {"steps": [{"action": "MoveAhead"}]},
                }],
            },
            "AlarmClock",
            memory,
        )

        state = memory.export_state()["objects"]
        self.assertFalse(state["Drawer|1"]["searched"])
        self.assertFalse(state["Drawer|2"]["searched"])
        self.assertEqual(1, memory.receptacle_visit_counts["Drawer"])

    def test_unidentified_receptacle_interaction_does_not_mark_all_instances(self):
        metadata = _metadata()
        metadata["objects"] = [
            {
                "objectId": "Drawer|1", "objectType": "Drawer",
                "receptacle": True, "visibleBounds2D": [0, 0, 10, 10],
                "position": {"x": 0.0, "z": 1.0},
            },
            {
                "objectId": "Drawer|2", "objectType": "Drawer",
                "receptacle": True, "visibleBounds2D": [10, 0, 20, 10],
                "position": {"x": 1.0, "z": 1.0},
            },
        ]
        memory = SemanticMemory()
        memory.update(metadata, metadata["objects"], "LookAround", True, None, "")

        _check_and_mark_searched(
            {
                "intent": "open Drawer",
                "target": "Drawer",
                "completed": True,
                "steps": [{
                    "action": "MoveSequence",
                    "action_params": {
                        "steps": [{"action": "OpenObject", "params": {"objectType": "Drawer"}}],
                    },
                }],
            },
            "AlarmClock",
            memory,
        )

        state = memory.export_state()["objects"]
        self.assertFalse(state["Drawer|1"]["searched"])
        self.assertFalse(state["Drawer|2"]["searched"])
        self.assertEqual(1, memory.receptacle_open_counts["Drawer"])
        self.assertEqual(1, memory.receptacle_visit_counts["Drawer"])

    def test_three_collisions_with_one_obstacle_trigger_global_stuck_guard(self):
        tracker = StuckTracker()
        for step, target in enumerate(("AlarmClock", "Desk", "DeskLamp"), start=1):
            tracker.update(
                step=step, intent="approach", target=target, success=False,
                blocked_by="Floor_a4e03a2b", blocked_dir="ahead", action="MoveAhead",
            )

        self.assertEqual("Floor_a4e03a2b", tracker.repeated_obstacle())

    def test_blocker_identity_uses_the_unity_object_id_from_sequence_errors(self):
        memory = SemanticMemory()

        blocker_id = memory._parse_blocker_id(
            "MoveSequence: first step MoveRight failed immediately: "
            "Floor_a4e03a2b is blocking Agent 0 from moving"
        )

        self.assertEqual("Floor_a4e03a2b", blocker_id)

    def test_hide_object_injection_moves_target_memory_to_drawer(self):
        metadata = {
            "agent": {"position": {"x": 0.0, "z": 0.0}, "rotation": {"y": 0.0}},
            "inventoryObjects": [],
            "objects": [
                {
                    "objectId": "SideTable|1", "objectType": "SideTable",
                    "receptacle": True, "visibleBounds2D": [0, 0, 10, 10],
                    "position": {"x": 0.0, "z": 1.0},
                },
                {
                    "objectId": "Drawer|1", "objectType": "Drawer",
                    "receptacle": True, "visibleBounds2D": [10, 0, 20, 10],
                    "position": {"x": 0.1, "z": 1.0},
                },
                {
                    "objectId": "AlarmClock|1", "objectType": "AlarmClock",
                    "pickupable": True, "visibleBounds2D": [2, 2, 4, 4],
                    "position": {"x": 0.0, "z": 1.0},
                    "parentReceptacles": ["SideTable|1"],
                },
            ],
        }
        memory = SemanticMemory()
        memory.update(
            metadata, metadata["objects"], "LookAround", True, None,
            "Pick up the AlarmClock and put it on the Desk.",
        )

        _sync_memory_after_injection(memory, {
            "modification_success": True,
            "injection": {
                "method": "hide_object",
                "params": {
                    "object_id": "AlarmClock|1",
                    "container_id": "Drawer|1",
                },
            },
        })

        target = memory.export_state()["objects"]["AlarmClock|1"]
        self.assertEqual("Drawer|1", target["parent_receptacle_id"])
        self.assertEqual("remembered", target["status"])
        self.assertIn("AlarmClock \u2014 last seen fresh at Drawer.", memory.render())
        self.assertNotIn("inside SideTable", memory.render())

    def test_snapshot_round_trip_preserves_semantic_state(self):
        memory = SemanticMemory()
        metadata = _metadata()
        memory.update(metadata, metadata["objects"], "LookAround", True, None,
                      "Apple must be inside Cabinet")
        memory.mark_searched(object_type="Cabinet")
        memory.record_receptacle_open("Cabinet")

        snapshot = memory.export_state()
        restored = SemanticMemory.from_state(json.loads(json.dumps(snapshot)))

        self.assertEqual(snapshot, restored.export_state())

    def test_aging_keeps_goal_and_prunes_stale_non_goal_state(self):
        memory = SemanticMemory()
        metadata = _metadata()
        metadata["objects"].append({
            "objectId": "Mug|1", "objectType": "Mug", "pickupable": True,
            "isToggled": False, "visibleBounds2D": [20, 0, 25, 5],
            "position": {"x": 1.0, "z": 1.0},
        })
        criteria = "Apple must be inside Cabinet"
        memory.update(metadata, metadata["objects"], "LookAround", True, None, criteria)
        memory._objects["Mug|1"].status = "placed"

        for _ in range(memory.AGING_THRESHOLD):
            memory.update(metadata, [], "RotateLeft", True, None, criteria)

        self.assertTrue(memory.has_type("Apple"))
        self.assertTrue(memory.has_type("Cabinet"))
        self.assertNotIn("Mug|1", memory._objects)
        self.assertNotIn("Mug|1", memory._object_state)

    def test_put_keeps_only_the_previously_held_duplicate_as_placed(self):
        memory = SemanticMemory()
        metadata = _metadata([{"objectId": "Apple|1", "objectType": "Apple"}])
        metadata["objects"].append({
            "objectId": "Apple|2", "objectType": "Apple", "pickupable": True,
            "visibleBounds2D": [20, 0, 25, 5], "position": {"x": 1.0, "z": 1.0},
        })
        criteria = "Apple must be inside Cabinet"
        memory.update(metadata, metadata["objects"], "PickupObject", True, None, criteria)

        metadata["inventoryObjects"] = []
        memory.update(metadata, metadata["objects"], "PutObject", True, None, criteria)

        self.assertEqual("placed", memory._objects["Apple|1"].status)
        self.assertEqual("visible", memory._objects["Apple|2"].status)

    def test_resume_uses_saved_snapshot_instead_of_rebuilding_memory(self):
        memory = SemanticMemory()
        metadata = _metadata()
        memory.update(metadata, metadata["objects"], "MoveAhead", False,
                      "Cabinet is blocking movement.", "Apple must be inside Cabinet")

        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager("ep", output_dir, {
                "task_goal": "put apple in cabinet", "scene": "FloorPlan1",
                "task_type": "pick_and_place_simple",
                "alfred_scene": {"object_poses": [{"objectName": "Apple_1"}]},
            })
            step = {
                "step_id": "main__s0", "branch_id": "main", "step_index_in_branch": 0,
                "action": "MoveAhead", "action_params": {}, "success": False,
            }
            episode.add_step(step)
            episode.set_semantic_memory_state("main", memory.export_state())
            env = MagicMock()
            env.get_state_snapshot.return_value = {"metadata": metadata}

            with patch("src.branch_runner.EnvController", return_value=env), \
                 patch("src.branch_runner.replay_steps") as replay, \
                 patch.object(BranchRunner, "run", return_value="resumed") as run:
                self.assertEqual("resumed", BranchRunner.resume(
                    episode.file_path, "main", object(), object(), memory_mode="semantic",
                ))

        replay.assert_called_once_with(
            env, [step], skip_failed=True, pddl_params={},
        )
        self.assertEqual(memory.export_state(), run.call_args.kwargs["_memory"].export_state())
        env.close.assert_called_once()

    def test_resume_does_not_restore_a_hand_emptied_by_an_injection(self):
        memory = SemanticMemory()
        metadata = _metadata()
        memory.update(metadata, metadata["objects"], "PickupObject", True, None,
                      "Apple must be inside Cabinet")

        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager("ep", output_dir, {
                "task_goal": "put apple in cabinet", "scene": "FloorPlan1",
                "task_type": "pick_and_place_simple",
                "alfred_scene": {"object_poses": [{"objectName": "Apple_1"}]},
            })
            episode.add_step({
                "step_id": "main__s0", "branch_id": "main", "step_index_in_branch": 0,
                "action": "PickupObject", "action_params": {"objectType": "Apple"}, "success": True,
                "oracle_injection_decision": {
                    "decided_to_inject": True,
                    "modification_success": True,
                    "injection": {"method": "drop_held_object", "params": {}},
                },
            })
            episode.set_semantic_memory_state("main", memory.export_state())
            env = MagicMock()
            env.get_state_snapshot.return_value = {"metadata": metadata}

            with patch("src.branch_runner.EnvController", return_value=env), \
                 patch("src.branch_runner.replay_steps"), \
                 patch.object(BranchRunner, "run", return_value="resumed"):
                BranchRunner.resume(
                    episode.file_path, "main", object(), object(), memory_mode="semantic",
                )

        env.step.assert_not_called()

    def test_replay_applies_saved_injection_before_its_action(self):
        calls = []

        class ReplayEnv:
            controller = object()

            def step(self, action, **params):
                calls.append(("action", action, params))
                return {"success": True, "error": None}

        def replay_injection(controller, method, pddl_params, **params):
            calls.append(("injection", method, pddl_params, params))
            return {"success": True, "error": None}

        step = {
            "step_id": "main__s0", "action": "MoveAhead", "action_params": {}, "success": True,
            "oracle_injection_decision": {
                "decided_to_inject": True,
                "modification_success": True,
                "injection": {"method": "close_container", "params": {"object_type": "Cabinet"}},
            },
        }
        with patch("src.branch_runner.inject", side_effect=replay_injection):
            replay_steps(ReplayEnv(), [step], pddl_params={"parent_target": "Table"})

        self.assertEqual("injection", calls[0][0])
        self.assertEqual("action", calls[1][0])

    def test_replay_expands_each_repeated_movement_action(self):
        calls = []

        class ReplayEnv:
            controller = SimpleNamespace(
                last_event=SimpleNamespace(metadata={"objects": []}),
            )

            def step(self, action, **params):
                calls.append((action, params))
                return {"success": True, "error": None}

        replay_steps(ReplayEnv(), [{
            "step_id": "main__s0",
            "action": "MoveSequence",
            "action_params": {
                "steps": [
                    {"action": "MoveAhead", "repeat": 3},
                    {"action": "RotateRight"},
                ],
            },
            "success": True,
        }])

        self.assertEqual(
            ["MoveAhead", "MoveAhead", "MoveAhead", "RotateRight"],
            [action for action, _params in calls],
        )

    def test_replay_uses_recorded_successful_micro_action_prefix(self):
        calls = []

        class ReplayEnv:
            controller = SimpleNamespace(
                last_event=SimpleNamespace(metadata={"objects": []}),
            )

            def step(self, action, **params):
                calls.append((action, params))
                return {"success": True, "error": None}

        replay_steps(ReplayEnv(), [{
            "step_id": "main__s0",
            "action": "MoveSequence",
            "action_params": {
                "steps": [{"action": "MoveAhead", "repeat": 5}],
            },
            "success": False,
            "execution_trace": [
                {"action": "MoveAhead", "params": {}, "success": True, "error": None},
                {"action": "MoveAhead", "params": {}, "success": False, "error": "blocked"},
            ],
        }])

        self.assertEqual([("MoveAhead", {})], calls)

    def test_replay_preserves_post_scan_facing_rotation(self):
        calls = []

        class ReplayEnv:
            controller = SimpleNamespace(
                last_event=SimpleNamespace(metadata={"objects": []}),
            )

            def step(self, action, **params):
                calls.append((action, params))
                return {"success": True, "error": None}

        replay_steps(ReplayEnv(), [{
            "step_id": "main__s0",
            "action": "LookAround",
            "action_params": {},
            "success": True,
            "post_scan_actions": [
                {"action": "RotateLeft", "action_params": {}},
                {"action": "RotateLeft", "action_params": {}},
                {"action": "RotateLeft", "action_params": {}},
            ],
        }])

        self.assertEqual(["RotateLeft"] * 7, [action for action, _ in calls])

    def test_replay_resolves_interactions_from_the_post_movement_state(self):
        calls = []
        initial_metadata = {
            "objects": [{
                "objectId": "Cabinet|stale", "objectType": "Cabinet",
                "visibleBounds2D": [0, 0, 1, 1],
            }],
        }
        post_move_metadata = {
            "objects": [{
                "objectId": "Cabinet|current", "objectType": "Cabinet",
                "visibleBounds2D": [0, 0, 1, 1],
            }],
        }

        class ReplayEnv:
            controller = SimpleNamespace(
                last_event=SimpleNamespace(metadata=initial_metadata),
            )

            def step(self, action, **params):
                calls.append((action, params))
                if action == "MoveAhead":
                    self.controller.last_event.metadata = post_move_metadata
                return {"success": True, "error": None}

        replay_steps(ReplayEnv(), [{
            "step_id": "main__s0",
            "action": "MoveSequence",
            "action_params": {
                "steps": [
                    {"action": "MoveAhead"},
                    {"action": "OpenObject", "params": {"objectType": "Cabinet"}},
                ],
            },
            "success": True,
        }])

        self.assertEqual("Cabinet|current", calls[-1][1]["objectId"])

    def test_replay_rejects_a_failure_for_a_recorded_success(self):
        class ReplayEnv:
            controller = SimpleNamespace(
                last_event=SimpleNamespace(metadata={"objects": []}),
            )

            def step(self, action, **params):
                return {"success": False, "error": "collision"}

        with self.assertRaisesRegex(RuntimeError, "Replay divergence"):
            replay_steps(ReplayEnv(), [{
                "step_id": "main__s0",
                "action": "MoveSequence",
                "action_params": {"steps": [{"action": "MoveAhead"}]},
                "success": True,
            }])

    def test_replay_allows_a_failure_that_was_recorded_as_failed(self):
        calls = []

        class ReplayEnv:
            controller = SimpleNamespace(
                last_event=SimpleNamespace(metadata={"objects": []}),
            )

            def step(self, action, **params):
                calls.append(action)
                return {"success": False, "error": "collision"}

        replay_steps(ReplayEnv(), [{
            "step_id": "main__s0",
            "action": "MoveSequence",
            "action_params": {"steps": [{"action": "MoveAhead"}]},
            "success": False,
        }])

        self.assertEqual(["MoveAhead"], calls)

    def test_replay_hide_injection_does_not_require_camera_visibility(self):
        calls = []

        class ReplayEnv:
            controller = object()

            def step(self, action, **params):
                calls.append(("action", action, params))
                return {"success": True, "error": None}

        def replay_injection(controller, method, pddl_params, **params):
            calls.append(("injection", method, pddl_params, params))
            self.assertEqual("hide_object", method)
            self.assertFalse(params["require_visible"])
            return {"success": True, "error": None}

        step = {
            "step_id": "main__s7", "action": "MoveAhead", "action_params": {}, "success": True,
            "oracle_injection_decision": {
                "decided_to_inject": True,
                "modification_success": True,
                "injection": {
                    "method": "hide_object",
                    "params": {
                        "object_type": "AlarmClock",
                        "object_id": "AlarmClock|1",
                        "container_type": "Drawer",
                        "container_id": "Drawer|1",
                    },
                },
            },
        }
        with patch("src.branch_runner.inject", side_effect=replay_injection):
            replay_steps(ReplayEnv(), [step], pddl_params={})

        self.assertEqual("injection", calls[0][0])
