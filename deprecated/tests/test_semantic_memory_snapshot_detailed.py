import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.branch_runner import BranchConfig, BranchRunner
from src.episode_manager import EpisodeManager
from src.semantic_memory import SemanticMemory
from src.task_conditions import get_completion_criteria_text


def _metadata():
    return {
        "agent": {
            "position": {"x": 0.0, "z": 0.0},
            "rotation": {"y": 90.0},
        },
        "inventoryObjects": [],
        "objects": [
            {
                "objectId": "Cabinet|1",
                "objectType": "Cabinet",
                "visibleBounds2D": [0, 0, 10, 10],
                "visible": True,
                "receptacle": True,
                "position": {"x": 0.0, "z": 1.0},
            },
            {
                "objectId": "Apple|1",
                "objectType": "Apple",
                "visibleBounds2D": [0, 0, 5, 5],
                "visible": True,
                "pickupable": True,
                "parentReceptacles": ["Cabinet|1"],
                "position": {"x": 0.0, "z": 1.0},
            },
        ],
    }


class SemanticMemorySnapshotTest(unittest.TestCase):
    def test_snapshot_callback_persists_type_marker_removal_without_objects(self):
        snapshots = []
        memory = SemanticMemory()
        memory.set_state_change_callback(snapshots.append)

        memory.mark_searched(object_type="Cabinet")
        memory.unmark_searched("Cabinet")

        self.assertEqual(2, len(snapshots))
        self.assertEqual({}, snapshots[-1]["searched_types"])

    def test_snapshot_round_trip_preserves_rendered_semantic_state(self):
        memory = SemanticMemory()
        metadata = _metadata()
        criteria = "Pick up the Apple and put it inside Cabinet."
        for _ in range(3):
            memory.update(
                metadata,
                metadata["objects"],
                "MoveAhead",
                False,
                "Cabinet is blocking movement.",
                criteria,
                "approach Apple",
            )
        memory.mark_searched(object_id="Cabinet|1")
        memory.record_receptacle_visit("Cabinet")
        memory.record_receptacle_open("Cabinet")

        snapshot = memory.export_state()
        self.assertEqual(snapshot, json.loads(json.dumps(snapshot)))

        restored = SemanticMemory.from_state(snapshot)

        self.assertEqual(memory.render(), restored.render())
        self.assertEqual(memory._searched_types, restored._searched_types)
        self.assertEqual(memory.receptacle_visit_counts, restored.receptacle_visit_counts)
        self.assertEqual(memory.receptacle_open_counts, restored.receptacle_open_counts)
        self.assertEqual(
            memory._stuck_tracker._intent_failure,
            restored._stuck_tracker._intent_failure,
        )

    def test_successful_update_clears_stale_last_error(self):
        memory = SemanticMemory()
        metadata = _metadata()
        criteria = "Pick up the Apple and put it inside Cabinet."
        memory.update(
            metadata,
            metadata["objects"],
            "MoveAhead",
            False,
            "Cabinet is blocking movement.",
            criteria,
            "approach Apple",
        )
        memory.update(
            metadata,
            metadata["objects"],
            "MoveBack",
            True,
            None,
            criteria,
            "escape",
        )

        self.assertIsNone(memory._last_error)
        self.assertNotIn("LAST ERROR", memory.render())

    def test_aging_keeps_task_target_and_receptacle(self):
        memory = SemanticMemory()
        metadata = _metadata()
        metadata["objects"].append(
            {
                "objectId": "Mug|1",
                "objectType": "Mug",
                "visibleBounds2D": [0, 0, 5, 5],
                "visible": True,
                "pickupable": True,
                "position": {"x": 0.0, "z": 1.0},
            }
        )
        criteria = get_completion_criteria_text(
            "pick_and_place_simple",
            {"object_target": "Apple", "parent_target": "Cabinet"},
        )
        memory.update(metadata, metadata["objects"], "LookAround", True, None, criteria)

        for _ in range(memory.AGING_THRESHOLD):
            memory.update(metadata, [], "RotateLeft", True, None, criteria)

        self.assertTrue(memory.has_type("Apple"))
        self.assertTrue(memory.has_type("Cabinet"))
        self.assertFalse(memory.has_type("Mug"))
        self.assertIn("WHERE MY TARGET IS", memory.render())

    def test_criteria_text_extracts_target_and_final_receptacle(self):
        memory = SemanticMemory()
        metadata = _metadata()
        criteria = get_completion_criteria_text(
            "pick_and_place_with_movable_recep",
            {
                "object_target": "Apple",
                "mrecep_target": "Bowl",
                "parent_target": "CounterTop",
            },
        )

        memory.update(metadata, metadata["objects"], "LookAround", True, None, criteria)

        self.assertEqual("Apple", memory._task_target)
        self.assertEqual("CounterTop", memory._task_receptacle)

    def test_holding_one_duplicate_does_not_mark_the_other_held_or_placed(self):
        memory = SemanticMemory()
        metadata = _metadata()
        duplicate = {
            "objectId": "Apple|2",
            "objectType": "Apple",
            "visibleBounds2D": [20, 0, 25, 5],
            "visible": True,
            "pickupable": True,
            "position": {"x": 1.0, "z": 1.0},
        }
        metadata["objects"].append(duplicate)
        criteria = "Apple must be inside Cabinet"
        memory.update(metadata, metadata["objects"], "LookAround", True, None, criteria)

        metadata["inventoryObjects"] = [
            {"objectId": "Apple|1", "objectType": "Apple"},
        ]
        memory.update(metadata, [metadata["objects"][0]], "PickupObject", True, None, criteria)
        self.assertEqual("held", memory._objects["Apple|1"].status)
        self.assertEqual("remembered", memory._objects["Apple|2"].status)

        metadata["inventoryObjects"] = []
        memory.update(
            metadata, [metadata["objects"][1]],
            "PutObject", True, None, criteria,
        )
        self.assertEqual("placed", memory._objects["Apple|1"].status)
        self.assertEqual("remembered", memory._objects["Apple|2"].status)

    def test_aging_prunes_state_for_non_task_objects(self):
        memory = SemanticMemory()
        metadata = _metadata()
        mug = {
            "objectId": "Mug|1",
            "objectType": "Mug",
            "visibleBounds2D": [20, 0, 25, 5],
            "visible": True,
            "pickupable": True,
            "isToggled": False,
            "position": {"x": 1.0, "z": 1.0},
        }
        metadata["objects"].append(mug)
        criteria = "Apple must be inside Cabinet"
        memory.update(metadata, metadata["objects"], "LookAround", True, None, criteria)

        for _ in range(memory.AGING_THRESHOLD):
            memory.update(metadata, [], "RotateLeft", True, None, criteria)

        self.assertNotIn("Mug|1", memory._objects)
        self.assertNotIn("Mug|1", memory._object_state)

    def test_episode_persists_latest_semantic_state_per_branch(self):
        state = {"objects": {}, "step_counter": 3}
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "ep",
                output_dir,
                {
                    "task_goal": "put the apple on the table",
                    "scene": "FloorPlan1",
                    "task_type": "pick_and_place_simple",
                },
            )
            episode.set_semantic_memory_state("main", state)
            episode.set_semantic_memory_state("fork_s1_main", {"step_counter": 7})

            reloaded = EpisodeManager.load(episode.file_path)

        self.assertEqual(state, reloaded.get_semantic_memory_state("main"))
        self.assertEqual(
            {"step_counter": 7},
            reloaded.get_semantic_memory_state("fork_s1_main"),
        )

    def test_episode_persists_step_and_semantic_state_together(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "ep",
                output_dir,
                {
                    "task_goal": "put the apple on the table",
                    "scene": "FloorPlan1",
                    "task_type": "pick_and_place_simple",
                },
            )
            step = {
                "step_id": "main_s0",
                "branch_id": "main",
                "step_index_in_branch": 0,
                "action": "MoveAhead",
                "success": False,
            }
            episode.add_step_with_semantic_memory_state(step, {"step_counter": 1})

            reloaded = EpisodeManager.load(episode.file_path)

        self.assertEqual([step], reloaded.get_steps_for_branch("main"))
        self.assertEqual(
            {"step_counter": 1},
            reloaded.get_semantic_memory_state("main"),
        )

    def test_semantic_resume_replays_environment_without_rebuilding_memory(self):
        memory = SemanticMemory()
        metadata = _metadata()
        memory.update(
            metadata,
            metadata["objects"],
            "MoveAhead",
            False,
            "Cabinet is blocking movement.",
            "Pick up the Apple and put it inside Cabinet.",
            "approach Apple",
        )

        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "ep",
                output_dir,
                {
                    "task_goal": "put the apple on the table",
                    "scene": "FloorPlan1",
                    "task_type": "pick_and_place_simple",
                    "alfred_scene": {"object_poses": [{"objectName": "Apple_1"}]},
                },
            )
            step = {
                "step_id": "main_s0",
                "branch_id": "main",
                "step_index_in_branch": 0,
                "action": "MoveAhead",
                "action_params": {},
                "success": False,
            }
            episode.add_step(step)
            episode.set_semantic_memory_state("main", memory.export_state())

            env = MagicMock()
            env.get_state_snapshot.return_value = {
                "metadata": {"inventoryObjects": [{"objectType": "Apple"}]}
            }
            with patch("src.branch_runner.EnvController", return_value=env), \
                 patch("src.branch_runner.replay_steps") as replay, \
                 patch.object(BranchRunner, "run", return_value="resumed") as run:
                result = BranchRunner.resume(
                    episode_path=episode.file_path,
                    branch_id="main",
                    eb_agent=object(),
                    oracle_agent=object(),
                    memory_mode="semantic",
                )

        self.assertEqual("resumed", result)
        replay.assert_called_once_with(env, [step], skip_failed=True)
        restored_memory = run.call_args.kwargs["_memory"]
        self.assertEqual(memory.render(), restored_memory.render())

    def test_semantic_resume_rejects_episode_without_snapshot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "ep",
                output_dir,
                {
                    "task_goal": "put the apple on the table",
                    "scene": "FloorPlan1",
                    "task_type": "pick_and_place_simple",
                    "alfred_scene": {"object_poses": [{"objectName": "Apple_1"}]},
                },
            )
            episode.add_step(
                {
                    "step_id": "main_s0",
                    "branch_id": "main",
                    "step_index_in_branch": 0,
                    "action": "LookAround",
                    "action_params": {},
                    "success": True,
                }
            )

            env = MagicMock()
            with patch("src.branch_runner.EnvController", return_value=env):
                with self.assertRaisesRegex(ValueError, "no saved semantic memory"):
                    BranchRunner.resume(
                        episode_path=episode.file_path,
                        branch_id="main",
                        eb_agent=object(),
                        oracle_agent=object(),
                        memory_mode="semantic",
                    )

    def test_terminal_branch_persists_initial_semantic_snapshot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "ep",
                output_dir,
                {
                    "task_goal": "put the apple on the table",
                    "scene": "FloorPlan1",
                    "task_type": "pick_and_place_simple",
                },
            )
            runner = BranchRunner(
                eb_agent=MagicMock(),
                oracle_agent=MagicMock(),
                output_dir=output_dir,
                memory_mode="semantic",
                enable_phase2=False,
            )
            env = MagicMock()
            env.get_state_snapshot.return_value = {"frame": None, "metadata": {}}

            with patch("src.branch_runner.check_task_complete", return_value=(True, None)):
                runner.run(
                    BranchConfig("ep", "main", None),
                    env,
                    episode,
                    start_step_index=1,
                )

        self.assertEqual({}, episode.get_semantic_memory_state("main")["objects"])

    def test_interrupted_failure_does_not_persist_memory_ahead_of_steps(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "ep",
                output_dir,
                {
                    "task_goal": "put the apple on the table",
                    "scene": "FloorPlan1",
                    "task_type": "pick_and_place_simple",
                },
            )
            eb_agent = MagicMock()
            eb_agent.propose_action.return_value = {
                "action": "MoveAhead",
                "params": {},
                "reasoning": "move closer",
            }
            eb_agent.diagnose_failure.side_effect = RuntimeError("diagnosis interrupted")
            runner = BranchRunner(
                eb_agent=eb_agent,
                oracle_agent=MagicMock(),
                output_dir=output_dir,
                memory_mode="semantic",
                enable_phase2=False,
            )
            metadata = {"agent": {"position": {}, "rotation": {}}, "objects": []}
            env = MagicMock()
            env.get_state_snapshot.return_value = {"frame": None, "metadata": metadata}
            env.step.return_value = {
                "success": False,
                "error": "movement blocked",
                "frame": None,
                "metadata": metadata,
            }

            with patch("src.branch_runner.check_task_complete", return_value=(False, "not done")):
                with self.assertRaisesRegex(RuntimeError, "diagnosis interrupted"):
                    runner.run(
                        BranchConfig("ep", "main", None),
                        env,
                        episode,
                        start_step_index=1,
                    )

        self.assertEqual([], episode.get_steps_for_branch("main"))
        self.assertEqual(0, episode.get_semantic_memory_state("main")["step_counter"])
