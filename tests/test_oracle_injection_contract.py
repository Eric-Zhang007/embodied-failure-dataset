import tempfile
import unittest

from src.branch_runner import _matching_active_trap_id
from src.episode_manager import EpisodeManager


class OracleInjectionContractTest(unittest.TestCase):
    def test_already_open_trap_does_not_match_an_unopenable_object(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "oracle-open", output_dir,
                {"task_goal": "open Cabinet", "scene": "FloorPlan1", "task_type": "pick_and_place"},
            )
            episode.add_runtime_trap({
                "trap_id": "trap_main_open", "branch_id": "main", "status": "active",
                "injection": {
                    "method": "set_object_property",
                    "params": {"object_id": "Cabinet|1", "property": "is_open", "value": True},
                },
                "expected_failure": {"action": "OpenObject", "error_marker": "open"},
            })

            self.assertIsNone(_matching_active_trap_id(
                episode, "main", "OpenObject", "Toaster|1 is not an Openable object",
                action_params={"objectId": "Toaster|1"},
            ))
            self.assertEqual("trap_main_open", _matching_active_trap_id(
                episode, "main", "OpenObject", "Cabinet|1 is already open",
                action_params={"objectId": "Cabinet|1"},
            ))

    def test_invalidated_trap_retains_its_original_trigger_evidence(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "oracle-invalid", output_dir,
                {"task_goal": "open Cabinet", "scene": "FloorPlan1", "task_type": "pick_and_place"},
            )
            episode.add_runtime_trap({
                "trap_id": "trap_main_open", "branch_id": "main", "status": "triggered",
                "env_error": "Toaster|1 is not an Openable object",
            })

            episode.invalidate_runtime_trap("trap_main_open", "error marker did not match")

            trap = episode.data["runtime_traps"][0]
            self.assertEqual("invalidated", trap["status"])
            self.assertEqual("Toaster|1 is not an Openable object", trap["env_error"])
            self.assertEqual("error marker did not match", trap["invalidation_reason"])

    def test_raw_oracle_trap_uses_its_declared_trigger_and_recovery(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "oracle-contract",
                output_dir,
                {"task_goal": "put Tomato in Fridge", "scene": "FloorPlan1", "task_type": "pick_and_place"},
            )
            episode.add_runtime_trap({
                "trap_id": "trap_main_1",
                "branch_id": "main",
                "status": "active",
                "injection": {"method": "close_container", "params": {"object_type": "Fridge"}},
                "expected_failure": {"action": "PutObject", "error_marker": "closed"},
                "recovery_action": {"action": "OpenObject", "params": {"objectId": "Fridge|1"}},
            })

            self.assertEqual(
                "trap_main_1",
                _matching_active_trap_id(episode, "main", "PutObject", "Target receptacle is CLOSED"),
            )
            episode.trigger_runtime_trap("trap_main_1", "main__s1", "Target receptacle is CLOSED")
            self.assertEqual(
                ["trap_main_1"],
                episode.recover_runtime_trap(
                    "main", "OpenObject", {"objectId": "Fridge|1"}, "main__s2",
                ),
            )

    def test_closed_receptacle_matches_oracles_not_open_marker(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "oracle-marker", output_dir,
                {"task_goal": "put Apple in Fridge", "scene": "FloorPlan1", "task_type": "pick_and_place"},
            )
            episode.add_runtime_trap({
                "trap_id": "trap_main_2", "branch_id": "main", "status": "active",
                "injection": {"method": "close_container", "params": {"object_type": "Fridge"}},
                "expected_failure": {"action": "PutObject", "error_marker": "not open"},
                "recovery_action": {"action": "OpenObject", "params": {"objectType": "Fridge"}},
            })

            self.assertEqual(
                "trap_main_2",
                _matching_active_trap_id(
                    episode, "main", "PutObject", "Target openable Receptacle is CLOSED",
                ),
            )

    def test_hidden_pickup_matches_oracles_not_visible_marker(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "oracle-visibility", output_dir,
                {"task_goal": "pick up Apple", "scene": "FloorPlan1", "task_type": "pick_and_place"},
            )
            episode.add_runtime_trap({
                "trap_id": "trap_main_3", "branch_id": "main", "status": "active",
                "injection": {"method": "hide_object", "params": {"object_type": "Apple"}},
                "expected_failure": {"action": "PickupObject", "error_marker": "not visible"},
                "recovery_action": {"action": "OpenObject", "params": {"objectType": "Cabinet"}},
            })

            self.assertEqual(
                "trap_main_3",
                _matching_active_trap_id(
                    episode, "main", "PickupObject", "Target object not found within the specified visibility",
                ),
            )


if __name__ == "__main__":
    unittest.main()
