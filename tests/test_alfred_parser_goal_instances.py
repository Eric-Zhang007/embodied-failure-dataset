import unittest

from src.alfred_parser import extract_goal_instances, extract_metadata


def _traj(low_actions):
    return {
        "task_id": "trial_1",
        "task_type": "pick_and_place_simple",
        "pddl_params": {"object_target": "AlarmClock", "parent_target": "Desk"},
        "scene": {"floor_plan": "FloorPlan307"},
        "plan": {"low_actions": low_actions},
    }


class ExtractGoalInstancesTest(unittest.TestCase):
    def test_extracts_final_put_and_movable_ids(self):
        traj = _traj([
            {"high_idx": 0, "api_action": {"action": "PickupObject", "objectId": "Spoon|1"}},
            {"high_idx": 0, "api_action": {"action": "PutObject", "objectId": "Spoon|1", "receptacleObjectId": "Pan|1"}},
            {"high_idx": 1, "api_action": {"action": "PickupObject", "objectId": "Pan|1"}},
            {"high_idx": 1, "api_action": {"action": "PutObject", "objectId": "Pan|1", "receptacleObjectId": "Sink|1"}},
        ])

        goal = extract_goal_instances(traj)

        self.assertEqual(goal["pickup_object_ids"], ["Spoon|1", "Pan|1"])
        self.assertEqual(goal["put_receptacle_ids"], ["Pan|1", "Sink|1"])
        self.assertEqual(goal["final_put"], {"objectId": "Pan|1", "receptacleObjectId": "Sink|1"})
        self.assertEqual(goal["movable_receptacle_id"], "Pan|1")
        self.assertEqual(goal["movable_target_object_id"], "Spoon|1")

    def test_extracts_toggle_on_for_examine_tasks(self):
        traj = _traj([
            {"high_idx": 0, "api_action": {"action": "ToggleObjectOn", "objectId": "DeskLamp|1"}},
            {"high_idx": 0, "api_action": {"action": "PickupObject", "objectId": "CD|1"}},
        ])

        goal = extract_goal_instances(traj)

        self.assertEqual(goal["toggle_on_object_ids"], ["DeskLamp|1"])
        self.assertEqual(goal["pickup_object_ids"], ["CD|1"])
        self.assertNotIn("final_put", goal)

    def test_extract_metadata_includes_goal_instances(self):
        traj = _traj([
            {"high_idx": 0, "api_action": {"action": "PickupObject", "objectId": "AlarmClock|A"}},
            {"high_idx": 0, "api_action": {"action": "PutObject", "objectId": "AlarmClock|A", "receptacleObjectId": "Desk|goal"}},
        ])

        meta = extract_metadata(traj)

        self.assertEqual(meta["goal_instances"]["final_put"]["receptacleObjectId"], "Desk|goal")


if __name__ == "__main__":
    unittest.main()
