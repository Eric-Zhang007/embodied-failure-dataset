import unittest

from src.task_spec import (
    Goal,
    TaskSpec,
    evaluate_episode_progress,
    evaluate_goal,
    evaluate_task_progress,
    task_spec_criteria_text,
    task_spec_from_dict,
)


def make_obj(
    object_id,
    object_type,
    *,
    pickupable=False,
    receptacle=False,
    toggleable=False,
    visible=True,
    visibleBounds2D=None,
    is_toggled=False,
    receptacle_ids=None,
):
    obj = {
        "objectId": object_id,
        "objectType": object_type,
        "pickupable": pickupable,
        "receptacle": receptacle,
        "toggleable": toggleable,
        "visible": visible,
        "isToggled": is_toggled,
        "receptacleObjectIds": receptacle_ids,
    }
    if visibleBounds2D is not None:
        obj["visibleBounds2D"] = visibleBounds2D
    return obj


class EvaluateGoalTest(unittest.TestCase):
    def test_place_requires_exact_instance_in_exact_receptacle(self):
        goal = Goal(kind="place", target_object_id="AlarmClock|A", receptacle_id="Desk|goal")
        metadata = {
            "objects": [
                make_obj("AlarmClock|A", "AlarmClock", pickupable=True),
                make_obj("Desk|source", "Desk", receptacle=True, receptacle_ids=["AlarmClock|A"]),
                make_obj("Desk|goal", "Desk", receptacle=True),
            ],
            "inventoryObjects": [],
        }

        self.assertFalse(evaluate_goal(metadata, {}, goal).satisfied)

        metadata["objects"][1]["receptacleObjectIds"] = []
        metadata["objects"][2]["receptacleObjectIds"] = ["AlarmClock|A"]
        self.assertTrue(evaluate_goal(metadata, {}, goal).satisfied)

    def test_toggle_requires_exact_lamp_instance(self):
        goal = Goal(kind="toggle", target_object_id="CD|1", toggle_target_id="DeskLamp|1")
        metadata = {
            "objects": [
                make_obj("CD|1", "CD", pickupable=True),
                make_obj("DeskLamp|1", "DeskLamp", toggleable=True, visibleBounds2D=True, is_toggled=False),
                make_obj("DeskLamp|2", "DeskLamp", toggleable=True, visibleBounds2D=True, is_toggled=True),
            ],
            "inventoryObjects": [{"objectId": "CD|1"}],
        }

        self.assertFalse(evaluate_goal(metadata, {}, goal).satisfied)

        metadata["objects"][1]["isToggled"] = True
        self.assertTrue(evaluate_goal(metadata, {}, goal).satisfied)

    def test_state_goal_requires_final_receptacle(self):
        goal = Goal(kind="state", target_object_id="Potato|1", receptacle_id="CounterTop|1", state="heated")
        metadata = {
            "objects": [
                make_obj("Potato|1", "Potato", pickupable=True),
                make_obj("Microwave|1", "Microwave", receptacle=True, receptacle_ids=["Potato|1"]),
                make_obj("CounterTop|1", "CounterTop", receptacle=True),
            ],
            "inventoryObjects": [],
        }
        task_state = {"heated_objects": {"Potato|1"}}

        self.assertFalse(evaluate_goal(metadata, task_state, goal).satisfied)

        metadata["objects"][1]["receptacleObjectIds"] = []
        metadata["objects"][2]["receptacleObjectIds"] = ["Potato|1"]
        self.assertTrue(evaluate_goal(metadata, task_state, goal).satisfied)

    def test_hold_requires_the_exact_object_in_hand(self):
        goal = Goal(kind="hold", target_object_id="Mug|1")
        metadata = {
            "objects": [make_obj("Mug|1", "Mug", pickupable=True)],
            "inventoryObjects": [{"objectId": "Mug|2"}],
        }

        self.assertFalse(evaluate_goal(metadata, {}, goal).satisfied)

        metadata["inventoryObjects"] = [{"objectId": "Mug|1"}]
        self.assertTrue(evaluate_goal(metadata, {}, goal).satisfied)

    def test_composite_sequence_and_or(self):
        objects = [
            make_obj("AlarmClock|A", "AlarmClock", pickupable=True),
            make_obj("Desk|goal", "Desk", receptacle=True, receptacle_ids=["AlarmClock|A"]),
        ]
        metadata = {"objects": objects, "inventoryObjects": []}
        place = Goal(kind="place", target_object_id="AlarmClock|A", receptacle_id="Desk|goal")
        hold = Goal(kind="hold", target_object_id="AlarmClock|A")

        sequence = Goal(kind="composite", op="sequence", children=(place, hold))
        or_goal = Goal(kind="composite", op="or", children=(hold, place))

        self.assertFalse(evaluate_goal(metadata, {}, sequence).satisfied)
        self.assertTrue(evaluate_goal(metadata, {}, or_goal).satisfied)

    def test_task_spec_holds_goal_sequence(self):
        spec = TaskSpec(
            task_id="long_1",
            backend="ai2thor",
            scene="FloorPlan307",
            goals=(Goal(kind="hold", target_object_id="Mug|1"),),
            seed=42,
        )

        self.assertEqual(spec.schema_version, "1.0")
        self.assertEqual(spec.seed, 42)
        self.assertEqual(len(spec.goals), 1)

    def test_task_progress_reports_first_unsatisfied_stage(self):
        spec = TaskSpec(
            task_id="long_1",
            backend="ai2thor",
            scene="FloorPlan1",
            goals=(
                Goal(kind="place", target_object_id="AlarmClock|A", receptacle_id="Desk|goal"),
                Goal(kind="hold", target_object_id="Mug|1"),
            ),
        )
        metadata = {
            "objects": [
                make_obj("AlarmClock|A", "AlarmClock", pickupable=True),
                make_obj("Desk|goal", "Desk", receptacle=True, receptacle_ids=["AlarmClock|A"]),
                make_obj("Mug|1", "Mug", pickupable=True),
            ],
            "inventoryObjects": [],
        }

        progress = evaluate_task_progress(spec, metadata, {})

        self.assertFalse(progress.all_complete)
        self.assertEqual(progress.stage_index, 1)
        self.assertIn("Mug|1 must be held", progress.reason)

        metadata["inventoryObjects"] = [{"objectId": "Mug|1"}]
        progress = evaluate_task_progress(spec, metadata, {})
        self.assertTrue(progress.all_complete)
        self.assertEqual(progress.stage_index, 2)

    def test_task_spec_roundtrip_from_dict(self):
        data = {
            "task_id": "long_1",
            "backend": "ai2thor",
            "scene": "FloorPlan1",
            "seed": 7,
            "goals": [
                {"kind": "place", "target_object_id": "AlarmClock|A", "receptacle_id": "Desk|goal"},
                {
                    "kind": "composite",
                    "op": "or",
                    "children": [{"kind": "hold", "target_object_id": "Mug|1"}],
                },
            ],
        }

        spec = task_spec_from_dict(data)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.task_id, "long_1")
        self.assertEqual(spec.seed, 7)
        self.assertEqual(len(spec.goals), 2)
        self.assertEqual(spec.goals[1].kind, "composite")
        self.assertEqual(spec.goals[1].children[0].kind, "hold")
        self.assertIsNone(task_spec_from_dict(None))
        self.assertIsNone(task_spec_from_dict({"task_id": "x"}))

    def test_episode_progress_falls_back_to_legacy_checker(self):
        ep_data = {
            "task_type": "pick_and_place_simple",
            "pddl_params": {"object_target": "AlarmClock", "parent_target": "Desk"},
            "goal_instances": {
                "final_put": {"objectId": "AlarmClock|A", "receptacleObjectId": "Desk|goal"},
            },
        }
        metadata = {
            "objects": [
                make_obj("AlarmClock|A", "AlarmClock", pickupable=True),
                make_obj("Desk|goal", "Desk", receptacle=True, receptacle_ids=["AlarmClock|A"]),
            ],
            "inventoryObjects": [],
        }

        progress = evaluate_episode_progress(ep_data, metadata, {})

        self.assertTrue(progress.all_complete)
        self.assertEqual(progress.stage_index, 1)
        self.assertEqual(len(progress.stage_statuses), 1)

    def test_task_spec_criteria_text_lists_stages(self):
        spec = TaskSpec(
            task_id="t",
            backend="ai2thor",
            scene="s",
            goals=(Goal(kind="hold", target_object_id="Mug|1"),),
        )

        text = task_spec_criteria_text(spec)

        self.assertIn("Stage 1", text)
        self.assertIn("Mug|1 must be held in hand", text)


if __name__ == "__main__":
    unittest.main()
