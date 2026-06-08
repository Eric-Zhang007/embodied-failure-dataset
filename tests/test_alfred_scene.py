import unittest

from src.alfred_scene import (
    apply_init_action,
    restore_alfred_scene,
    update_alfred_task_state,
)


class FakeEvent:
    def __init__(self, metadata):
        self.metadata = metadata
        self.frame = None


class FakeController:
    def __init__(self, metadata=None):
        self.calls = []
        self.metadata = metadata or {"lastActionSuccess": True, "objects": []}

    def step(self, action=None, **params):
        call = dict(params)
        if isinstance(action, dict):
            call.update(action)
        elif action is not None:
            call["action"] = action
        self.calls.append(call)
        return FakeEvent(self.metadata)


class AlfredSceneTest(unittest.TestCase):
    def test_restore_uses_official_alfred_order(self):
        controller = FakeController({
            "lastActionSuccess": True,
            "objects": [
                {
                    "objectId": "DeskLamp|1",
                    "name": "DeskLamp_1",
                    "objectType": "DeskLamp",
                    "toggleable": True,
                    "pickupable": True,
                    "isToggled": True,
                    "position": {"x": 0, "y": 0, "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                },
                {
                    "objectId": "Cup|1",
                    "name": "Cup_1",
                    "objectType": "Cup",
                    "pickupable": True,
                    "dirtyable": True,
                    "isDirty": False,
                    "canFillWithLiquid": True,
                    "isFilledWithLiquid": True,
                    "position": {"x": 2, "y": 0, "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                }
            ],
        })
        scene = {
            "object_poses": [{"objectName": "DeskLamp_1", "position": {"x": 1, "y": 2, "z": 3}, "rotation": {}}],
            "object_toggles": [{"objectType": "DeskLamp", "isOn": False}],
            "dirty_and_empty": True,
        }

        restore_alfred_scene(controller, scene)

        self.assertEqual(
            [
                "Initialize",
                "Pass",
                "ToggleObjectOff",
                "Pass",
                "DirtyObject",
                "EmptyLiquidFromObject",
                "Pass",
                "SetObjectPoses",
            ],
            [c["action"] for c in controller.calls],
        )
        self.assertEqual("DeskLamp|1", controller.calls[2]["objectId"])
        self.assertEqual("Cup|1", controller.calls[4]["objectId"])
        self.assertEqual("Cup|1", controller.calls[5]["objectId"])
        self.assertEqual(["DeskLamp_1", "Cup_1"], [p["objectName"] for p in controller.calls[7]["objectPoses"]])

    def test_restore_merges_alfred_poses_with_current_ai2thor_objects(self):
        controller = FakeController({
            "lastActionSuccess": True,
            "objects": [
                {
                    "objectId": "Apple|1",
                    "name": "Apple_1",
                    "objectType": "Apple",
                    "pickupable": True,
                    "position": {"x": 0, "y": 0, "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                },
                {
                    "objectId": "Vase|1",
                    "name": "Vase_1",
                    "objectType": "Vase",
                    "pickupable": True,
                    "position": {"x": 9, "y": 9, "z": 9},
                    "rotation": {"x": 0, "y": 180, "z": 0},
                },
            ],
        })
        scene = {
            "object_poses": [
                {"objectName": "Apple_1", "position": {"x": 1, "y": 1, "z": 1}, "rotation": {"x": 0, "y": 90, "z": 0}},
                {"objectName": "Missing_1", "position": {"x": 2, "y": 2, "z": 2}, "rotation": {"x": 0, "y": 0, "z": 0}},
            ],
            "object_toggles": [],
            "dirty_and_empty": False,
        }

        restore_alfred_scene(controller, scene)

        poses = controller.calls[-1]["objectPoses"]
        self.assertEqual(["Apple_1", "Vase_1"], [p["objectName"] for p in poses])
        self.assertEqual({"x": 1, "y": 1, "z": 1}, poses[0]["position"])
        self.assertEqual({"x": 9, "y": 9, "z": 9}, poses[1]["position"])

    def test_init_action_runs_after_restore(self):
        controller = FakeController()
        apply_init_action(
            controller,
            {"action": "TeleportFull", "params": {"x": 1, "y": 0.9, "z": 2, "rotation": 90}},
        )

        self.assertEqual("TeleportFull", controller.calls[0]["action"])
        self.assertEqual({"x": 0, "y": 90, "z": 0}, controller.calls[0]["rotation"])
        self.assertTrue(controller.calls[0]["standing"])
        self.assertTrue(controller.calls[0]["forceAction"])

    def test_alfred_state_tracking_matches_official_side_effects(self):
        metadata = {
            "lastActionSuccess": True,
            "objects": [
                {
                    "objectId": "Faucet|1",
                    "objectType": "Faucet",
                    "position": {"x": 0, "y": 0, "z": 0},
                    "receptacleObjectIds": None,
                },
                {
                    "objectId": "SinkBasin|1",
                    "objectType": "SinkBasin",
                    "visible": True,
                    "position": {"x": 0.1, "y": 0, "z": 0},
                    "receptacleObjectIds": ["Mug|1"],
                },
                {
                    "objectId": "Microwave|1",
                    "objectType": "Microwave",
                    "receptacleObjectIds": ["Potato|1"],
                },
                {
                    "objectId": "Fridge|1",
                    "objectType": "Fridge",
                    "receptacleObjectIds": ["Apple|1"],
                },
            ],
        }
        state = {"cleaned_objects": set(), "heated_objects": set(), "cooled_objects": set()}

        update_alfred_task_state(state, "ToggleObjectOn", {"objectId": "Faucet|1"}, metadata)
        update_alfred_task_state(state, "ToggleObjectOn", {"objectId": "Microwave|1"}, metadata)
        update_alfred_task_state(state, "CloseObject", {"objectId": "Fridge|1"}, metadata)

        self.assertEqual({"Mug|1"}, state["cleaned_objects"])
        self.assertEqual({"Potato|1"}, state["heated_objects"])
        self.assertEqual({"Apple|1"}, state["cooled_objects"])


if __name__ == "__main__":
    unittest.main()
