import unittest

from src.env_injector import _occlude_object, _swap_object, inject


class FakeEvent:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeController:
    def __init__(self):
        self.calls = []
        self.metadata = {
            "lastActionSuccess": True,
            "agent": {
                "position": {"x": 0, "y": 0.9, "z": 0},
                "rotation": {"y": 0},
            },
            "objects": [
                {
                    "objectId": "Apple|1",
                    "name": "Apple_1",
                    "objectType": "Apple",
                    "visible": True,
                    "pickupable": True,
                    "position": {"x": 0, "y": 1, "z": 2},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                },
                {
                    "objectId": "Mug|1",
                    "name": "Mug_1",
                    "objectType": "Mug",
                    "visible": False,
                    "pickupable": True,
                    "position": {"x": 1, "y": 1, "z": 1},
                    "rotation": {"x": 0, "y": 90, "z": 0},
                },
                {
                    "objectId": "Cabinet|1",
                    "name": "Cabinet_1",
                    "objectType": "Cabinet",
                    "visible": True,
                    "pickupable": False,
                    "openable": True,
                    "isOpen": True,
                    "position": {"x": 2, "y": 1, "z": 1},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                },
            ],
        }

    def step(self, action=None, **params):
        call = dict(params)
        if action is not None:
            call["action"] = action
        self.calls.append(call)
        return FakeEvent(self.metadata)


class EnvInjectorTest(unittest.TestCase):
    def test_occlude_uses_single_object_placement_not_set_object_poses(self):
        controller = FakeController()

        _occlude_object(controller, "Apple")

        actions = [c["action"] for c in controller.calls]
        self.assertNotIn("SetObjectPoses", actions)
        self.assertIn("PlaceObjectAtPoint", actions)
        place_call = next(c for c in controller.calls if c["action"] == "PlaceObjectAtPoint")
        self.assertEqual("Mug|1", place_call["objectId"])

    def test_swap_uses_single_object_placement_not_set_object_poses(self):
        controller = FakeController()

        _swap_object(controller, "Apple", "Mug")

        actions = [c["action"] for c in controller.calls]
        self.assertNotIn("SetObjectPoses", actions)
        self.assertEqual(2, actions.count("PlaceObjectAtPoint"))

    def test_declared_injection_methods_execute_real_ai2thor_actions(self):
        controller = FakeController()

        remove_result = inject(controller, "remove_object", object_type="Mug")
        close_result = inject(controller, "close_container", object_type="Cabinet")

        self.assertTrue(remove_result["success"])
        self.assertTrue(close_result["success"])
        actions = [c["action"] for c in controller.calls]
        self.assertIn("DisableObject", actions)
        self.assertIn("CloseObject", actions)


if __name__ == "__main__":
    unittest.main()
