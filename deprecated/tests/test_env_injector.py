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
                {
                    "objectId": "CounterTop|1",
                    "name": "CounterTop_1",
                    "objectType": "CounterTop",
                    "visible": True,
                    "receptacle": True,
                    "pickupable": False,
                    "position": {"x": 2, "y": 1, "z": 2},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                },
                {
                    "objectId": "FloorLamp|1",
                    "name": "FloorLamp_1",
                    "objectType": "FloorLamp",
                    "visible": True,
                    "toggleable": True,
                    "isToggled": False,
                    "pickupable": False,
                    "position": {"x": 3, "y": 1, "z": 2},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                },
            ],
            "inventoryObjects": [{"objectId": "Apple|1", "objectType": "Apple"}],
        }

    def step(self, action=None, **params):
        call = dict(params)
        if action is not None:
            call["action"] = action
        self.calls.append(call)
        if action in {"ToggleObjectOn", "ToggleObjectOff"}:
            for obj in self.metadata["objects"]:
                if obj["objectId"] == params.get("objectId"):
                    obj["isToggled"] = action == "ToggleObjectOn"
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
        controller.metadata["objects"][2]["isOpen"] = False
        open_result = inject(controller, "open_container", object_type="Cabinet")
        temporary_place_result = inject(
            controller,
            "place_held_object_in_visible_receptacle",
            object_id="Apple|1",
            receptacle_id="CounterTop|1",
        )
        toggle_on_result = inject(
            controller,
            "turn_on_lamp",
            object_id="FloorLamp|1",
        )

        self.assertTrue(remove_result["success"])
        self.assertTrue(close_result["success"])
        self.assertTrue(open_result["success"])
        self.assertTrue(temporary_place_result["success"])
        self.assertTrue(toggle_on_result["success"])
        actions = [c["action"] for c in controller.calls]
        self.assertIn("DisableObject", actions)
        self.assertIn("CloseObject", actions)
        self.assertIn("OpenObject", actions)
        self.assertIn("PutObject", actions)
        self.assertIn("ToggleObjectOn", actions)


if __name__ == "__main__":
    unittest.main()
