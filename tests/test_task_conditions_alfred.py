import unittest

from src.task_conditions import check_task_complete, get_completion_criteria_text


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
    parent_ids=None,
):
    obj = {
        "objectId": object_id,
        "name": object_id.split("|")[0],
        "objectType": object_type,
        "pickupable": pickupable,
        "receptacle": receptacle,
        "toggleable": toggleable,
        "visible": visible,
        "isToggled": is_toggled,
        "receptacleObjectIds": receptacle_ids,
        "parentReceptacles": parent_ids,
    }
    if visibleBounds2D is not None:
        obj["visibleBounds2D"] = visibleBounds2D
    return obj


class AlfredTaskConditionsTest(unittest.TestCase):
    def test_clean_requires_cleaned_object_and_final_receptacle(self):
        metadata = {
            "objects": [
                make_obj("Mug|1", "Mug", pickupable=True),
                make_obj("Table|1", "Table", receptacle=True, receptacle_ids=["Mug|1"]),
            ],
            "inventoryObjects": [],
        }
        ep_data = {
            "task_type": "pick_clean_then_place_in_recep",
            "pddl_params": {"object_target": "Mug", "parent_target": "Table"},
        }

        ok, _ = check_task_complete(metadata, ep_data, {"cleaned_objects": set()})
        self.assertFalse(ok)
        ok, _ = check_task_complete(metadata, ep_data, {"cleaned_objects": {"Mug|1"}})
        self.assertTrue(ok)

    def test_heat_and_cool_require_same_object_in_place_and_state_set(self):
        metadata = {
            "objects": [
                make_obj("Potato|1", "Potato", pickupable=True),
                make_obj("Apple|1", "Apple", pickupable=True),
                make_obj("CounterTop|1", "CounterTop", receptacle=True, receptacle_ids=["Potato|1", "Apple|1"]),
            ],
            "inventoryObjects": [],
        }

        heat_ep = {
            "task_type": "pick_heat_then_place_in_recep",
            "pddl_params": {"object_target": "Potato", "parent_target": "CounterTop"},
        }
        cool_ep = {
            "task_type": "pick_cool_then_place_in_recep",
            "pddl_params": {"object_target": "Apple", "parent_target": "CounterTop"},
        }

        ok, _ = check_task_complete(metadata, heat_ep, {"heated_objects": {"Apple|1"}})
        self.assertFalse(ok)
        ok, _ = check_task_complete(metadata, heat_ep, {"heated_objects": {"Potato|1"}})
        self.assertTrue(ok)
        ok, _ = check_task_complete(metadata, cool_ep, {"cooled_objects": {"Potato|1"}})
        self.assertFalse(ok)
        ok, _ = check_task_complete(metadata, cool_ep, {"cooled_objects": {"Apple|1"}})
        self.assertTrue(ok)

    def test_movable_receptacle_requires_full_stack_at_parent(self):
        ep_data = {
            "task_type": "pick_and_place_with_movable_recep",
            "pddl_params": {
                "object_target": "Apple",
                "mrecep_target": "Bowl",
                "parent_target": "Table",
            },
        }
        base_objects = [
            make_obj("Apple|1", "Apple", pickupable=True),
            make_obj("Bowl|1", "Bowl", pickupable=True, receptacle=True, receptacle_ids=["Apple|1"]),
            make_obj("Table|1", "Table", receptacle=True, receptacle_ids=[]),
        ]

        ok, _ = check_task_complete({"objects": base_objects, "inventoryObjects": []}, ep_data)
        self.assertFalse(ok)

        base_objects[1]["parentReceptacles"] = ["Table|1"]
        base_objects[2]["receptacleObjectIds"] = ["Bowl|1"]
        ok, _ = check_task_complete({"objects": base_objects, "inventoryObjects": []}, ep_data)
        self.assertTrue(ok)

    def test_criteria_text_uses_same_task_type_resolution_as_checker(self):
        pddl = {
            "object_target": "Apple",
            "mrecep_target": "Bowl",
            "parent_target": "Table",
        }

        text = get_completion_criteria_text("pick_and_place", pddl)

        self.assertIn("Apple must be inside Bowl", text)
        self.assertIn("Bowl must be inside Table", text)
        self.assertNotIn("Apple must be inside Table", text)

    def test_examine_uses_inventory_and_visible_toggled_lamp(self):
        ep_data = {
            "task_type": "look_at_obj_in_light",
            "pddl_params": {"object_target": "CD", "toggle_target": "DeskLamp"},
        }
        metadata = {
            "objects": [
                make_obj("CD|1", "CD", pickupable=True),
                make_obj("DeskLamp|1", "DeskLamp", toggleable=True, visibleBounds2D=False, is_toggled=True),
            ],
            "inventoryObjects": [{"objectId": "CD|1"}],
        }

        ok, _ = check_task_complete(metadata, ep_data)
        self.assertFalse(ok)
        metadata["objects"][1]["visibleBounds2D"] = True
        ok, _ = check_task_complete(metadata, ep_data)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
