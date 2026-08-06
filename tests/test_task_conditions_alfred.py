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

    def test_movable_receptacle_requires_the_same_bowl_to_hold_the_object_and_reach_parent(self):
        ep_data = {
            "task_type": "pick_and_place_with_movable_recep",
            "pddl_params": {
                "object_target": "Apple",
                "mrecep_target": "Bowl",
                "parent_target": "Table",
            },
        }
        objects = [
            make_obj("Apple|1", "Apple", pickupable=True),
            make_obj("Bowl|1", "Bowl", pickupable=True, receptacle=True, receptacle_ids=["Apple|1"]),
            make_obj("Bowl|2", "Bowl", pickupable=True, receptacle=True),
            make_obj("Table|1", "Table", receptacle=True, receptacle_ids=["Bowl|2"]),
        ]

        ok, _ = check_task_complete({"objects": objects, "inventoryObjects": []}, ep_data)

        self.assertFalse(ok)

    def test_sliced_goal_requires_the_placed_instance_to_be_sliced(self):
        ep_data = {
            "task_type": "pick_and_place_simple",
            "pddl_params": {
                "object_target": "Tomato",
                "parent_target": "Bowl",
                "object_sliced": True,
            },
        }
        objects = [
            make_obj("Tomato|raw", "Tomato", pickupable=True),
            make_obj("TomatoSliced|other", "TomatoSliced", pickupable=True),
            make_obj("Bowl|1", "Bowl", receptacle=True, receptacle_ids=["Tomato|raw"]),
        ]

        ok, _ = check_task_complete({"objects": objects, "inventoryObjects": []}, ep_data)

        self.assertFalse(ok)

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

    def test_pick_and_place_requires_exact_goal_receptacle_instance(self):
        ep_data = {
            "task_type": "pick_and_place_simple",
            "pddl_params": {"object_target": "AlarmClock", "parent_target": "Desk"},
            "goal_instances": {
                "final_put": {"objectId": "AlarmClock|A", "receptacleObjectId": "Desk|goal"},
            },
        }
        objects = [
            make_obj("AlarmClock|A", "AlarmClock", pickupable=True),
            make_obj("Desk|source", "Desk", receptacle=True, receptacle_ids=["AlarmClock|A"]),
            make_obj("Desk|goal", "Desk", receptacle=True),
        ]
        metadata = {"objects": objects, "inventoryObjects": []}

        ok, _ = check_task_complete(metadata, ep_data)
        self.assertFalse(ok)

        objects[1]["receptacleObjectIds"] = []
        objects[2]["receptacleObjectIds"] = ["AlarmClock|A"]
        ok, _ = check_task_complete(metadata, ep_data)
        self.assertTrue(ok)

    def test_pick_and_place_requires_exact_object_instance(self):
        ep_data = {
            "task_type": "pick_and_place_simple",
            "pddl_params": {"object_target": "AlarmClock", "parent_target": "Desk"},
            "goal_instances": {
                "final_put": {"objectId": "AlarmClock|A", "receptacleObjectId": "Desk|goal"},
            },
        }
        objects = [
            make_obj("AlarmClock|A", "AlarmClock", pickupable=True),
            make_obj("AlarmClock|B", "AlarmClock", pickupable=True),
            make_obj("Desk|goal", "Desk", receptacle=True, receptacle_ids=["AlarmClock|B"]),
        ]

        ok, _ = check_task_complete({"objects": objects, "inventoryObjects": []}, ep_data)
        self.assertFalse(ok)

    def test_examine_requires_exact_object_and_lamp_instances(self):
        ep_data = {
            "task_type": "look_at_obj_in_light",
            "pddl_params": {"object_target": "CD", "toggle_target": "DeskLamp"},
            "goal_instances": {
                "pickup_object_ids": ["CD|1"],
                "toggle_on_object_ids": ["DeskLamp|1"],
            },
        }
        metadata = {
            "objects": [
                make_obj("CD|1", "CD", pickupable=True),
                make_obj("DeskLamp|1", "DeskLamp", toggleable=True, visibleBounds2D=True, is_toggled=False),
                make_obj("DeskLamp|2", "DeskLamp", toggleable=True, visibleBounds2D=True, is_toggled=True),
            ],
            "inventoryObjects": [{"objectId": "CD|1"}],
        }

        ok, _ = check_task_complete(metadata, ep_data)
        self.assertFalse(ok)

        metadata["objects"][1]["isToggled"] = True
        ok, _ = check_task_complete(metadata, ep_data)
        self.assertTrue(ok)

        metadata["inventoryObjects"] = [{"objectId": "CD|2"}]
        metadata["objects"].append(make_obj("CD|2", "CD", pickupable=True))
        ok, _ = check_task_complete(metadata, ep_data)
        self.assertFalse(ok)

    def test_heat_uses_final_put_receptacle_instance(self):
        ep_data = {
            "task_type": "pick_heat_then_place_in_recep",
            "pddl_params": {"object_target": "Potato", "parent_target": "CounterTop"},
            "goal_instances": {
                "final_put": {"objectId": "Potato|1", "receptacleObjectId": "CounterTop|1"},
            },
        }
        objects = [
            make_obj("Potato|1", "Potato", pickupable=True),
            make_obj("Microwave|1", "Microwave", receptacle=True, receptacle_ids=["Potato|1"]),
            make_obj("CounterTop|1", "CounterTop", receptacle=True),
        ]
        metadata = {"objects": objects, "inventoryObjects": []}

        ok, _ = check_task_complete(metadata, ep_data, {"heated_objects": {"Potato|1"}})
        self.assertFalse(ok)

        objects[1]["receptacleObjectIds"] = []
        objects[2]["receptacleObjectIds"] = ["Potato|1"]
        ok, _ = check_task_complete(metadata, ep_data, {"heated_objects": {"Potato|1"}})
        self.assertTrue(ok)

    def test_movable_receptacle_requires_exact_instances(self):
        ep_data = {
            "task_type": "pick_and_place_with_movable_recep",
            "pddl_params": {"object_target": "Apple", "mrecep_target": "Bowl", "parent_target": "Table"},
            "goal_instances": {
                "movable_target_object_id": "Apple|1",
                "movable_receptacle_id": "Bowl|1",
                "final_put": {"objectId": "Bowl|1", "receptacleObjectId": "Table|1"},
            },
        }
        objects = [
            make_obj("Apple|1", "Apple", pickupable=True),
            make_obj("Bowl|1", "Bowl", pickupable=True, receptacle=True, receptacle_ids=["Apple|1"]),
            make_obj("Bowl|2", "Bowl", pickupable=True, receptacle=True, receptacle_ids=["Apple|1"]),
            make_obj("Table|1", "Table", receptacle=True, receptacle_ids=["Bowl|2"]),
        ]

        ok, _ = check_task_complete({"objects": objects, "inventoryObjects": []}, ep_data)
        self.assertFalse(ok)

        objects[2]["receptacleObjectIds"] = []
        objects[3]["receptacleObjectIds"] = ["Bowl|1"]
        ok, _ = check_task_complete({"objects": objects, "inventoryObjects": []}, ep_data)
        self.assertTrue(ok)

    def test_pick_two_counts_only_plan_instances(self):
        ep_data = {
            "task_type": "pick_two_obj_and_place",
            "pddl_params": {"object_target": "Pen", "parent_target": "SideTable"},
            "goal_instances": {
                "pickup_object_ids": ["Pen|1", "Pen|2"],
                "put_receptacle_ids": ["SideTable|1"],
            },
        }
        objects = [
            make_obj("Pen|1", "Pen", pickupable=True),
            make_obj("Pen|2", "Pen", pickupable=True),
            make_obj("Pen|3", "Pen", pickupable=True),
            make_obj("SideTable|1", "SideTable", receptacle=True, receptacle_ids=["Pen|1", "Pen|3"]),
        ]

        ok, _ = check_task_complete({"objects": objects, "inventoryObjects": []}, ep_data)
        self.assertFalse(ok)

        objects[3]["receptacleObjectIds"] = ["Pen|1", "Pen|2", "Pen|3"]
        ok, _ = check_task_complete({"objects": objects, "inventoryObjects": []}, ep_data)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
