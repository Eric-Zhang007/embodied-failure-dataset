import unittest

import numpy as np

from src.env_injector import get_actionable_injections
from src.eb_agent import build_phase3_prompt
from src.oracle_agent import OracleAgent, build_phase2_prompt, build_phase4_prompt


def _object(object_type, **extra):
    obj = {
        "objectId": f"{object_type}|1",
        "objectType": object_type,
        "visibleBounds2D": [0, 0, 10, 10],
        "visible": True,
    }
    obj.update(extra)
    return obj


class ActionAwareInjectionTest(unittest.TestCase):
    def test_runtime_phase2_system_disallows_arbitrary_injection_methods(self):
        class CapturingOracleClient:
            def __init__(self):
                self.system_prompt = ""
                self.user_text = ""

            def chat_with_image_json(self, *, system_prompt, **kwargs):
                self.system_prompt = system_prompt
                self.user_text = kwargs["user_text"]
                return {"inject": False, "candidate_id": None, "reasoning": "skip"}

        client = CapturingOracleClient()
        agent = OracleAgent(client)
        agent.decide_injection(
            task_goal="put egg in microwave",
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            env_state={"objects": []},
            proposed_action="PutObject",
            proposed_params={"objectType": "Egg", "receptacleType": "Microwave"},
            eb_reasoning="ready",
            action_history=[],
            cascade_level=0,
            candidates=[{
                "candidate_id": "close_open_receptacle_before_put:Microwave|1",
                "failure_type": "close_open_receptacle_before_put",
                "difficulty": {"level": "low", "recovery_steps": 1},
                "description": "Close the Microwave.",
                "expected_effect": "The placement fails.",
                "recovery_action": {"action": "OpenObject", "params": {"objectType": "Microwave"}},
            }],
        )

        self.assertIn("VALIDATED CANDIDATES", client.user_text)
        for forbidden in ("hide_object", "occlude_object", "swap_object", '"injection"'):
            self.assertNotIn(forbidden, client.system_prompt)

    def test_candidates_have_stable_ids_and_recovery_actions(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
            ],
            "inventoryObjects": [{"objectType": "Egg"}],
        }

        candidate = get_actionable_injections(
            metadata,
            "PutObject",
            {"objectType": "Egg", "receptacleType": "Microwave"},
        )[0]

        self.assertEqual("close_open_receptacle_before_put:Microwave|1", candidate["candidate_id"])
        self.assertEqual(
            {"action": "OpenObject", "params": {"objectType": "Microwave"}},
            candidate["recovery_action"],
        )
        self.assertEqual(
            {"level": "low", "recovery_steps": 1},
            candidate["difficulty"],
        )

    def test_phase2_prompt_exposes_only_valid_candidates(self):
        candidates = [{
            "candidate_id": "close_open_receptacle_before_put:Microwave|1",
            "failure_type": "close_open_receptacle_before_put",
            "description": "Close the open Microwave before the placement attempt.",
            "difficulty": {"level": "low", "recovery_steps": 1},
            "recovery_action": {"action": "OpenObject", "params": {"objectType": "Microwave"}},
            "method": "close_container",
            "params": {"object_type": "Microwave", "object_id": "Microwave|1"},
        }]

        prompt = build_phase2_prompt(
            task_goal="put egg in microwave",
            proposed_action="PutObject",
            proposed_params={"objectType": "Egg", "receptacleType": "Microwave"},
            eb_reasoning="the microwave is open",
            env_state={"objects": []},
            action_history=[],
            cascade_level=0,
            is_fork=False,
            remaining_injections=3,
            candidates=candidates,
        )

        self.assertIn("VALIDATED CANDIDATES", prompt)
        self.assertIn(candidates[0]["candidate_id"], prompt)
        self.assertIn("candidate_id", prompt)
        self.assertIn('"difficulty": {"level": "low", "recovery_steps": 1}', prompt)
        self.assertNotIn("occlude_object", prompt)

    def test_phase2_prompt_exposes_environment_confirmed_trap_lifecycle(self):
        prompt = build_phase2_prompt(
            task_goal="put egg in microwave",
            proposed_action="PutObject",
            proposed_params={"objectType": "Egg", "receptacleType": "Microwave"},
            eb_reasoning="retry after reopening the microwave",
            env_state={"objects": []},
            action_history=[],
            cascade_level=0,
            is_fork=False,
            trap_state=[{
                "trap_id": "trap_main_1",
                "candidate_id": "close_open_receptacle_before_put:Microwave|1",
                "failure_type": "close_open_receptacle_before_put",
                "difficulty": {"level": "low", "recovery_steps": 1},
                "expected_effect": "PutObject fails because Microwave is closed.",
                "recovery_action": {"action": "OpenObject", "params": {"objectType": "Microwave"}},
                "status": "recovered",
                "triggered_at_step_id": "main__s1",
                "env_error": "Target openable Receptacle is CLOSED",
                "recovered_at_step_id": "main__s2",
                "recovered_by_action": {"action": "OpenObject", "params": {"objectId": "Microwave|1"}},
            }],
        )

        self.assertIn("TRAP STATE (environment-confirmed)", prompt)
        self.assertIn("close_open_receptacle_before_put:Microwave|1", prompt)
        self.assertIn('"status": "recovered"', prompt)
        self.assertIn("Target openable Receptacle is CLOSED", prompt)
        self.assertIn("recovered_by_action", prompt)
        self.assertIn("recovery_steps", prompt)

    def test_phase3_prompt_exposes_recovery_for_a_triggered_runtime_trap(self):
        prompt = build_phase3_prompt(
            task_goal="empty the pot",
            action_history=[],
            error_message="Target openable Receptacle is CLOSED",
            cascade_description=None,
            trap_state=[{
                "candidate_id": "close_open_receptacle_before_put:Microwave|1",
                "failure_type": "close_open_receptacle_before_put",
                "expected_effect": "PutObject rejects until the Microwave is opened again.",
                "recovery_action": {
                    "action": "OpenObject",
                    "params": {"objectType": "Microwave"},
                },
                "status": "triggered",
            }],
        )

        self.assertIn("ENVIRONMENT-CONFIRMED RUNTIME TRAP", prompt)
        self.assertIn("close_open_receptacle_before_put:Microwave|1", prompt)
        self.assertIn("OpenObject", prompt)

    def test_phase4_prompt_exposes_environment_confirmed_triggered_trap(self):
        prompt = build_phase4_prompt(
            task_goal="put egg in microwave",
            error_message="Target openable Receptacle is CLOSED",
            action_history=[],
            eb_diagnosis="The microwave is closed.",
            eb_recovery_reasoning="Open it, then retry.",
            eb_counterfactual=None,
            eb_proposed_recovery={"action": "OpenObject", "params": {"objectType": "Microwave"}},
            trap_state=[{
                "trap_id": "trap_main_1",
                "candidate_id": "close_open_receptacle_before_put:Microwave|1",
                "failure_type": "close_open_receptacle_before_put",
                "status": "triggered",
                "expected_effect": "PutObject rejects until the Microwave is opened again.",
                "recovery_action": {"action": "OpenObject", "params": {"objectType": "Microwave"}},
                "env_error": "Target openable Receptacle is CLOSED",
            }],
        )

        self.assertIn("ENVIRONMENT-CONFIRMED TRAP STATE", prompt)
        self.assertIn("close_open_receptacle_before_put:Microwave|1", prompt)
        self.assertIn('"status": "triggered"', prompt)
        self.assertIn("Target openable Receptacle is CLOSED", prompt)

    def test_open_receptacle_before_put_has_close_candidate(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
            ],
            "inventoryObjects": [{"objectType": "Egg"}],
        }

        candidates = get_actionable_injections(
            metadata,
            "PutObject",
            {"objectType": "Egg", "receptacleType": "Microwave"},
        )

        self.assertEqual(1, len(candidates))
        self.assertEqual("close_open_receptacle_before_put", candidates[0]["failure_type"])
        self.assertEqual("close_container", candidates[0]["method"])
        self.assertEqual(
            {"object_type": "Microwave", "object_id": "Microwave|1"},
            candidates[0]["params"],
        )

    def test_put_with_visible_countertop_has_temporary_placement_candidate(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
                _object("CounterTop", receptacle=True, openable=False),
            ],
            "inventoryObjects": [{"objectId": "Egg|1", "objectType": "Egg"}],
        }

        candidates = get_actionable_injections(
            metadata,
            "PutObject",
            {"objectType": "Egg", "receptacleType": "Microwave"},
        )

        temporary = next(
            candidate for candidate in candidates
            if candidate["failure_type"] == "temporarily_place_held_object_before_put"
        )
        self.assertEqual(
            "temporarily_place_held_object_before_put:Egg|1:CounterTop|1",
            temporary["candidate_id"],
        )
        self.assertEqual(
            "place_held_object_in_visible_receptacle",
            temporary["method"],
        )
        self.assertEqual(
            {"action": "PickupObject", "params": {"objectType": "Egg"}},
            temporary["recovery_action"],
        )
        self.assertEqual(
            {"level": "medium", "recovery_steps": 2},
            temporary["difficulty"],
        )

    def test_temporary_placement_requires_an_identified_held_object(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
                _object("CounterTop", receptacle=True, openable=False),
            ],
            "inventoryObjects": [{"objectType": "Egg"}],
        }

        candidates = get_actionable_injections(
            metadata,
            "PutObject",
            {"objectType": "Egg", "receptacleType": "Microwave"},
        )

        self.assertFalse(any(
            candidate["failure_type"] == "temporarily_place_held_object_before_put"
            for candidate in candidates
        ))

    def test_temporary_placement_requires_an_usable_target_receptacle(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=False),
                _object("CounterTop", receptacle=True, openable=False),
            ],
            "inventoryObjects": [{"objectId": "Egg|1", "objectType": "Egg"}],
        }

        self.assertEqual(
            [],
            get_actionable_injections(
                metadata,
                "PutObject",
                {"objectType": "Egg", "receptacleType": "Microwave"},
            ),
        )

    def test_temporary_placement_rejects_unverified_receptacle_types(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
                _object("CoffeeMachine", receptacle=True, openable=False),
            ],
            "inventoryObjects": [{"objectId": "Egg|1", "objectType": "Egg"}],
        }

        candidates = get_actionable_injections(
            metadata,
            "PutObject",
            {"objectType": "Egg", "receptacleType": "Microwave"},
        )

        self.assertFalse(any(
            candidate["failure_type"] == "temporarily_place_held_object_before_put"
            for candidate in candidates
        ))

    def test_closed_receptacle_before_put_has_no_candidate(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=False),
            ]
        }

        candidates = get_actionable_injections(
            metadata,
            "PutObject",
            {"objectType": "Egg", "receptacleType": "Microwave"},
        )

        self.assertEqual([], candidates)

    def test_unrelated_action_has_no_candidate(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
            ],
            "inventoryObjects": [{"objectType": "Egg"}],
        }

        for action, params in (
            ("MoveAhead", {}),
            ("FillObjectWithLiquid", {"objectType": "Pot", "fillLiquid": "water"}),
            ("EmptyLiquidFromObject", {"objectType": "Pot"}),
            ("ToggleObjectOff", {"objectType": "FloorLamp"}),
        ):
            self.assertEqual([], get_actionable_injections(metadata, action, params))

    def test_put_without_held_object_has_no_candidate(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
            ],
            "inventoryObjects": [],
        }

        candidates = get_actionable_injections(
            metadata,
            "PutObject",
            {"objectType": "Egg", "receptacleType": "Microwave"},
        )

        self.assertEqual([], candidates)

    def test_closed_toggleable_device_has_open_candidate(self):
        metadata = {
            "objects": [
                _object(
                    "Microwave", receptacle=True, openable=True,
                    toggleable=True, isOpen=False, isToggled=False,
                ),
            ],
        }

        candidates = get_actionable_injections(
            metadata,
            "ToggleObjectOn",
            {"objectType": "Microwave"},
        )

        self.assertEqual(1, len(candidates))
        self.assertEqual(
            "open_closed_device_before_toggle:Microwave|1",
            candidates[0]["candidate_id"],
        )
        self.assertEqual("open_container", candidates[0]["method"])
        self.assertEqual(
            {"action": "CloseObject", "params": {"objectType": "Microwave"}},
            candidates[0]["recovery_action"],
        )

    def test_open_device_has_no_toggle_candidate(self):
        metadata = {
            "objects": [
                _object(
                    "Microwave", receptacle=True, openable=True,
                    toggleable=True, isOpen=True, isToggled=False,
                ),
            ],
        }

        self.assertEqual(
            [],
            get_actionable_injections(
                metadata,
                "ToggleObjectOn",
                {"objectType": "Microwave"},
            ),
        )

    def test_device_without_a_confirmed_closed_state_has_no_toggle_candidate(self):
        metadata = {
            "objects": [
                _object(
                    "Microwave", receptacle=True, openable=True,
                    toggleable=True, isToggled=False,
                ),
            ],
        }

        self.assertEqual(
            [],
            get_actionable_injections(
                metadata,
                "ToggleObjectOn",
                {"objectType": "Microwave"},
            ),
        )

    def test_already_on_device_has_no_toggle_candidate(self):
        metadata = {
            "objects": [
                _object(
                    "Microwave", receptacle=True, openable=True,
                    toggleable=True, isOpen=False, isToggled=True,
                ),
            ],
        }

        self.assertEqual(
            [],
            get_actionable_injections(
                metadata,
                "ToggleObjectOn",
                {"objectType": "Microwave"},
            ),
        )

    def test_off_lamps_have_turn_on_candidates_for_toggle_on(self):
        for object_type in ("DeskLamp", "FloorLamp"):
            with self.subTest(object_type=object_type):
                candidates = get_actionable_injections(
                    {"objects": [_object(object_type, toggleable=True, isToggled=False)]},
                    "ToggleObjectOn",
                    {"objectType": object_type},
                )

                self.assertEqual(1, len(candidates))
                candidate = candidates[0]
                self.assertEqual(
                    f"turn_on_lamp_before_toggle_on:{object_type}|1",
                    candidate["candidate_id"],
                )
                self.assertEqual("turn_on_lamp", candidate["method"])
                self.assertEqual({"object_id": f"{object_type}|1"}, candidate["params"])
                self.assertEqual(
                    {"action": "ToggleObjectOff", "params": {"objectType": object_type}},
                    candidate["recovery_action"],
                )

    def test_toggle_on_rejects_already_on_or_unsupported_lamps(self):
        for obj in (
            _object("FloorLamp", toggleable=True, isToggled=True),
            _object("LightSwitch", toggleable=True, isToggled=False),
            _object("Toaster", toggleable=True, isToggled=False),
        ):
            self.assertEqual(
                [],
                get_actionable_injections(
                    {"objects": [obj]},
                    "ToggleObjectOn",
                    {"objectType": obj["objectType"]},
                ),
            )

    def test_move_sequence_starting_with_put_has_close_candidate(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
            ],
            "inventoryObjects": [{"objectType": "Egg"}],
        }

        candidates = get_actionable_injections(
            metadata,
            "MoveSequence",
            {
                "steps": [{
                    "action": "PutObject",
                    "params": {"objectType": "Egg", "receptacleType": "Microwave"},
                }],
            },
        )

        self.assertEqual(1, len(candidates))
        self.assertEqual("close_container", candidates[0]["method"])

    def test_move_sequence_with_later_put_has_no_candidate(self):
        metadata = {
            "objects": [
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
            ],
            "inventoryObjects": [{"objectType": "Egg"}],
        }

        candidates = get_actionable_injections(
            metadata,
            "MoveSequence",
            {
                "steps": [
                    {"action": "MoveAhead"},
                    {
                        "action": "PutObject",
                        "params": {"objectType": "Egg", "receptacleType": "Microwave"},
                    },
                ],
            },
        )

        self.assertEqual([], candidates)

    def test_visible_pickup_in_open_container_has_close_candidate(self):
        metadata = {
            "objects": [
                _object(
                    "Apple", objectId="Apple|1", pickupable=True,
                    parentReceptacles=["Microwave|1"],
                ),
                _object(
                    "Microwave", receptacle=True, openable=True, isOpen=True,
                ),
            ],
            "inventoryObjects": [],
        }

        candidates = get_actionable_injections(
            metadata,
            "PickupObject",
            {"objectType": "Apple"},
        )

        self.assertEqual(1, len(candidates))
        self.assertEqual("close_open_container_before_pickup", candidates[0]["failure_type"])
        self.assertEqual("close_container", candidates[0]["method"])
        self.assertEqual(
            {"object_type": "Microwave", "object_id": "Microwave|1"},
            candidates[0]["params"],
        )

    def test_pickup_without_an_open_parent_container_has_no_candidate(self):
        metadata = {
            "objects": [
                _object("Apple", objectId="Apple|1", pickupable=True),
                _object("Microwave", receptacle=True, openable=True, isOpen=True),
            ],
            "inventoryObjects": [],
        }

        candidates = get_actionable_injections(
            metadata,
            "PickupObject",
            {"objectType": "Apple"},
        )

        self.assertEqual([], candidates)

    def test_put_candidate_uses_the_same_receptacle_order_as_action_resolution(self):
        metadata = {
            "objects": [
                _object(
                    "Microwave", objectId="Microwave|second", receptacle=True,
                    openable=True, isOpen=True,
                ),
                _object(
                    "Microwave", objectId="Microwave|first", receptacle=True,
                    openable=True, isOpen=True,
                ),
            ],
            "inventoryObjects": [{"objectType": "Egg"}],
        }

        candidates = get_actionable_injections(
            metadata,
            "PutObject",
            {"objectType": "Egg", "receptacleType": "Microwave"},
        )

        self.assertEqual("Microwave|second", candidates[0]["params"]["object_id"])

    def test_pickup_candidate_uses_the_same_target_order_as_action_resolution(self):
        metadata = {
            "objects": [
                _object(
                    "Apple", objectId="Apple|second", pickupable=True,
                    parentReceptacles=["Microwave|second"],
                ),
                _object(
                    "Apple", objectId="Apple|first", pickupable=True,
                    parentReceptacles=["Microwave|first"],
                ),
                _object(
                    "Microwave", objectId="Microwave|second", receptacle=True,
                    openable=True, isOpen=True,
                ),
                _object(
                    "Microwave", objectId="Microwave|first", receptacle=True,
                    openable=True, isOpen=True,
                ),
            ],
            "inventoryObjects": [],
        }

        candidates = get_actionable_injections(
            metadata,
            "PickupObject",
            {"objectType": "Apple"},
        )

        self.assertEqual("Microwave|second", candidates[0]["params"]["object_id"])


if __name__ == "__main__":
    unittest.main()
