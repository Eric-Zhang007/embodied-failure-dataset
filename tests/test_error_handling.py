import json
import tempfile
import unittest

import numpy as np

from src.branch_runner import (
    BranchConfig,
    BranchRunner,
    _initial_state_satisfies_goal,
    _normalize_recovery_verdict,
    _requires_oracle_injection,
    _value_or_fallback,
    replay_steps,
)
import src.branch_runner as branch_runner_module
from src.episode_manager import EpisodeManager
from src.vlm_client import VLMClient


def _obj(object_id, object_type, **extra):
    data = {
        "objectId": object_id,
        "objectType": object_type,
        "visible": True,
        "pickupable": False,
        "receptacle": False,
        "position": {"x": 0, "y": 0, "z": 1},
    }
    data.update(extra)
    return data


class OracleInjectionGateTest(unittest.TestCase):
    def test_skips_oracle_for_pure_navigation_sequences(self):
        self.assertFalse(_requires_oracle_injection(
            "MoveSequence",
            {"steps": [{"action": "MoveAhead", "repeat": 4}, {"action": "RotateLeft"}]},
        ))

    def test_keeps_oracle_for_object_interactions(self):
        self.assertTrue(_requires_oracle_injection(
            "MoveSequence",
            {"steps": [{"action": "MoveAhead", "repeat": 2}, {"action": "PickupObject", "params": {"objectType": "Mug"}}]},
        ))
        self.assertTrue(_requires_oracle_injection(
            "OpenObject", {"objectType": "Cabinet"},
        ))

    def test_navigation_sequences_use_the_fast_validation_path(self):
        actions = [
            {"action": "MoveBack", "repeat": 8},
            {"action": "RotateRight"},
        ]
        self.assertFalse(_requires_oracle_injection("MoveSequence", {"steps": actions}))


class ResultFallbackTest(unittest.TestCase):
    def test_keeps_a_numpy_frame_and_only_falls_back_for_none(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        fallback = np.ones((2, 2, 3), dtype=np.uint8)

        self.assertIs(frame, _value_or_fallback(frame, fallback))
        self.assertIs(fallback, _value_or_fallback(None, fallback))


class InitialStateFilterTest(unittest.TestCase):
    def _meta(self, **extra):
        meta = {
            "alfred_task_type": "pick_and_place_simple",
            "pddl_params": {"object_target": "AlarmClock", "parent_target": "Desk"},
        }
        meta.update(extra)
        return meta

    def test_marks_a_goal_already_true_in_the_initial_scene(self):
        metadata = {
            "inventoryObjects": [],
            "objects": [
                _obj("AlarmClock|1", "AlarmClock", pickupable=True),
                _obj(
                    "Desk|1",
                    "Desk",
                    receptacle=True,
                    receptacleObjectIds=["AlarmClock|1"],
                ),
            ],
        }
        meta = self._meta(goal_instances={
            "final_put": {"objectId": "AlarmClock|1", "receptacleObjectId": "Desk|1"},
        })

        self.assertTrue(_initial_state_satisfies_goal(metadata, meta))

    def test_does_not_skip_when_only_a_different_receptacle_holds_target(self):
        metadata = {
            "inventoryObjects": [],
            "objects": [
                _obj("AlarmClock|1", "AlarmClock", pickupable=True),
                _obj("Desk|source", "Desk", receptacle=True, receptacleObjectIds=["AlarmClock|1"]),
                _obj("Desk|goal", "Desk", receptacle=True),
            ],
        }
        meta = self._meta(goal_instances={
            "final_put": {"objectId": "AlarmClock|1", "receptacleObjectId": "Desk|goal"},
        })

        self.assertFalse(_initial_state_satisfies_goal(metadata, meta))

    def test_does_not_skip_without_instance_level_goal_info(self):
        metadata = {
            "inventoryObjects": [],
            "objects": [
                _obj("AlarmClock|1", "AlarmClock", pickupable=True),
                _obj("Desk|1", "Desk", receptacle=True, receptacleObjectIds=["AlarmClock|1"]),
            ],
        }

        self.assertFalse(_initial_state_satisfies_goal(metadata, self._meta()))


class InvalidActionAgent:
    def __init__(self):
        self.last_errors = []

    def plan_intent(self, **kwargs):
        return {"intent": "approach Table", "target": "Table", "reasoning": "test"}

    def review_actions(self, **kwargs):
        return {"approved": True, "reason": "ok", "corrected_actions": []}

    def analyze_scan_room(self, **kwargs):
        return {"face_direction": "ahead", "intent": "approach Table", "target": "Table", "reasoning": "test"}

    def propose_action(self, **kwargs):
        self.last_errors.append(kwargs.get("last_error"))
        if len(self.last_errors) == 1:
            return {"action": "Fly", "params": {}, "reasoning": "invalid action"}
        return {"action": "MoveAhead", "params": {}, "reasoning": "valid retry"}


class UnresolvedObjectAgent:
    def __init__(self):
        self.last_errors = []

    def plan_intent(self, **kwargs):
        return {"intent": "approach Table", "target": "Table", "reasoning": "test"}

    def review_actions(self, **kwargs):
        return {"approved": True, "reason": "ok", "corrected_actions": []}

    def analyze_scan_room(self, **kwargs):
        return {"face_direction": "ahead", "intent": "approach Table", "target": "Table", "reasoning": "test"}

    def propose_action(self, **kwargs):
        self.last_errors.append(kwargs.get("last_error"))
        if len(self.last_errors) == 1:
            return {"action": "PickupObject", "params": {"objectType": "RedCloth"}, "reasoning": "bad type"}
        return {"action": "MoveAhead", "params": {}, "reasoning": "valid retry"}


class DoneRejectedAgent:
    def __init__(self):
        self.last_errors = []

    def plan_intent(self, **kwargs):
        self.last_errors.append(kwargs.get("last_error"))
        if len(self.last_errors) == 1:
            return {"intent": "approach Table", "target": "Table", "reasoning": "step first"}
        if len(self.last_errors) == 2:
            return {"intent": "Done", "target": "", "reasoning": "done too early"}
        return {"intent": "approach Table", "target": "Table", "reasoning": "continue after rejected done"}

    def review_actions(self, **kwargs):
        return {"approved": True, "reason": "ok", "corrected_actions": []}

    def analyze_scan_room(self, **kwargs):
        return {"face_direction": "ahead", "intent": "approach Table", "target": "Table", "reasoning": "test"}

    def propose_action(self, **kwargs):
        self.last_errors.append(kwargs.get("last_error"))
        if len(self.last_errors) == 1:
            return {"action": "MoveAhead", "params": {}, "reasoning": "step first"}
        if len(self.last_errors) == 2:
            return {"action": "Done", "params": {}, "reasoning": "done too early"}
        return {"action": "MoveAhead", "params": {}, "reasoning": "continue after rejected done"}


class RepeatedLookAroundAgent:
    def __init__(self):
        self.propose_errors = []
        self.lookaround_calls = 0

    def plan_intent(self, **kwargs):
        self.propose_errors.append(kwargs.get("last_error"))
        if len(self.propose_errors) == 1:
            return {"intent": "scan room", "target": "", "reasoning": "scan once"}
        return {"intent": "approach Table", "target": "Table", "reasoning": "use retry feedback"}

    def review_actions(self, **kwargs):
        return {"approved": True, "reason": "ok", "corrected_actions": []}

    def analyze_scan_room(self, **kwargs):
        self.lookaround_calls += 1
        return {"face_direction": "ahead", "intent": "approach Table", "target": "Table", "reasoning": "scan done"}

    def propose_action(self, **kwargs):
        self.propose_errors.append(kwargs.get("last_error"))
        if len(self.propose_errors) == 1:
            return {"action": "LookAround", "params": {}, "reasoning": "scan once"}
        return {"action": "MoveAhead", "params": {}, "reasoning": "use retry feedback"}

    def propose_action_lookaround(self, **kwargs):
        self.lookaround_calls += 1
        return {"action": "LookAround", "params": {}, "reasoning": "scan again"}


class EnvironmentFailureAgent:
    def plan_intent(self, **kwargs):
        return {"intent": "approach Table", "target": "Table", "reasoning": "test"}

    def review_actions(self, **kwargs):
        return {"approved": True, "reason": "ok", "corrected_actions": []}

    def analyze_scan_room(self, **kwargs):
        return {"face_direction": "ahead", "intent": "approach Table", "target": "Table", "reasoning": "test"}

    def propose_action(self, **kwargs):
        return {"action": "MoveAhead", "params": {}, "reasoning": "move forward"}

    def diagnose_failure(self, **kwargs):
        return {
            "diagnosis": "MoveAhead was blocked.",
            "recovery_reasoning": "Turn before moving again.",
            "counterfactual": None,
            "proposed_recovery_action": {"action": "RotateLeft", "params": {}},
        }


class EmptyPartialExecutor:
    def __init__(self):
        self.calls = 0

    def execute_intent(self, **kwargs):
        self.calls += 1
        return {
            "actions": [],
            "status": "partial",
            "reasoning": "I need another call.",
            "status_reason": "No executable action was produced.",
        }


class EmptyDoneThenMoveExecutor:
    def __init__(self):
        self.calls = 0

    def execute_intent(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "actions": [],
                "status": "done",
                "reasoning": "The current approach intent is already satisfied.",
                "status_reason": "Already within approach range.",
            }
        return {
            "actions": [{"action": "MoveAhead"}],
            "status": "done",
            "reasoning": "Move once to complete the next intent.",
            "status_reason": "The next intent will be complete after this move.",
        }


class AlwaysEmptyDoneExecutor:
    def __init__(self):
        self.calls = 0

    def execute_intent(self, **kwargs):
        self.calls += 1
        return {
            "actions": [],
            "status": "done",
            "reasoning": "Already there.",
            "status_reason": "No movement needed.",
        }


class EmptyDonePlanner:
    def __init__(self, repeat_same_intent=False):
        self.repeat_same_intent = repeat_same_intent
        self.plan_calls = []
        self.review_calls = 0
        self.dedup_stats = {}

    def plan_intent(self, **kwargs):
        self.plan_calls.append(kwargs)
        if self.repeat_same_intent or len(self.plan_calls) == 1:
            return {"intent": "approach Table", "target": "Table", "reasoning": "test"}
        return {"intent": "approach Apple", "target": "Apple", "reasoning": "next intent"}

    def analyze_scan_room(self, **kwargs):
        return {"face_direction": "ahead", "intent": "approach Table", "target": "Table", "reasoning": "test"}

    def review_actions(self, **kwargs):
        self.review_calls += 1
        return {"approved": True, "reason": "ok", "corrected_actions": []}

    def diagnose_failure(self, **kwargs):
        raise AssertionError("empty done must not enter Phase 3")


class NoopOracle:
    def decide_injection(self, **kwargs):
        return {"inject": False, "reasoning": "", "injection": None}

    def evaluate_failure(self, **kwargs):
        return {
            "diagnosis_correct": True,
            "ground_truth": "The path is blocked.",
            "counterfactual_grade": "WA",
            "counterfactual_gold": None,
            "recovery_verdict": "unrecoverable",
            "should_fork": False,
            "fork_reasoning": None,
        }


class SelectingOracle(NoopOracle):
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id
        self.calls = []

    def decide_injection(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "inject": True,
            "candidate_id": self.candidate_id,
            "reasoning": "This is the right moment for the validated trap.",
        }


class CascadeSelectingOracle(NoopOracle):
    def __init__(self, candidate_ids):
        self.candidate_ids = list(candidate_ids)
        self.calls = []

    def decide_injection(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) > len(self.candidate_ids):
            return {
                "inject": False,
                "candidate_id": None,
                "reasoning": "No further intervention is useful.",
            }
        return {
            "inject": True,
            "candidate_id": self.candidate_ids[len(self.calls) - 1],
            "reasoning": "The prior trap is resolved, so this different validated trap is useful.",
        }

    def evaluate_failure(self, **kwargs):
        return {
            "diagnosis_correct": True,
            "ground_truth": "The injected environment state prevented the action.",
            "counterfactual_grade": "AC",
            "counterfactual_gold": None,
            "recovery_verdict": "recoverable",
            "should_fork": False,
            "fork_reasoning": None,
        }


class OneShotRecoverableOracle(NoopOracle):
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id
        self.calls = 0

    def decide_injection(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "inject": True,
                "candidate_id": self.candidate_id,
                "reasoning": "Use the verified temporary placement trap once.",
            }
        return {"inject": False, "candidate_id": None, "reasoning": "Do not repeat it."}

    def evaluate_failure(self, **kwargs):
        return {
            "diagnosis_correct": True,
            "ground_truth": "The held object was moved to a visible countertop.",
            "counterfactual_grade": "AC",
            "counterfactual_gold": None,
            "recovery_verdict": "recoverable",
            "should_fork": False,
            "fork_reasoning": None,
        }


class CompletingAfterMoveEnv:
    def __init__(self):
        self.step_calls = []
        self.snapshots = 0
        self.moved = False
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def get_state_snapshot(self):
        self.snapshots += 1
        if self.moved:
            objects = [
                _obj("Apple|1", "Apple", pickupable=True),
                _obj("Table|1", "Table", receptacle=True, receptacleObjectIds=["Apple|1"]),
            ]
        else:
            objects = [_obj("Apple|1", "Apple", pickupable=True)]
        return {
            "frame": self.frame,
            "metadata": {"objects": objects, "agent": {"position": {}, "rotation": {"y": 0}}},
            "task_state": {},
        }

    def step(self, action, **params):
        self.step_calls.append((action, params))
        if action == "MoveAhead":
            self.moved = True
        return {
            "success": True,
            "error": None,
            "frame": self.frame,
            "metadata": {"objects": [], "agent": {"position": {}, "rotation": {"y": 0}}},
            "task_state": {},
        }


class FailingMoveEnv:
    def __init__(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def get_state_snapshot(self):
        return {
            "frame": self.frame,
            "metadata": {"objects": [], "agent": {"position": {}, "rotation": {"y": 0}}},
            "task_state": {},
        }

    def step(self, action, **params):
        if action in ("RotateLeft", "RotateRight", "Pass"):
            return {"success": True, "error": None, "frame": self.frame,
                    "metadata": {"objects": [], "agent": {"position": {}, "rotation": {"y": 0}}},
                    "task_state": {}}
        return {
            "success": False,
            "error": "blocked by obstacle",
            "frame": self.frame,
            "metadata": {"objects": [], "agent": {"position": {}, "rotation": {"y": 0}}},
            "task_state": {},
        }


class LookAroundThenMoveEnv:
    def __init__(self):
        self.step_calls = []
        self.completed = False
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def get_state_snapshot(self):
        table_ids = ["Apple|1"] if self.completed else []
        return {
            "frame": self.frame,
            "metadata": {
                "objects": [
                    _obj("Apple|1", "Apple", pickupable=True),
                    _obj("Table|1", "Table", receptacle=True, receptacleObjectIds=table_ids),
                ],
                "inventoryObjects": [],
                "agent": {"position": {}, "rotation": {"y": 0}},
            },
            "task_state": {},
        }

    def step(self, action, **params):
        self.step_calls.append((action, params))
        if action == "MoveAhead":
            self.completed = True
        return {
            "success": True,
            "error": None,
            "frame": self.frame,
            "metadata": self.get_state_snapshot()["metadata"],
            "task_state": {},
        }


class RaisingStepEnv(FailingMoveEnv):
    def step(self, action, **params):
        if action in ("RotateLeft", "RotateRight", "Pass"):
            return {"success": True, "error": None, "frame": self.frame,
                    "metadata": {"objects": [], "agent": {"position": {}, "rotation": {"y": 0}}},
                    "task_state": {}}
        raise RuntimeError("controller crashed")


class RaisingReplayEnv:
    def step(self, action, **params):
        raise ValueError("bad replay action")


class InjectionRetryOracle(NoopOracle):
    def __init__(self):
        self.calls = []

    def decide_injection(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "inject": True,
                "reasoning": "try bad injection",
                "injection": {"method": "bad_method", "params": {}},
            }
        return {"inject": False, "reasoning": "give up after setup failure", "injection": None}


class FailingAgent:
    def propose_action(self, **kwargs):
        raise AssertionError("BranchRunner should not start when initial trap setup fails")


class PutIntoOpenReceptacleAgent:
    dedup_stats = {}

    def propose_action(self, **kwargs):
        return {
            "action": "PutObject",
            "params": {"objectType": "Egg", "receptacleType": "Microwave"},
            "reasoning": "put the held egg into the open microwave",
        }

    def diagnose_failure(self, **kwargs):
        return {
            "diagnosis": "The microwave was closed before placement.",
            "recovery_reasoning": "Open the microwave and retry.",
            "counterfactual": None,
            "proposed_recovery_action": {"action": "OpenObject", "params": {"objectType": "Microwave"}},
        }


class PutIntoOpenReceptacleSequenceAgent(PutIntoOpenReceptacleAgent):
    def propose_action(self, **kwargs):
        return {
            "action": "MoveSequence",
            "params": {
                "steps": [{
                    "action": "PutObject",
                    "params": {"objectType": "Egg", "receptacleType": "Microwave"},
                }],
            },
            "reasoning": "put the held egg into the open microwave",
        }


class PickupFromOpenContainerAgent:
    dedup_stats = {}

    def propose_action(self, **kwargs):
        return {
            "action": "PickupObject",
            "params": {"objectType": "Apple"},
            "reasoning": "pick up the visible apple",
        }

    def diagnose_failure(self, **kwargs):
        return {
            "diagnosis": "The microwave was closed before pickup.",
            "recovery_reasoning": "Open the microwave and retry.",
            "counterfactual": None,
            "proposed_recovery_action": {"action": "OpenObject", "params": {"objectType": "Microwave"}},
        }


class RecoveringPutAgent(PutIntoOpenReceptacleAgent):
    def __init__(self):
        self.calls = 0
        self.trap_states = []

    def propose_action(self, **kwargs):
        self.calls += 1
        return super().propose_action(**kwargs)

    def diagnose_failure(self, **kwargs):
        self.trap_states.append(kwargs.get("trap_state"))
        return {
            "diagnosis": "The microwave was closed before placement.",
            "recovery_reasoning": "Open the microwave and retry.",
            "counterfactual": None,
            "proposed_recovery_action": {"action": "OpenObject", "params": {"objectType": "Microwave"}},
        }


class TemporaryPlacementRecoveryAgent(PutIntoOpenReceptacleAgent):
    def diagnose_failure(self, **kwargs):
        return {
            "diagnosis": "The egg was placed on the visible CounterTop before the intended put.",
            "recovery_reasoning": "Pick up the visible egg and retry the intended placement.",
            "counterfactual": None,
            "proposed_recovery_action": {
                "action": "PickupObject", "params": {"objectType": "Egg"},
            },
        }


class TemporaryThenCloseRecoveryAgent(PutIntoOpenReceptacleAgent):
    def diagnose_failure(self, **kwargs):
        error = kwargs.get("error_message", "")
        if "isn't holding anything" in error:
            return {
                "diagnosis": "The egg was placed on the visible CounterTop.",
                "recovery_reasoning": "Pick up the visible egg before retrying.",
                "counterfactual": None,
                "proposed_recovery_action": {
                    "action": "PickupObject", "params": {"objectType": "Egg"},
                },
            }
        return {
            "diagnosis": "The microwave was closed before the retry.",
            "recovery_reasoning": "Open the microwave and retry the placement.",
            "counterfactual": None,
            "proposed_recovery_action": {
                "action": "OpenObject", "params": {"objectType": "Microwave"},
            },
        }


class _ControllerEvent:
    def __init__(self, metadata):
        self.metadata = metadata


class ActionAwarePutEnv:
    class Controller:
        def __init__(self, env):
            self.env = env
            self.actions = []

        def step(self, action, **params):
            self.actions.append((action, params))
            if action == "CloseObject":
                self.env.microwave_open = False
            elif action == "OpenObject":
                self.env.microwave_open = True
            metadata = self.env._metadata()
            metadata["lastActionSuccess"] = True
            return _ControllerEvent(metadata)

    def __init__(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.microwave_open = True
        self.completed = False
        self.controller = self.Controller(self)

    def _metadata(self):
        return {
            "objects": [
                _obj("Egg|1", "Egg", pickupable=True, visibleBounds2D=[0, 0, 4, 4]),
                _obj(
                    "Microwave|1", "Microwave", receptacle=True, openable=True,
                    isOpen=self.microwave_open, visibleBounds2D=[0, 0, 4, 4],
                    receptacleObjectIds=["Egg|1"] if self.completed else [],
                ),
            ],
            "inventoryObjects": [{"objectId": "Egg|1", "objectType": "Egg"}],
            "agent": {"position": {}, "rotation": {"y": 0}, "cameraHorizon": 0},
        }

    def get_state_snapshot(self):
        return {"frame": self.frame, "metadata": self._metadata(), "task_state": {}}

    def step(self, action, **params):
        if action == "OpenObject":
            self.microwave_open = True
        if action == "PutObject" and not self.microwave_open:
            return {
                "success": False,
                "error": "Target openable Receptacle is CLOSED, can't place if target is not open!",
                "frame": self.frame,
                "metadata": self._metadata(),
                "task_state": {},
            }
        if action == "PutObject":
            self.completed = True
        return {
            "success": True,
            "error": None,
            "frame": self.frame,
            "metadata": self._metadata(),
            "task_state": {},
        }


class ActionAwarePickupEnv(ActionAwarePutEnv):
    def _metadata(self):
        return {
            "objects": [
                _obj(
                    "Apple|1", "Apple", pickupable=True, visibleBounds2D=[0, 0, 4, 4],
                    parentReceptacles=["Microwave|1"],
                ),
                _obj(
                    "Microwave|1", "Microwave", receptacle=True, openable=True,
                    isOpen=self.microwave_open, visibleBounds2D=[0, 0, 4, 4],
                ),
            ],
            "inventoryObjects": [],
            "agent": {"position": {}, "rotation": {"y": 0}, "cameraHorizon": 0},
        }

    def step(self, action, **params):
        if action == "PickupObject" and not self.microwave_open:
            return {
                "success": False,
                "error": "Target object not found within the specified visibility.",
                "frame": self.frame,
                "metadata": self._metadata(),
                "task_state": {},
            }
        if action == "PickupObject":
            self.completed = True
        return {
            "success": True,
            "error": None,
            "frame": self.frame,
            "metadata": self._metadata(),
            "task_state": {},
        }


class ActionAwareTemporaryPlacementEnv:
    class Controller:
        def __init__(self, env):
            self.env = env
            self.actions = []

        def step(self, action, **params):
            self.actions.append((action, params))
            success = True
            error = None
            if action == "CloseObject":
                self.env.microwave_open = False
            elif action == "OpenObject":
                self.env.microwave_open = True
            elif action == "PutObject":
                if params.get("objectId") != "CounterTop|1" or not self.env.egg_held:
                    success = False
                    error = "Temporary placement target is unavailable"
                else:
                    self.env.egg_held = False
                    self.env.egg_parent = "CounterTop|1"
            metadata = self.env._metadata()
            metadata["lastActionSuccess"] = success
            if error:
                metadata["errorMessage"] = error
            return _ControllerEvent(metadata)

    def __init__(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.egg_held = True
        self.egg_parent = None
        self.microwave_open = True
        self.completed = False
        self.controller = self.Controller(self)

    def _metadata(self):
        egg = _obj(
            "Egg|1", "Egg", pickupable=True, isPickedUp=self.egg_held,
            visibleBounds2D=[0, 0, 4, 4],
        )
        if self.egg_parent:
            egg["parentReceptacles"] = [self.egg_parent]
        return {
            "objects": [
                egg,
                _obj(
                    "Microwave|1", "Microwave", receptacle=True, openable=True,
                    isOpen=self.microwave_open, visibleBounds2D=[0, 0, 4, 4],
                    receptacleObjectIds=["Egg|1"] if self.completed else [],
                ),
                _obj(
                    "CounterTop|1", "CounterTop", receptacle=True,
                    visibleBounds2D=[0, 0, 4, 4],
                    receptacleObjectIds=["Egg|1"] if self.egg_parent == "CounterTop|1" else [],
                ),
            ],
            "inventoryObjects": ([{"objectId": "Egg|1", "objectType": "Egg"}]
                                 if self.egg_held else []),
            "agent": {"position": {}, "rotation": {"y": 0}, "cameraHorizon": 0},
        }

    def get_state_snapshot(self):
        return {"frame": self.frame, "metadata": self._metadata(), "task_state": {}}

    def step(self, action, **params):
        if action == "OpenObject":
            self.microwave_open = True
        elif action == "CloseObject":
            self.microwave_open = False
        elif action == "PickupObject":
            if self.egg_parent != "CounterTop|1":
                return {
                    "success": False,
                    "error": "Target object not found within the specified visibility.",
                    "frame": self.frame,
                    "metadata": self._metadata(),
                    "task_state": {},
                }
            self.egg_held = True
            self.egg_parent = None
        elif action == "PutObject":
            if not self.egg_held:
                return {
                    "success": False,
                    "error": "Can't place an object if Agent isn't holding anything",
                    "frame": self.frame,
                    "metadata": self._metadata(),
                    "task_state": {},
                }
            if not self.microwave_open:
                return {
                    "success": False,
                    "error": "Target openable Receptacle is CLOSED, can't place if target is not open!",
                    "frame": self.frame,
                    "metadata": self._metadata(),
                    "task_state": {},
                }
            self.egg_held = False
            self.egg_parent = "Microwave|1"
            self.completed = True
        return {
            "success": True,
            "error": None,
            "frame": self.frame,
            "metadata": self._metadata(),
            "task_state": {},
        }


class ActionAwareCascadeEnv(ActionAwarePutEnv):
    def __init__(self):
        super().__init__()
        self.microwave_open = True
        self.microwave_toggled = False

    def _metadata(self):
        return {
            "objects": [
                _obj("Egg|1", "Egg", pickupable=True, visibleBounds2D=[0, 0, 4, 4]),
                _obj(
                    "Microwave|1", "Microwave", receptacle=True, openable=True,
                    toggleable=True, isOpen=self.microwave_open,
                    isToggled=self.microwave_toggled, visibleBounds2D=[0, 0, 4, 4],
                ),
            ],
            "inventoryObjects": [{"objectId": "Egg|1", "objectType": "Egg"}],
            "agent": {"position": {}, "rotation": {"y": 0}, "cameraHorizon": 0},
        }

    def step(self, action, **params):
        if action == "OpenObject":
            self.microwave_open = True
        elif action == "CloseObject":
            self.microwave_open = False
        elif action == "PutObject" and not self.microwave_open:
            return {
                "success": False,
                "error": "Target openable Receptacle is CLOSED, can't place if target is not open!",
                "frame": self.frame,
                "metadata": self._metadata(),
                "task_state": {},
            }
        elif action == "ToggleObjectOn":
            if self.microwave_open:
                return {
                    "success": False,
                    "error": "Target must be closed to Toggle On!",
                    "frame": self.frame,
                    "metadata": self._metadata(),
                    "task_state": {},
                }
            self.microwave_toggled = True
        return {
            "success": True,
            "error": None,
            "frame": self.frame,
            "metadata": self._metadata(),
            "task_state": {},
        }


class ErrorHandlingTest(unittest.TestCase):
    def _episode(self, output_dir):
        return EpisodeManager(
            "ep",
            output_dir,
            {
                "task_goal": "put the apple on the table",
                "scene": "FloorPlan1",
                "task_type": "pick_and_place_simple",
                "pddl_params": {"object_target": "Apple", "parent_target": "Table"},
                "alfred_scene": {"object_poses": [{"objectName": "Apple_1"}]},
            },
        )

    def test_vlm_json_parse_failure_raises_instead_of_pass_fallback(self):
        client = VLMClient("openai", "fake")
        prompts = []

        def fake_chat_text(system_prompt, user_text, json_mode=False, temperature=0.2, max_tokens=2048):
            prompts.append(user_text)
            return "not json"

        client.chat_text = fake_chat_text

        with self.assertRaises(ValueError):
            client.chat_text_json("sys", "user", max_retries=2)

        self.assertEqual(2, len(prompts))
        self.assertIn("previous response was not valid JSON", prompts[1])

    def test_vlm_missing_required_fields_retries_with_field_error(self):
        client = VLMClient("openai", "fake")
        prompts = []
        responses = iter([
            '{"action": "MoveAhead"}',
            '{"action": "MoveAhead", "params": {}, "reasoning": "ok"}',
        ])

        def fake_chat_text(system_prompt, user_text, json_mode=False, temperature=0.2, max_tokens=2048):
            prompts.append(user_text)
            return next(responses)

        client.chat_text = fake_chat_text

        result = client.chat_text_json(
            "sys",
            "user",
            max_retries=2,
            required_fields=("action", "params", "reasoning"),
        )

        self.assertEqual("MoveAhead", result["action"])
        self.assertIn("missing required JSON fields", prompts[1])
        self.assertIn("params", prompts[1])
        self.assertIn("reasoning", prompts[1])

    def test_replay_steps_raises_on_system_error(self):
        steps = [{"action": "MoveAhead", "action_params": {}, "success": True}]

        with self.assertRaises(ValueError):
            replay_steps(RaisingReplayEnv(), steps, skip_failed=True)

    def test_constructor_preserves_ablation_flags(self):
        flags = {
            "enable_searched_markers": False,
            "enable_intent_dedup": False,
            "enable_critic_guard": True,
            "enable_curiosity_scoreboard": False,
            "enable_contrastive_planner": False,
            "enable_progress_gating": False,
            "enable_search_trail": False,
        }
        runner = BranchRunner(object(), NoopOracle(), ".", **flags)

        for name, expected in flags.items():
            self.assertEqual(expected, getattr(runner, name))

    def test_executor_empty_partial_is_nonexecuted_model_error(self):
        message = BranchRunner._validate_executor_result([], "partial", "executor")

        self.assertIsNotNone(message)
        self.assertIn("was not executed", message)
        self.assertIn("non-empty JSON list", message)

    def test_redundant_interaction_is_rejected_from_current_object_state(self):
        closed_metadata = {
            "objects": [{
                "objectId": "ShowerCurtain|1",
                "objectType": "ShowerCurtain",
                "openable": True,
                "isOpen": False,
            }]
        }

        message = BranchRunner._redundant_interaction_error(
            "CloseObject", {"objectId": "ShowerCurtain|1"}, closed_metadata
        )

        self.assertIsNotNone(message)
        self.assertIn("already closed", message)
        self.assertIsNone(
            BranchRunner._redundant_interaction_error(
                "OpenObject", {"objectId": "ShowerCurtain|1"}, closed_metadata
            )
        )

    def test_non_openable_interaction_is_rejected_before_environment_step(self):
        metadata = {
            "objects": [{
                "objectId": "Toaster|1", "objectType": "Toaster", "openable": False,
            }],
        }

        message = BranchRunner._redundant_interaction_error(
            "OpenObject", {"objectId": "Toaster|1"}, metadata,
        )

        self.assertIn("not openable", message)

    def test_move_sequence_does_not_call_environment_for_non_openable_object(self):
        class NeverStepEnv:
            def step(self, *_args, **_kwargs):
                raise AssertionError("OpenObject on a non-openable object reached Unity")

        runner = BranchRunner(object(), NoopOracle(), ".")
        result, _message = runner._execute_move_sequence(
            {"steps": [{"action": "OpenObject", "params": {"objectType": "Toaster"}}]},
            NeverStepEnv(),
            {"objects": [{
                "objectId": "Toaster|1", "objectType": "Toaster",
                "openable": False, "visibleBounds2D": [0, 0, 1, 1],
            }]},
            "", "main", 1, "ep", "", None,
        )

        self.assertTrue(result["model_error"])

    def test_move_sequence_records_each_attempted_micro_action(self):
        class PartialEnv:
            def __init__(self):
                self.calls = 0

            def step(self, action, **params):
                self.calls += 1
                return {
                    "success": self.calls == 1,
                    "error": None if self.calls == 1 else "blocked",
                    "frame": None,
                    "metadata": {
                        "objects": [],
                        "agent": {"position": {"x": 0.0, "z": 0.0}},
                    },
                }

        runner = BranchRunner(object(), NoopOracle(), ".")
        result, _message = runner._execute_move_sequence(
            {"steps": [{"action": "MoveAhead", "repeat": 3}]},
            PartialEnv(),
            {"objects": [], "agent": {"position": {"x": 0.0, "z": 0.0}}},
            "", "main", 1, "ep", "", None,
        )

        self.assertEqual(
            [
                {"action": "MoveAhead", "params": {}, "success": True, "error": None},
                {"action": "MoveAhead", "params": {}, "success": False, "error": "blocked"},
            ],
            result["execution_trace"],
        )

    def test_empty_executor_result_retries_without_phase3_or_environment_step(self):
        executor = EmptyPartialExecutor()
        runner = BranchRunner(
            EnvironmentFailureAgent(), NoopOracle(), ".", executor_agent=executor, enable_phase2=False
        )
        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            with self.assertRaisesRegex(RuntimeError, "non-empty JSON list"):
                runner.run(
                    BranchConfig("ep", "main", None),
                    FailingMoveEnv(),
                    self._episode(tmp),
                )
            with open(f"{tmp}/ep/failures_main.jsonl", encoding="utf-8") as failure_file:
                failures = [json.loads(line) for line in failure_file]

        self.assertEqual(3, executor.calls)
        self.assertTrue(all(f["failure_type"] == "model_invalid_action_sequence" for f in failures))

    def test_executor_empty_done_completes_intent_without_review_or_phase3(self):
        planner = EmptyDonePlanner()
        executor = EmptyDoneThenMoveExecutor()
        oracle = NoopOracle()
        oracle.evaluate_failure = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("empty done must not enter Phase 4")
        )
        runner = BranchRunner(
            planner, oracle, ".", executor_agent=executor, enable_phase2=False
        )
        env = CompletingAfterMoveEnv()

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            result = runner.run(
                BranchConfig("ep", "main", None),
                env,
                self._episode(tmp),
            )

        self.assertEqual("task_complete", result.termination_reason)
        self.assertEqual(2, executor.calls)
        self.assertEqual(0, planner.review_calls)
        self.assertEqual(1, sum(action == "MoveAhead" for action, _ in env.step_calls))
        completed = planner.plan_calls[1]["intent_history"][-1]
        self.assertEqual("approach Table", completed["intent"])
        self.assertEqual("Table", completed["target"])
        self.assertTrue(completed["completed"])
        self.assertEqual([], completed["steps"])
        self.assertEqual("Already within approach range.", completed["completion_reason"])

    def test_repeated_empty_done_intent_fails_without_review_or_environment_step(self):
        planner = EmptyDonePlanner(repeat_same_intent=True)
        executor = AlwaysEmptyDoneExecutor()
        runner = BranchRunner(
            planner, NoopOracle(), ".", executor_agent=executor, enable_phase2=False
        )
        env = FailingMoveEnv()

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            with self.assertRaisesRegex(RuntimeError, "already completed without actions"):
                runner.run(
                    BranchConfig("ep", "main", None),
                    env,
                    self._episode(tmp),
                )
            with open(f"{tmp}/ep/failures_main.jsonl", encoding="utf-8") as failure_file:
                failures = [json.loads(line) for line in failure_file]

        self.assertEqual(4, executor.calls)
        self.assertEqual(0, planner.review_calls)
        self.assertEqual(3, len(failures))
        self.assertTrue(
            all(f["failure_type"] == "model_repeated_completed_intent" for f in failures)
        )

    def test_executor_malformed_status_is_nonexecuted_model_error(self):
        message = BranchRunner._validate_executor_result(
            [{"action": "MoveAhead"}], "retry", "executor"
        )

        self.assertIsNotNone(message)
        self.assertIn("status 'retry' is invalid", message)

    def test_reviewer_malformed_correction_is_nonexecuted_model_error(self):
        message = BranchRunner._validate_executor_result(
            [{"action": "MoveAhead", "params": []}], "done", "reviewer correction"
        )

        self.assertIsNotNone(message)
        self.assertIn("reviewer correction", message)
        self.assertIn("params must be a JSON object", message)

    def test_action_contract_rejects_missing_interaction_params_and_bad_repeat(self):
        self.assertIn("objectType", BranchRunner._validate_standalone_action(
            "PutObject", {}, 0
        ))
        self.assertIn("receptacleType", BranchRunner._validate_standalone_action(
            "PutObject", {"objectType": "Apple"}, 0
        ))
        self.assertIn("unsupported action", BranchRunner._validate_action_sequence(
            [{"action": "RotateLeft", "repeat": 2}], "sequence"
        ))
        self.assertIn("maximum is 200", BranchRunner._validate_action_sequence(
            [{"action": "MoveAhead", "repeat": 201}], "sequence"
        ))
        self.assertIn("maximum is 12", BranchRunner._validate_action_sequence(
            [{"action": "MoveAhead"}] * 13, "sequence"
        ))

    def test_camera_sequence_rejects_look_actions_outside_ai2thor_bounds(self):
        self.assertIsNotNone(
            BranchRunner._validate_camera_horizon_sequence(
                [{"action": "LookDown"}], 60
            )
        )
        self.assertIsNotNone(
            BranchRunner._validate_camera_horizon_sequence(
                [{"action": "LookUp"}], 330
            )
        )
        self.assertIsNone(
            BranchRunner._validate_camera_horizon_sequence(
                [{"action": "LookDown"}], 30
            )
        )
        self.assertIsNone(
            BranchRunner._validate_camera_horizon_sequence(
                [{"action": "LookUp"}], 0
            )
        )

    # Legacy candidate-pool coverage is archived in deprecated/tests.
    def legacy_phase2_closes_visible_open_receptacle_before_put(self):
        env = ActionAwarePutEnv()
        runner = BranchRunner(
            PutIntoOpenReceptacleAgent(),
            SelectingOracle("close_open_receptacle_before_put:Microwave|1"), ".",
            enable_phase2=True, enable_fork=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            ep = self._episode(tmp)
            ep.data["pddl_params"] = {"object_target": "Egg", "parent_target": "Microwave"}
            runner.run(
                BranchConfig("ep", "main", None), env, ep, start_step_index=1,
            )

        self.assertIn(
            ("CloseObject", {"objectId": "Microwave|1", "forceAction": True}),
            env.controller.actions,
        )
        self.assertFalse(env.microwave_open)
        trap = ep.data["runtime_traps"][0]
        self.assertEqual("close_open_receptacle_before_put", trap["failure_type"])
        self.assertEqual("triggered", trap["status"])
        self.assertEqual("main__s1", trap["triggered_at_step_id"])
        put_step = ep.data["steps"][-1]
        self.assertTrue(put_step["oracle_injection_decision"]["decided_to_inject"])
        self.assertEqual(trap["trap_id"], put_step["triggered_trap_id"])
        self.assertIn("CLOSED", put_step["error_message"])

    def legacy_phase2_executes_the_oracle_selected_valid_candidate(self):
        env = ActionAwarePutEnv()
        oracle = SelectingOracle("close_open_receptacle_before_put:Microwave|1")
        runner = BranchRunner(
            PutIntoOpenReceptacleAgent(), oracle, ".",
            enable_phase2=True, enable_fork=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            ep = self._episode(tmp)
            ep.data["pddl_params"] = {"object_target": "Egg", "parent_target": "Microwave"}
            runner.run(BranchConfig("ep", "main", None), env, ep, start_step_index=1)

        self.assertEqual(1, len(oracle.calls))
        self.assertEqual("close_open_receptacle_before_put:Microwave|1", oracle.calls[0]["candidates"][0]["candidate_id"])
        self.assertTrue(ep.data["steps"][-1]["oracle_injection_decision"]["decided_to_inject"])
        self.assertEqual("agentic_selector", ep.data["runtime_traps"][0]["created_by"])

    def legacy_phase2_rejects_unknown_oracle_candidate_without_mutating_environment(self):
        env = ActionAwarePutEnv()
        oracle = SelectingOracle("made_up_candidate")
        runner = BranchRunner(
            PutIntoOpenReceptacleAgent(), oracle, ".",
            enable_phase2=True, enable_fork=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            ep = self._episode(tmp)
            ep.data["pddl_params"] = {"object_target": "Egg", "parent_target": "Microwave"}
            runner.run(BranchConfig("ep", "main", None), env, ep, start_step_index=1)

        self.assertTrue(env.microwave_open)
        self.assertEqual([], ep.data["runtime_traps"])
        put_step = next(step for step in ep.data["steps"] if step["action"] == "PutObject")
        decision = put_step["oracle_injection_decision"]
        self.assertFalse(decision["decided_to_inject"])
        self.assertEqual("unknown_candidate", decision["reason"])

    def legacy_phase2_does_not_consult_oracle_while_a_trap_is_unresolved(self):
        env = ActionAwarePutEnv()
        oracle = SelectingOracle("close_open_receptacle_before_put:Microwave|1")
        runner = BranchRunner(
            PutIntoOpenReceptacleAgent(), oracle, ".",
            enable_phase2=True, enable_fork=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            ep = self._episode(tmp)
            ep.add_runtime_trap({
                "trap_id": "trap_main_prior",
                "branch_id": "main",
                "candidate_id": "close_open_container_before_pickup:Cabinet|1",
                "failure_type": "close_open_container_before_pickup",
                "difficulty": {"level": "low", "recovery_steps": 1},
                "status": "active",
                "injection": {"method": "close_container", "params": {"object_id": "Cabinet|1"}},
            })
            ep.data["pddl_params"] = {"object_target": "Egg", "parent_target": "Microwave"}
            result = runner.run(BranchConfig("ep", "main", None), env, ep, start_step_index=1)

        self.assertEqual("task_complete", result.termination_reason)
        self.assertEqual([], oracle.calls)
        self.assertEqual([], env.controller.actions)

    def test_legacy_recovered_verdict_becomes_executable_recovery(self):
        self.assertEqual("recoverable", _normalize_recovery_verdict("recovered"))
        self.assertEqual("recoverable", _normalize_recovery_verdict("recoverable"))
        self.assertEqual("unrecoverable", _normalize_recovery_verdict("unrecoverable"))

    def legacy_turn_on_lamp_recovers_only_after_turning_it_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = self._episode(tmp)
            ep.add_runtime_trap({
                "trap_id": "trap_main_1",
                "branch_id": "main",
                "candidate_id": "turn_on_lamp_before_toggle_on:FloorLamp|1",
                "failure_type": "turn_on_lamp_before_toggle_on",
                "status": "triggered",
                "injection": {
                    "method": "turn_on_lamp",
                    "params": {"object_id": "FloorLamp|1"},
                },
                "recovery_action": {
                    "action": "ToggleObjectOff",
                    "params": {"objectType": "FloorLamp"},
                },
            })

            wrong_action = ep.recover_runtime_trap(
                "main", "ToggleObjectOn", {"objectId": "FloorLamp|1"}, "main__s2",
            )
            recovered = ep.recover_runtime_trap(
                "main", "ToggleObjectOff", {"objectId": "FloorLamp|1"}, "main__s3",
            )

        self.assertEqual([], wrong_action)
        self.assertEqual(["trap_main_1"], recovered)
        trap = ep.data["runtime_traps"][0]
        self.assertEqual("recovered", trap["status"])
        self.assertEqual("main__s3", trap["recovered_at_step_id"])

    def legacy_temporary_countertop_placement_recovers_by_picking_up_and_retrying(self):
        env = ActionAwareTemporaryPlacementEnv()
        oracle = OneShotRecoverableOracle(
            "temporarily_place_held_object_before_put:Egg|1:CounterTop|1"
        )
        runner = BranchRunner(
            TemporaryPlacementRecoveryAgent(), oracle, ".",
            enable_phase2=True, enable_fork=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            ep = self._episode(tmp)
            ep.data["pddl_params"] = {"object_target": "Egg", "parent_target": "Microwave"}
            result = runner.run(BranchConfig("ep", "main", None), env, ep, start_step_index=1)

        self.assertEqual("task_complete", result.termination_reason)
        trap = ep.data["runtime_traps"][0]
        self.assertEqual("temporarily_place_held_object_before_put", trap["failure_type"])
        self.assertEqual(
            {"level": "medium", "recovery_steps": 2},
            trap["difficulty"],
        )
        self.assertEqual("recovered", trap["status"])
        self.assertEqual("PickupObject", trap["recovered_by_action"]["action"])
        self.assertEqual("CounterTop|1", trap["injection"]["params"]["receptacle_id"])
        actions = [step["action"] for step in ep.data["steps"]]
        self.assertEqual(["PutObject", "PickupObject", "PutObject", "Done"], actions)

    def legacy_explicit_put_cascade_keeps_second_trap_active_until_first_recovers(self):
        env = ActionAwareTemporaryPlacementEnv()
        oracle = CascadeSelectingOracle([
            "empty_hand_then_closed_receptacle:Egg|1:Microwave|1:CounterTop|1"
        ])
        runner = BranchRunner(
            TemporaryThenCloseRecoveryAgent(), oracle, ".",
            enable_phase2=True, enable_fork=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            ep = self._episode(tmp)
            ep.data["pddl_params"] = {"object_target": "Egg", "parent_target": "Microwave"}
            result = runner.run(BranchConfig("ep", "main", None), env, ep, start_step_index=1)

        self.assertEqual("task_complete", result.termination_reason)
        self.assertEqual(2, len(oracle.calls))
        trap = ep.data["runtime_traps"][0]
        self.assertEqual("empty_hand_then_closed_receptacle", trap["failure_type"])
        self.assertEqual({"level": "high", "recovery_steps": 2}, trap["difficulty"])
        self.assertEqual("recovered", trap["status"])
        self.assertEqual(2, len(trap["trigger_events"]))
        self.assertEqual(2, len(trap["recovery_events"]))
        self.assertEqual("PickupObject", trap["recovery_events"][0]["action"])
        self.assertEqual("OpenObject", trap["recovery_events"][1]["action"])
        actions = [step["action"] for step in ep.data["steps"]]
        self.assertEqual(
            ["PutObject", "PickupObject", "PutObject", "OpenObject", "PutObject", "Done"],
            actions,
        )

    def legacy_phase2_closes_receptacle_before_first_put_in_move_sequence(self):
        env = ActionAwarePutEnv()
        runner = BranchRunner(
            PutIntoOpenReceptacleSequenceAgent(),
            SelectingOracle("close_open_receptacle_before_put:Microwave|1"), ".",
            enable_phase2=True, enable_fork=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            ep = self._episode(tmp)
            ep.data["pddl_params"] = {"object_target": "Egg", "parent_target": "Microwave"}
            runner.run(
                BranchConfig("ep", "main", None), env, ep, start_step_index=1,
            )

        trap = ep.data["runtime_traps"][0]
        self.assertEqual("triggered", trap["status"])
        self.assertEqual("main__s1", trap["triggered_at_step_id"])
        self.assertIn("CLOSED", ep.data["steps"][-1]["error_message"])

    def legacy_phase2_closes_open_parent_container_before_pickup(self):
        env = ActionAwarePickupEnv()
        runner = BranchRunner(
            PickupFromOpenContainerAgent(),
            SelectingOracle("close_open_container_before_pickup:Microwave|1"), ".",
            enable_phase2=True, enable_fork=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner.output_dir = tmp
            ep = self._episode(tmp)
            ep.data["pddl_params"] = {"object_target": "Apple", "parent_target": "Microwave"}
            runner.run(
                BranchConfig("ep", "main", None), env, ep, start_step_index=1,
            )

        trap = ep.data["runtime_traps"][0]
        self.assertEqual("close_open_container_before_pickup", trap["failure_type"])
        self.assertEqual("triggered", trap["status"])
        self.assertIn("specified visibility", ep.data["steps"][-1]["error_message"])


class _FakeEp:
    """Minimal EpisodeManager stand-in for step-id/fork unit tests."""
    def __init__(self, steps=None):
        self.episode_id = "ep_test"
        self.data = {"steps": steps or []}


class StepIdUniquenessTest(unittest.TestCase):
    def test_sid_format_and_filesystem_safe(self):
        from src.branch_runner import _sid
        self.assertEqual("main__s0", _sid("main", 0))
        self.assertEqual("fork_s5_main__s3", _sid("fork_s5_main", 3))
        # Must be usable as a PNG filename on Windows: no reserved chars.
        for ch in '<>:"/\\|?*':
            self.assertNotIn(ch, _sid("fork_s5_main", 3))

    def test_sid_globally_unique_across_branches(self):
        from src.branch_runner import _sid
        # Same local index on different branches must not collide.
        self.assertNotEqual(_sid("main", 0), _sid("fork_s5_main", 0))

    def test_last_branch_step_id_none_when_empty(self):
        from src.branch_runner import _last_branch_step_id
        self.assertIsNone(_last_branch_step_id(_FakeEp(), "main"))

    def test_last_branch_step_id_returns_latest_on_branch(self):
        from src.branch_runner import _last_branch_step_id
        ep = _FakeEp([
            {"branch_id": "main", "step_id": "main__s0"},
            {"branch_id": "main", "step_id": "main__s1"},
            {"branch_id": "fork_s1_main", "step_id": "fork_s1_main__s0"},
        ])
        self.assertEqual("main__s1", _last_branch_step_id(ep, "main"))
        self.assertEqual("fork_s1_main__s0", _last_branch_step_id(ep, "fork_s1_main"))

    def test_last_branch_step_id_bridges_old_format_seam(self):
        # Resuming an episode recorded under the old bare-'s<idx>' scheme:
        # parent must point at the actual last id, not a recomputed prefix.
        from src.branch_runner import _last_branch_step_id
        ep = _FakeEp([
            {"branch_id": "main", "step_id": "s0"},
            {"branch_id": "main", "step_id": "s1"},
        ])
        self.assertEqual("s1", _last_branch_step_id(ep, "main"))


class ForkCounterfactualSourceTest(unittest.TestCase):
    """_build_fork_task must fork the structured counterfactual it is handed,
    whether that comes from EB (AC) or the Oracle gold (PA/WA)."""

    def _runner(self):
        return BranchRunner(object(), NoopOracle(), ".")

    def _history(self):
        return [
            {"branch_id": "main", "step_id": "main__s0",
             "step_index_in_branch": 0, "parent_step_id": None,
             "action": "PickupObject"},
            {"branch_id": "main", "step_id": "main__s1",
             "step_index_in_branch": 1, "parent_step_id": "main__s0",
             "action": "PutObject"},
        ]

    def _config(self):
        return BranchConfig(episode_id="ep_test", branch_id="main",
                            parent_branch_id=None)

    def test_structured_counterfactual_targets_named_step(self):
        runner = self._runner()
        cf = {"target_step": 0,
              "alternative_action": {"action": "OpenObject",
                                     "params": {"objectType": "Microwave"}},
              "reasoning": "should have opened it first"}
        task = runner._build_fork_task(
            config=self._config(), current_step_idx=1,
            counterfactual=cf, fallback_recovery={},
            history=self._history(), ep=_FakeEp(),
        )
        self.assertIsNotNone(task)
        self.assertEqual("main__s0", task["fork_config"]["replaces_step_id"])
        self.assertEqual({"action": "OpenObject",
                          "params": {"objectType": "Microwave"}},
                         task["fork_config"]["alternative_action"])
        # origin id must be branch-prefixed and globally unique
        self.assertEqual("main__s1", task["fork_config"]["origin_step_id"])

    def test_oracle_gold_is_forkable_same_as_eb(self):
        # A PA/WA gold has the identical structured shape and must build a task.
        runner = self._runner()
        gold = {"target_step": 1,
                "alternative_action": {"action": "MoveBack", "params": {}},
                "reasoning": "back off before retrying"}
        task = runner._build_fork_task(
            config=self._config(), current_step_idx=1,
            counterfactual=gold, fallback_recovery={},
            history=self._history(), ep=_FakeEp(),
        )
        self.assertIsNotNone(task)
        self.assertEqual("main__s1", task["fork_config"]["replaces_step_id"])
        self.assertEqual({"action": "MoveBack", "params": {}},
                         task["fork_config"]["alternative_action"])

    def test_missing_target_step_in_history_returns_none(self):
        runner = self._runner()
        cf = {"target_step": 99,
              "alternative_action": {"action": "MoveBack", "params": {}},
              "reasoning": "x"}
        task = runner._build_fork_task(
            config=self._config(), current_step_idx=1,
            counterfactual=cf, fallback_recovery={},
            history=self._history(), ep=_FakeEp(),
        )
        self.assertIsNone(task)

    def test_legacy_string_counterfactual_uses_fallback_recovery(self):
        runner = self._runner()
        task = runner._build_fork_task(
            config=self._config(), current_step_idx=1,
            counterfactual="I should have opened the microwave",
            fallback_recovery={"action": "OpenObject",
                               "params": {"objectType": "Microwave"}},
            history=self._history(), ep=_FakeEp(),
        )
        self.assertIsNotNone(task)
        # heuristic picks the most recent interaction step (PutObject @ main__s1)
        self.assertEqual("main__s1", task["fork_config"]["replaces_step_id"])
        self.assertEqual({"action": "OpenObject",
                          "params": {"objectType": "Microwave"}},
                         task["fork_config"]["alternative_action"])


if __name__ == "__main__":
    unittest.main()
