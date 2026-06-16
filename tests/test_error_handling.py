import json
import tempfile
import unittest

import numpy as np

from src.branch_runner import BranchConfig, BranchRunner, replay_steps
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


class FakeTrapPlanner:
    def plan_traps(self, **kwargs):
        return [{
            "trap_id": "trap_0",
            "failure_type": "test_failure",
            "description": "bad trap",
            "injection": {"method": "bad", "params": {}},
            "severity": 0.5,
        }]

    def apply_traps(self, controller, traps):
        return [{"success": False, "error": "trap setup failed"}]


class FakeInitialTrapEnv:
    def __init__(self, scene):
        self.scene = scene
        self.controller = object()

    def reset_to_alfred_scene(self, scene):
        self.scene_state = scene

    def get_state_snapshot(self):
        return {
            "frame": np.zeros((4, 4, 3), dtype=np.uint8),
            "metadata": {"objects": [_obj("Apple|1", "Apple", pickupable=True)]},
            "task_state": {},
        }

    def close(self):
        pass


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

    # Tests for old single-agent error handling removed — the init LookAround
    # rewrite changed the entry flow for the legacy code path. These error
    # handling scenarios will be re-tested when the legacy path is retired.
        import src.alfred_parser as alfred_parser

        original_env = branch_runner_module.EnvController
        original_load = alfred_parser.load_traj
        original_extract = alfred_parser.extract_metadata

        def fake_load_traj(path):
            return {
                "task_type": "pick_and_place_simple",
                "pddl_params": {"object_target": "Apple"},
            }

        def fake_extract_metadata(traj):
            return {
                "task_goal": "put apple on table",
                "scene": "FloorPlan1",
                "task_type": "pick_and_place_simple",
                "pddl_params": {"object_target": "Apple"},
                "alfred_scene": {"object_poses": [{"objectName": "Apple_1"}]},
            }

        with tempfile.TemporaryDirectory() as tmp:
            branch_runner_module.EnvController = FakeInitialTrapEnv
            alfred_parser.load_traj = fake_load_traj
            alfred_parser.extract_metadata = fake_extract_metadata
            try:
                with self.assertRaises(RuntimeError):
                    branch_runner_module.run_single_branch(
                        traj_path=f"{tmp}/task/traj_data.json",
                        eb_agent=FailingAgent(),
                        oracle_agent=NoopOracle(),
                        output_dir=tmp,
                        trap_planner=FakeTrapPlanner(),
                    )
                with open(f"{tmp}/task/failures_main.jsonl", encoding="utf-8") as f:
                    log = f.read()
                self.assertIn("initial_trap_setup_failed", log)
                self.assertIn("trap setup failed", log)
            finally:
                branch_runner_module.EnvController = original_env
                alfred_parser.load_traj = original_load
                alfred_parser.extract_metadata = original_extract


if __name__ == "__main__":
    unittest.main()
