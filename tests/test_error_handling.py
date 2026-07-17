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
        self.assertEqual(1, planner.review_calls)
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

    def test_initial_trap_setup_failure_is_logged_and_raises(self):
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
