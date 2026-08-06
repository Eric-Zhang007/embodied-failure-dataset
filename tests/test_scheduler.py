"""Deterministic tests for Scheduler accounting and worker isolation."""
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock
from collections import deque

from src.episode_manager import EpisodeManager
from src.scheduler import Scheduler, SchedulerConfig, BranchConfig, BranchResult
from src.branch_runner import BranchRunner, ReplayUnavailable


SEVEN_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
    "pick_and_place_with_movable_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)


def _task(ep_id="ep1"):
    return {
        "traj_path": f"/fake/{ep_id}/traj_data.json",
        "episode_id": ep_id,
        "meta": {"scene": "FloorPlan1"},
        "base_step_count": 5,
        "branch_config": BranchConfig(episode_id=ep_id, branch_id="main", parent_branch_id=None),
    }


def _lane_task(task_type: str, index: int):
    task = _task(f"{task_type}-{index}")
    task["meta"]["alfred_task_type"] = task_type
    return task


def _success_result():
    return BranchResult(branch_id="main", termination_reason="task_complete",
                        total_steps=3, fork_tasks=[], fork_source_step_ids=[])


def _skip_result():
    return BranchResult(branch_id="main", termination_reason="skipped",
                        total_steps=0, fork_tasks=[], fork_source_step_ids=[])


def _pre_satisfied_result():
    return BranchResult(branch_id="main", termination_reason="skipped_pre_satisfied",
                        total_steps=0, fork_tasks=[], fork_source_step_ids=[])


def _crash_result():
    return BranchResult(branch_id="main", termination_reason="worker_crash:boom",
                        total_steps=0, fork_tasks=[], fork_source_step_ids=[])


def _terminal_result(reason="step_hard_limit"):
    return BranchResult(branch_id="main", termination_reason=reason,
                        total_steps=200, fork_tasks=[], fork_source_step_ids=[])


class SchedulerAccountingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._warmup_patcher = patch("src.vlm_client.VLMClient.openai")
        cls._mock_vlm_client_cls = cls._warmup_patcher.start()
        mock_client = MagicMock()
        mock_client.chat_text.return_value = "OK"
        cls._mock_vlm_client_cls.return_value = mock_client

    @classmethod
    def tearDownClass(cls):
        cls._warmup_patcher.stop()

    def setUp(self):
        self.config = SchedulerConfig(
            data_dir="/nonexistent",
            output_dir="/tmp/test_scheduler",
            max_episodes=0,
            max_parallel=1,
            api_key="sk-test",
            api_base_url="https://api.fullcupai.com/v1",
        )

    def test_stats_all_success(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a"), _task("b"), _task("c")])
        scheduler._lock = MagicMock()

        with patch.object(scheduler, "_run_branch_worker", return_value=_success_result()):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(scheduler.stats["completed"], 3)
        self.assertEqual(scheduler.stats["failed"], 0)
        self.assertEqual(scheduler.stats["skipped"], 0)

    def test_stats_mixed_skipped_and_failed(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a"), _task("b"), _task("c")])
        scheduler._lock = MagicMock()

        def side_effect(task, n, total):
            ep = task["episode_id"]
            if ep == "a":
                return _success_result()
            if ep == "b":
                return _skip_result()
            if ep == "c":
                return _crash_result()
            raise RuntimeError("unknown")

        with patch.object(scheduler, "_run_branch_worker", side_effect=side_effect):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(scheduler.stats["completed"], 1)
        self.assertEqual(scheduler.stats["skipped"], 1)
        self.assertEqual(scheduler.stats["failed"], 1)

    def test_task_success_is_distinct_from_other_terminal_outcomes(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("success"), _task("limited")])
        scheduler._lock = MagicMock()

        def side_effect(task, n, total):
            return _success_result() if task["episode_id"] == "success" else _terminal_result()

        with patch.object(scheduler, "_run_branch_worker", side_effect=side_effect):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(2, scheduler.stats["terminal"])
        self.assertEqual(1, scheduler.stats["task_complete"])
        self.assertEqual(0, scheduler.stats["failed"])

    def test_pre_satisfied_episode_is_skipped_not_a_task_success(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("pre-satisfied")])
        scheduler._lock = MagicMock()

        with patch.object(scheduler, "_run_branch_worker", return_value=_pre_satisfied_result()):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(0, scheduler.stats["terminal"])
        self.assertEqual(0, scheduler.stats["task_complete"])
        self.assertEqual(1, scheduler.stats["skipped"])

    def test_worker_exception_increments_failed_only(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a"), _task("b")])
        scheduler._lock = MagicMock()

        def side_effect(task, n, total):
            if task["episode_id"] == "a":
                return _success_result()
            raise ValueError("simulated crash")

        with patch.object(scheduler, "_run_branch_worker", side_effect=side_effect):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(scheduler.stats["completed"], 1)
        self.assertEqual(scheduler.stats["failed"], 1)
        self.assertEqual(scheduler.stats["skipped"], 0)

    def test_main_worker_crash_retries_and_then_completes(self):
        scheduler = Scheduler(self.config)
        scheduler.config.max_worker_retries = 1
        scheduler.queue = deque([_task("retry-me")])
        scheduler._lock = MagicMock()

        with patch.object(
            scheduler,
            "_run_branch_worker",
            side_effect=[_crash_result(), _success_result()],
        ) as worker:
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(2, worker.call_count)
        self.assertEqual(1, scheduler.stats["completed"])
        self.assertEqual(0, scheduler.stats["failed"])

    def test_task_lanes_start_one_worker_per_alfred_task_type(self):
        scheduler = Scheduler(self.config)
        scheduler.config.max_parallel = 7
        scheduler.config.task_lanes = True
        scheduler.queue = deque(
            _lane_task(task_type, index)
            for task_type in SEVEN_TASK_TYPES
            for index in (1, 2)
        )
        scheduler._lock = MagicMock()
        seen = []

        def run_worker(task, n, total):
            seen.append(task["meta"]["alfred_task_type"])
            return _success_result()

        with patch.object(scheduler, "_run_branch_worker", side_effect=run_worker):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(14, len(seen))
        self.assertEqual(set(SEVEN_TASK_TYPES), set(seen[:7]))
        self.assertEqual(2, seen.count("pick_and_place_simple"))

    def test_task_lane_retries_before_advancing_to_its_next_episode(self):
        scheduler = Scheduler(self.config)
        scheduler.config.max_parallel = 7
        scheduler.config.max_worker_retries = 1
        scheduler.config.task_lanes = True
        scheduler.queue = deque(
            _lane_task(task_type, index)
            for task_type in SEVEN_TASK_TYPES
            for index in (1, 2)
        )
        scheduler._lock = MagicMock()
        lane_seen = []
        first_id = "pick_and_place_simple-1"

        def run_worker(task, n, total):
            if task["meta"]["alfred_task_type"] == "pick_and_place_simple":
                lane_seen.append(task["episode_id"])
                if task["episode_id"] == first_id and lane_seen.count(first_id) == 1:
                    return _crash_result()
            return _success_result()

        with patch.object(scheduler, "_run_branch_worker", side_effect=run_worker):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual([first_id, first_id, "pick_and_place_simple-2"], lane_seen)

    def test_skip_is_not_counted_as_completed(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a")])
        scheduler._lock = MagicMock()

        with patch.object(scheduler, "_run_branch_worker", return_value=_skip_result()):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(scheduler.stats["completed"], 0)
        self.assertEqual(scheduler.stats["skipped"], 1)

    def test_crash_is_not_counted_as_completed(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a")])
        scheduler._lock = MagicMock()

        with patch.object(scheduler, "_run_branch_worker", return_value=_crash_result()):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(scheduler.stats["completed"], 0)
        self.assertEqual(scheduler.stats["failed"], 1)

    def test_worker_isolation_creates_separate_agents(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a")])
        scheduler._lock = MagicMock()

        with patch.object(scheduler, "_create_agents") as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock(), MagicMock())
            with patch.object(scheduler, "load_tasks"):
                with patch("src.scheduler.run_single_branch", return_value=_success_result()):
                    scheduler.run()

        self.assertEqual(mock_create.call_count, 1)

    def test_main_resume_replay_unavailable_is_skipped_without_retry(self):
        with tempfile.TemporaryDirectory() as output_dir:
            scheduler = Scheduler(SchedulerConfig(
                output_dir=output_dir,
                api_key="sk-test",
            ))
            task = _task("resume-replay-unavailable")
            manager = EpisodeManager(task["episode_id"], output_dir, {
                "task_goal": "g", "scene": "FloorPlan1",
                "task_type": "pick_and_place_simple",
            })
            manager.add_step({
                "step_id": "main__s0", "branch_id": "main",
                "step_index_in_branch": 0, "action": "MoveAhead", "success": True,
            })
            manager.set_status("interrupted")

            with patch.object(
                BranchRunner, "resume", side_effect=ReplayUnavailable("bad checkpoint"),
            ):
                result = scheduler._run_branch_worker(task, 1, 1)

            self.assertEqual("replay_unavailable", result.termination_reason)
            saved = EpisodeManager.load(manager.file_path)
            self.assertEqual("completed", saved.data["status"])
            self.assertEqual(
                "replay_unavailable",
                saved.data["final_outcome"]["main_branch"]["termination_reason"],
            )

    def test_parallel_workers_each_create_agents(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a"), _task("b"), _task("c")])
        scheduler.config.max_parallel = 3
        scheduler._lock = MagicMock()

        with patch.object(scheduler, "_create_agents") as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock(), MagicMock())
            with patch.object(scheduler, "load_tasks"):
                with patch("src.scheduler.run_single_branch", return_value=_success_result()):
                    scheduler.run()

        self.assertEqual(mock_create.call_count, 3)

    def test_report_remaining_decreases_with_progress(self):
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a"), _task("b")])
        scheduler.stats = {"completed": 0, "skipped": 0, "failed": 0}
        scheduler._lock = MagicMock()

        with patch.object(scheduler, "_run_branch_worker", return_value=_success_result()):
            with patch.object(scheduler, "load_tasks"):
                scheduler.run()

        self.assertEqual(scheduler.stats["completed"], 2)

    def test_fork_tasks_queued_from_completed_parents(self):
        fork_data = {"episode_id": "a", "branch_id": "fork1",
                      "parent_branch_id": "main", "shared_context_step_ids": [],
                      "diverges_at_step_id": "s3", "fork_config": {}}
        result_with_fork = BranchResult(
            branch_id="main", termination_reason="task_complete",
            total_steps=5, fork_tasks=[fork_data], fork_source_step_ids=[],
        )

        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a")])
        scheduler._lock = MagicMock()

        with patch.object(scheduler, "_run_branch_worker", return_value=result_with_fork):
            with patch.object(scheduler, "load_tasks"):
                with patch.object(scheduler, "_run_fork", return_value=_success_result()):
                    scheduler.run()

        self.assertEqual(scheduler.stats["completed"], 2)  # main + fork

    def test_parent_result_executes_each_fork_once(self):
        fork_data = {
            "episode_id": "a",
            "branch_id": "fork_s3_main",
            "parent_branch_id": "main",
            "shared_context_step_ids": [],
            "diverges_at_step_id": "main__s2",
            "fork_config": {},
        }
        main_result = BranchResult(
            branch_id="main", termination_reason="task_complete", total_steps=3,
            fork_tasks=[fork_data], fork_source_step_ids=[],
        )
        fork_result = BranchResult(
            branch_id="fork_s3_main", termination_reason="task_complete", total_steps=3,
            fork_tasks=[], fork_source_step_ids=[],
        )
        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a")])
        seen = []

        def run_task(task, n, total):
            branch_id = task["branch_config"].branch_id
            seen.append(branch_id)
            if branch_id == "main":
                return main_result
            return fork_result

        with patch.object(scheduler, "load_tasks"):
            with patch.object(scheduler, "_run_task_worker", side_effect=run_task):
                scheduler.run()

        self.assertEqual(["main", "fork_s3_main"], seen)
        self.assertEqual(2, scheduler.stats["completed"])

    def test_task_lanes_execute_a_returned_fork_once(self):
        fork_data = {
            "episode_id": "pick_and_place_simple-1",
            "branch_id": "fork_s3_main",
            "parent_branch_id": "main",
            "shared_context_step_ids": [],
            "diverges_at_step_id": "main__s2",
            "fork_config": {},
        }
        scheduler = Scheduler(self.config)
        scheduler.config.max_parallel = 7
        scheduler.config.task_lanes = True
        scheduler.queue = deque(_lane_task(task_type, 1) for task_type in SEVEN_TASK_TYPES)
        seen = []

        def run_task(task, n, total):
            config = task["branch_config"]
            if config.branch_id == "fork_s3_main":
                seen.append(config.branch_id)
                return BranchResult(config.branch_id, "task_complete", 3, [], [])
            if task["episode_id"] == "pick_and_place_simple-1":
                return BranchResult("main", "task_complete", 3, [fork_data], [])
            return _success_result()

        with patch.object(scheduler, "load_tasks"):
            with patch.object(scheduler, "_run_task_worker", side_effect=run_task):
                scheduler.run()

        self.assertEqual(["fork_s3_main"], seen)

    def test_task_lanes_drop_callback_descendants_when_parent_crashes(self):
        fork_data = {
            "episode_id": "pick_and_place_simple-1",
            "branch_id": "fork_s3_main",
            "parent_branch_id": "main",
            "shared_context_step_ids": [],
            "diverges_at_step_id": "main__s2",
            "fork_config": {},
        }
        scheduler = Scheduler(self.config)
        scheduler.config.max_parallel = 7
        scheduler.config.max_worker_retries = 0
        scheduler.config.task_lanes = True
        scheduler.queue = deque(_lane_task(task_type, 1) for task_type in SEVEN_TASK_TYPES)
        seen = []

        def run_task(task, n, total):
            config = task["branch_config"]
            seen.append(config.branch_id)
            if task["episode_id"] == "pick_and_place_simple-1" and config.branch_id == "main":
                callback_queue = task.get("_fork_queue")
                if callback_queue is not None:
                    callback_queue.put(scheduler._fork_entry(task, fork_data))
                return _crash_result()
            return _success_result()

        with patch.object(scheduler, "load_tasks"):
            with patch.object(scheduler, "_run_task_worker", side_effect=run_task):
                scheduler.run()

        self.assertNotIn("fork_s3_main", seen)

    def test_default_max_parallel_is_four(self):
        self.assertEqual(SchedulerConfig().max_parallel, 4)

    def test_recursive_forks_run_in_same_pool(self):
        first_fork = {"episode_id": "a", "branch_id": "fork1",
                      "parent_branch_id": "main", "shared_context_step_ids": [],
                      "diverges_at_step_id": "s1", "fork_config": {}}
        second_fork = {"episode_id": "a", "branch_id": "fork2",
                       "parent_branch_id": "fork1", "shared_context_step_ids": [],
                       "diverges_at_step_id": "s2", "fork_config": {}}
        main_result = BranchResult("main", "task_complete", 1, [first_fork], [])
        fork_result = BranchResult("fork1", "task_complete", 1, [second_fork], [])
        final_result = BranchResult("fork2", "task_complete", 1, [], [])

        scheduler = Scheduler(self.config)
        scheduler.queue = deque([_task("a")])
        seen = []

        def run_task(task, n, total):
            branch_id = task["branch_config"].branch_id
            seen.append(branch_id)
            return {"main": main_result, "fork1": fork_result, "fork2": final_result}[branch_id]

        with patch.object(scheduler, "load_tasks"):
            with patch.object(scheduler, "_run_task_worker", side_effect=run_task):
                scheduler.run()

        self.assertEqual(seen, ["main", "fork1", "fork2"])
        self.assertEqual(scheduler.stats["completed"], 3)


class SchedulerLoadTasksTest(unittest.TestCase):
    def test_terminal_main_outcome_is_not_resumed_when_status_is_stale(self):
        with tempfile.TemporaryDirectory() as output_dir:
            metadata = {
                "task_goal": "put an AlarmClock on the Desk",
                "scene": "FloorPlan1",
                "task_type": "pick_and_place_simple",
            }
            manager = EpisodeManager("ep-terminal", output_dir, metadata)
            manager.update_final_outcome(
                {
                    "branch_id": "main",
                    "termination_reason": "dead_loop",
                    "total_steps": 5,
                },
                is_main=True,
            )
            manager.set_status("running", 999999)

            scheduler = Scheduler(SchedulerConfig(
                data_dir="/fake-data",
                output_dir=output_dir,
                api_key="sk-test",
            ))
            with patch("src.scheduler.glob.glob", return_value=["/fake/ep-terminal/traj_data.json"]):
                with patch("src.scheduler.load_traj", return_value={}):
                    with patch("src.scheduler.extract_metadata", return_value=metadata):
                        with patch.object(scheduler, "_pid_is_alive", return_value=False):
                            scheduler.load_tasks()

            self.assertEqual([], list(scheduler.queue))
            self.assertEqual("completed", EpisodeManager.load(manager.file_path).data["status"])


class ForkAltActionResolutionTest(unittest.TestCase):
    """_run_fork must resolve objectType->objectId before env.step, like the
    main loop. Otherwise structured EB/Oracle counterfactual actions (which
    carry objectType) fail with AI2-THOR 'target not found'."""

    @classmethod
    def setUpClass(cls):
        cls._warmup_patcher = patch("src.vlm_client.VLMClient.openai")
        mock_cls = cls._warmup_patcher.start()
        mock_client = MagicMock()
        mock_client.chat_text.return_value = "OK"
        mock_cls.return_value = mock_client

    @classmethod
    def tearDownClass(cls):
        cls._warmup_patcher.stop()

    def test_alt_action_objecttype_is_resolved_before_env_step(self):
        egg_id = "Egg|+01.67|+00.97|+02.02"
        objects = [{"objectType": "Egg", "objectId": egg_id,
                    "visibleBounds2D": True, "pickupable": True, "receptacle": False}]

        recorded = {}

        import numpy as np

        class FakeEnv:
            def __init__(self, scene=None):
                pass
            def reset_to_alfred_scene(self, *a, **k):
                pass
            def get_state_snapshot(self):
                return {"metadata": {"objects": objects},
                        "frame": np.zeros((4, 4, 3), dtype=np.uint8)}
            def step(self, action, **params):
                recorded["action"] = action
                recorded["params"] = params
                # Fail immediately so _run_fork short-circuits at the fork_failed
                # branch, BEFORE any reasoning-rewrite API call. We only care that
                # the params reaching env.step were resolved.
                return {"success": False, "error": "stop here",
                        "frame": np.zeros((4, 4, 3), dtype=np.uint8)}
            def close(self):
                pass

        config = SchedulerConfig(output_dir=tempfile.mkdtemp())
        with patch("src.scheduler.EnvController", FakeEnv):
            scheduler = Scheduler(config)
            ep_id = "epfork"
            out_file = os.path.join(config.output_dir, f"{ep_id}.json")
            # minimal episode file the fork loader can read
            manager = EpisodeManager(ep_id, config.output_dir, {
                "task_goal": "g", "scene": "FloorPlan1", "task_type": "heat",
                "alfred_scene": {"object_poses": [1]},
            })
            manager.set_status("completed")
            fork_cfg = BranchConfig(
                episode_id=ep_id, branch_id="fork_s1_main",
                parent_branch_id="main", shared_context_step_ids=[],
                diverges_at_step_id="main__s1",
                fork_config={
                    "replaces_step_id": "main__s1",
                    "origin_step_id": "main__s2",
                    "alternative_action": {"action": "PickupObject",
                                           "params": {"objectType": "Egg"}},
                    "counterfactual_text": "should have grabbed the egg",
                },
            )
            task = {"episode_id": ep_id, "meta": {"scene": "FloorPlan1"},
                    "traj_path": "", "base_step_count": 0,
                    "branch_config": fork_cfg}
            with patch("src.scheduler.replay_steps"):
                scheduler._run_fork(task, scheduler._create_agents())

        # objectType must be gone; a resolved objectId must be present.
        self.assertEqual("PickupObject", recorded["action"])
        self.assertEqual(egg_id, recorded["params"].get("objectId"))
        self.assertNotIn("objectType", recorded["params"])
        self.assertEqual("completed", EpisodeManager.load(out_file).data["status"])

    def test_existing_fork_branch_is_not_replayed_or_appended_again(self):
        config = SchedulerConfig(output_dir=tempfile.mkdtemp())
        scheduler = Scheduler(config)
        ep_id = "ep-existing-fork"
        manager = EpisodeManager(ep_id, config.output_dir, {
            "task_goal": "g", "scene": "FloorPlan1", "task_type": "heat",
            "alfred_scene": {"object_poses": [1]},
        })
        manager.add_step({
            "step_id": "fork_s1_main__s0", "branch_id": "fork_s1_main",
            "step_index_in_branch": 0, "action": "MoveAhead", "success": True,
        })
        fork_cfg = BranchConfig(
            episode_id=ep_id, branch_id="fork_s1_main", parent_branch_id="main",
            shared_context_step_ids=[], diverges_at_step_id="main__s0",
            fork_config={},
        )
        task = {
            "episode_id": ep_id, "meta": {"scene": "FloorPlan1"},
            "traj_path": "", "base_step_count": 0, "branch_config": fork_cfg,
        }

        with patch("src.scheduler.EnvController", side_effect=AssertionError("must not replay")):
            result = scheduler._run_fork(task, scheduler._create_agents())

        self.assertEqual("skipped_duplicate_fork", result.termination_reason)
        self.assertEqual(1, len(EpisodeManager.load(manager.file_path).get_steps_for_branch("fork_s1_main")))

    def test_incomplete_fork_after_restart_cancels_its_descendants(self):
        config = SchedulerConfig(output_dir=tempfile.mkdtemp())
        scheduler = Scheduler(config)
        ep_id = "ep-incomplete-fork"
        manager = EpisodeManager(ep_id, config.output_dir, {
            "task_goal": "g", "scene": "FloorPlan1", "task_type": "heat",
            "alfred_scene": {"object_poses": [1]},
        })
        fork_task = {
            "episode_id": ep_id, "branch_id": "fork_s1_main",
            "parent_branch_id": "main", "shared_context_step_ids": [],
            "diverges_at_step_id": "main__s0", "fork_config": {},
        }
        manager.add_pending_fork(fork_task)
        manager.add_pending_fork({
            "episode_id": ep_id, "branch_id": "fork_s2_fork_s1_main",
            "parent_branch_id": "fork_s1_main", "shared_context_step_ids": [],
            "diverges_at_step_id": "fork_s1_main__s0", "fork_config": {},
        })
        manager.add_step({
            "step_id": "fork_s1_main__s0", "branch_id": "fork_s1_main",
            "step_index_in_branch": 0, "action": "MoveAhead", "success": True,
        })
        task = {
            "episode_id": ep_id, "meta": {"scene": "FloorPlan1"},
            "traj_path": "", "base_step_count": 0,
            "branch_config": BranchConfig(**fork_task),
        }

        with patch("src.scheduler.EnvController", side_effect=AssertionError("must not replay")):
            result = scheduler._run_fork(task, scheduler._create_agents())

        self.assertEqual("fork_incomplete_before_restart", result.termination_reason)
        saved = EpisodeManager.load(manager.file_path)
        self.assertEqual([], saved.get_pending_fork_tasks())
        descendants = [
            entry for entry in saved.data["pending_forks"]
            if entry["branch_id"] == "fork_s2_fork_s1_main"
        ]
        self.assertEqual("cancelled", descendants[0]["state"])

    def test_nested_fork_replays_its_full_ancestor_context_in_lineage_order(self):
        """A child fork must replay main history before its parent fork root."""
        import numpy as np

        replayed_step_ids = []

        class FakeEnv:
            def __init__(self, scene=None):
                pass

            def reset_to_alfred_scene(self, *args, **kwargs):
                pass

            def get_state_snapshot(self):
                return {
                    "metadata": {"objects": []},
                    "frame": np.zeros((4, 4, 3), dtype=np.uint8),
                }

            def step(self, action, **params):
                return {
                    "success": False,
                    "error": "stop before BranchRunner",
                    "frame": np.zeros((4, 4, 3), dtype=np.uint8),
                }

            def close(self):
                pass

        config = SchedulerConfig(output_dir=tempfile.mkdtemp())
        scheduler = Scheduler(config)
        ep_id = "ep-nested-lineage"
        manager = EpisodeManager(ep_id, config.output_dir, {
            "task_goal": "g", "scene": "FloorPlan1", "task_type": "heat",
            "alfred_scene": {"object_poses": [1]},
        })
        manager.add_step({
            "step_id": "main__s0", "branch_id": "main",
            "step_index_in_branch": 0, "action": "LookAround", "success": True,
        })
        manager.add_step({
            "step_id": "main__s1", "branch_id": "main",
            "parent_step_id": "main__s0", "step_index_in_branch": 1,
            "action": "RotateRight", "success": True,
        })
        manager.add_step({
            "step_id": "fork_s2_main__s0", "branch_id": "fork_s2_main",
            "parent_step_id": "main__s1", "step_index_in_branch": 0,
            "action": "MoveBack", "success": True,
        })
        fork_cfg = BranchConfig(
            episode_id=ep_id,
            branch_id="fork_s1_fork_s2_main",
            parent_branch_id="fork_s2_main",
            shared_context_step_ids=[
                "main__s0", "main__s1", "fork_s2_main__s0",
            ],
            diverges_at_step_id="fork_s2_main__s0",
            fork_config={
                "replaces_step_id": "fork_s2_main__s1",
                "origin_step_id": "fork_s2_main__s2",
                "alternative_action": {"action": "MoveAhead", "params": {}},
            },
        )
        task = {
            "episode_id": ep_id,
            "meta": {"scene": "FloorPlan1"},
            "traj_path": "",
            "base_step_count": 0,
            "branch_config": fork_cfg,
        }

        def capture_replay(_env, steps, **_kwargs):
            replayed_step_ids.extend(step["step_id"] for step in steps)

        with patch("src.scheduler.EnvController", FakeEnv), \
             patch("src.scheduler.replay_steps", side_effect=capture_replay):
            scheduler._run_fork(task, scheduler._create_agents())

        self.assertEqual(
            ["main__s0", "main__s1", "fork_s2_main__s0"],
            replayed_step_ids,
        )

    def test_replay_unavailable_is_recorded_without_a_worker_crash(self):
        class FakeEnv:
            def __init__(self, scene=None):
                pass

            def reset_to_alfred_scene(self, *args, **kwargs):
                pass

            def close(self):
                pass

        config = SchedulerConfig(output_dir=tempfile.mkdtemp())
        scheduler = Scheduler(config)
        ep_id = "ep-replay-unavailable"
        manager = EpisodeManager(ep_id, config.output_dir, {
            "task_goal": "g", "scene": "FloorPlan1", "task_type": "heat",
            "alfred_scene": {"object_poses": [1]},
        })
        manager.add_step({
            "step_id": "main__s0", "branch_id": "main",
            "step_index_in_branch": 0, "action": "MoveAhead", "success": True,
        })
        fork_cfg = BranchConfig(
            episode_id=ep_id,
            branch_id="fork_s1_main",
            parent_branch_id="main",
            shared_context_step_ids=["main__s0"],
            diverges_at_step_id="main__s0",
            fork_config={"origin_step_id": "main__s1"},
        )
        task = {
            "episode_id": ep_id,
            "meta": {"scene": "FloorPlan1", "alfred_task_type": "pick_heat_then_place_in_recep"},
            "traj_path": "",
            "base_step_count": 0,
            "branch_config": fork_cfg,
            "_lane_type": "pick_heat_then_place_in_recep",
        }

        with patch("src.scheduler.EnvController", FakeEnv), \
             patch("src.scheduler.replay_steps", side_effect=ReplayUnavailable("bad lineage")):
            result = scheduler._run_fork(task, scheduler._create_agents())

        self.assertEqual("replay_unavailable", result.termination_reason)
        outcome = EpisodeManager.load(manager.file_path).data["final_outcome"]
        self.assertEqual("replay_unavailable", outcome["forks"][0]["termination_reason"])


class EpisodeManagerConcurrencyTest(unittest.TestCase):
    def _metadata(self):
        return {
            "task_goal": "test",
            "scene": "FloorPlan1",
            "task_type": "test",
        }

    def test_two_managers_preserve_concurrent_steps(self):
        with tempfile.TemporaryDirectory() as output_dir:
            first = EpisodeManager("ep", output_dir, self._metadata())
            second = EpisodeManager.load(first.file_path)
            barrier = threading.Barrier(2)

            def add(manager, branch_id):
                barrier.wait()
                manager.add_step({"step_id": "s0", "branch_id": branch_id})

            threads = [
                threading.Thread(target=add, args=(first, "main")),
                threading.Thread(target=add, args=(second, "fork1")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

            with open(first.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual({step["branch_id"] for step in data["steps"]}, {"main", "fork1"})
            self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_status_is_persisted_with_pid(self):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = EpisodeManager("ep", output_dir, self._metadata())
            manager.set_status("running", 123)
            loaded = EpisodeManager.load(manager.file_path)
            self.assertEqual(loaded.data["status"], "running")
            self.assertEqual(loaded.data["pid"], 123)

    def test_cancels_all_pending_descendants_of_an_invalid_parent(self):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = EpisodeManager("ep", output_dir, self._metadata())
            manager.add_pending_fork({
                "episode_id": "ep", "branch_id": "fork_a", "parent_branch_id": "main",
            })
            manager.add_pending_fork({
                "episode_id": "ep", "branch_id": "fork_b", "parent_branch_id": "fork_a",
            })

            manager.cancel_pending_descendants("main", "parent replay unavailable")

            loaded = EpisodeManager.load(manager.file_path)
            self.assertEqual([], loaded.get_pending_fork_tasks())
            self.assertTrue(all(
                entry["state"] == "cancelled"
                for entry in loaded.data["pending_forks"]
            ))


if __name__ == "__main__":
    unittest.main()
