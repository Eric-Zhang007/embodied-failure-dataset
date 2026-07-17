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


def _task(ep_id="ep1"):
    return {
        "traj_path": f"/fake/{ep_id}/traj_data.json",
        "episode_id": ep_id,
        "meta": {"scene": "FloorPlan1"},
        "base_step_count": 5,
        "branch_config": BranchConfig(episode_id=ep_id, branch_id="main", parent_branch_id=None),
    }


def _success_result():
    return BranchResult(branch_id="main", termination_reason="task_complete",
                        total_steps=3, fork_tasks=[], fork_source_step_ids=[])


def _skip_result():
    return BranchResult(branch_id="main", termination_reason="skipped",
                        total_steps=0, fork_tasks=[], fork_source_step_ids=[])


def _crash_result():
    return BranchResult(branch_id="main", termination_reason="worker_crash:boom",
                        total_steps=0, fork_tasks=[], fork_source_step_ids=[])


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
            EpisodeManager(ep_id, config.output_dir, {
                "task_goal": "g", "scene": "FloorPlan1", "task_type": "heat",
                "alfred_scene": {"object_poses": [1]},
            })
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


if __name__ == "__main__":
    unittest.main()
