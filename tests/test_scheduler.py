"""Deterministic tests for Scheduler accounting and worker isolation."""
import unittest
from unittest.mock import patch, MagicMock
from collections import deque

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

        self.assertEqual(len(scheduler.fork_queue), 0)  # consumed by run()
        self.assertEqual(scheduler.stats["completed"], 2)  # main + fork


if __name__ == "__main__":
    unittest.main()
