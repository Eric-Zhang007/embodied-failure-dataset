import tempfile
import unittest
from collections import deque
from unittest.mock import Mock, patch

from src.branch_runner import BranchConfig
from src.episode_manager import EpisodeManager
from src.scheduler import Scheduler, SchedulerConfig
from src.step_recorder import StepRecorder


def _metadata():
    return {
        "task_goal": "Put an Apple in a Bowl.",
        "scene": "FloorPlan1",
        "task_type": "pick_and_place_simple",
        "alfred_scene": {"floor_plan": "FloorPlan1"},
    }


def _fork_task():
    return {
        "episode_id": "episode-1",
        "branch_id": "fork_s3_main",
        "parent_branch_id": "main",
        "shared_context_step_ids": ["main__s0", "main__s1"],
        "diverges_at_step_id": "main__s1",
        "fork_config": {
            "replaces_step_id": "main__s2",
            "origin_step_id": "main__s3",
            "alternative_action": {"action": "MoveBack", "params": {}},
        },
    }


class PendingForkPersistenceTest(unittest.TestCase):
    def test_pending_fork_survives_reload_until_it_reaches_a_terminal_outcome(self):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = EpisodeManager("episode-1", output_dir, _metadata())
            manager.add_pending_fork(_fork_task())

            reloaded = EpisodeManager.load(manager.file_path)
            self.assertEqual([_fork_task()], reloaded.get_pending_fork_tasks())

            reloaded.mark_pending_fork_running("fork_s3_main")
            interrupted = EpisodeManager.load(manager.file_path)
            self.assertEqual([_fork_task()], interrupted.get_pending_fork_tasks())

            interrupted.update_final_outcome(
                {
                    "branch_id": "fork_s3_main",
                    "termination_reason": "fork_failed",
                    "total_steps": 1,
                },
                is_main=False,
                fork_source_step_id="main__s3",
                counterfactual_verified=False,
            )

            completed = EpisodeManager.load(manager.file_path)
            self.assertEqual([], completed.get_pending_fork_tasks())
            self.assertEqual(
                "fork_failed",
                completed.data["final_outcome"]["forks"][0]["termination_reason"],
            )

    def test_scheduler_reloads_pending_fork_after_main_has_finished(self):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = EpisodeManager("episode-1", output_dir, _metadata())
            manager.add_pending_fork(_fork_task())
            manager.update_final_outcome(
                {
                    "branch_id": "main",
                    "termination_reason": "task_complete",
                    "total_steps": 4,
                },
                is_main=True,
            )
            manager.set_status("completed")

            scheduler = Scheduler.__new__(Scheduler)
            scheduler.config = SchedulerConfig(
                data_dir="/fake-data",
                output_dir=output_dir,
                api_key="test-key",
            )
            scheduler.queue = deque()

            with patch("src.scheduler.glob.glob", side_effect=[["/fake/episode-1/traj_data.json"], [], []]), \
                 patch("src.scheduler.load_traj", return_value={}), \
                 patch("src.scheduler.extract_metadata", return_value=_metadata()), \
                 patch("src.scheduler.extract_low_actions", return_value=[]):
                scheduler.load_tasks()

            self.assertEqual(1, len(scheduler.queue))
            config: BranchConfig = scheduler.queue[0]["branch_config"]
            self.assertEqual("fork_s3_main", config.branch_id)

    def test_scheduler_does_not_claim_pending_fork_from_live_episode(self):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = EpisodeManager("episode-1", output_dir, _metadata())
            manager.add_pending_fork(_fork_task())
            manager.set_status("running", pid=12345)

            scheduler = Scheduler.__new__(Scheduler)
            scheduler.config = SchedulerConfig(
                data_dir="/fake-data",
                output_dir=output_dir,
                api_key="test-key",
            )
            scheduler.queue = deque()

            with patch("src.scheduler.glob.glob", side_effect=[["/fake/episode-1/traj_data.json"], [], []]), \
                 patch("src.scheduler.load_traj", return_value={}), \
                 patch("src.scheduler.extract_metadata", return_value=_metadata()), \
                 patch("src.scheduler.extract_low_actions", return_value=[]), \
                 patch.object(scheduler, "_pid_is_alive", return_value=True):
                scheduler.load_tasks()

            self.assertEqual([], list(scheduler.queue))

    def test_failed_fork_root_is_recorded_as_a_terminal_outcome(self):
        class FailingEnv:
            def __init__(self, scene):
                self.scene = scene

            def reset_to_alfred_scene(self, _scene):
                return None

            def get_state_snapshot(self):
                return {"metadata": {"objects": []}}

            def step(self, _action, **_params):
                return {
                    "success": False,
                    "error": "blocked by environment",
                    "frame": None,
                    "metadata": {"objects": []},
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as output_dir:
            manager = EpisodeManager("episode-1", output_dir, _metadata())
            fork_task = _fork_task()
            fork_task["shared_context_step_ids"] = []
            fork_task["fork_config"]["alternative_action"] = {"action": "MoveAhead", "params": {}}
            manager.add_pending_fork(fork_task)

            scheduler = Scheduler.__new__(Scheduler)
            scheduler.config = SchedulerConfig(output_dir=output_dir, api_key="test-key")
            scheduler.recorder = StepRecorder()
            task = {
                "episode_id": "episode-1",
                "meta": _metadata(),
                "branch_config": BranchConfig(**fork_task),
            }

            with patch("src.scheduler.EnvController", FailingEnv):
                result = scheduler._run_fork(task, (Mock(), Mock(), Mock()))

            self.assertEqual("fork_failed", result.termination_reason)
            saved = EpisodeManager.load(manager.file_path)
            self.assertEqual(
                "fork_failed",
                saved.data["final_outcome"]["forks"][0]["termination_reason"],
            )
            self.assertFalse(
                saved.data["final_outcome"]["forks"][0]["counterfactual_root_feasible"]
            )

    def test_fork_outcome_persists_a_feasible_root_separately_from_completion(self):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = EpisodeManager("episode-1", output_dir, _metadata())
            manager.update_final_outcome(
                {
                    "branch_id": "fork_s3_main",
                    "termination_reason": "unrecoverable",
                    "total_steps": 3,
                },
                is_main=False,
                fork_source_step_id="main__s3",
                counterfactual_verified=False,
                counterfactual_root_feasible=True,
            )

            saved = EpisodeManager.load(manager.file_path)
            fork = saved.data["final_outcome"]["forks"][0]
            self.assertTrue(fork["counterfactual_root_feasible"])
            self.assertFalse(fork["counterfactual_verified"])
            self.assertEqual([], saved.get_pending_fork_tasks())


if __name__ == "__main__":
    unittest.main()
