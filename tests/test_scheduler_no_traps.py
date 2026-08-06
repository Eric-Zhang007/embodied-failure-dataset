import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.branch_runner import BranchConfig, BranchResult
from src.episode_manager import EpisodeManager
from src.scheduler import Scheduler, SchedulerConfig


class SchedulerNoTrapsTest(unittest.TestCase):
    def test_resume_preserves_the_no_traps_collection_mode(self):
        with tempfile.TemporaryDirectory() as output_dir:
            metadata = {
                "task_goal": "Put an Apple in a Bowl.",
                "scene": "FloorPlan1",
                "task_type": "pick_and_place_simple",
            }
            manager = EpisodeManager("episode-1", output_dir, metadata)
            manager.add_step({
                "step_id": "main__s0",
                "branch_id": "main",
                "step_index_in_branch": 0,
            })
            manager.set_status("interrupted")

            scheduler = Scheduler.__new__(Scheduler)
            scheduler.config = SchedulerConfig(
                output_dir=output_dir,
                api_key="test-key",
                no_traps=True,
            )
            scheduler._create_agents = Mock(return_value=(Mock(), Mock(), Mock()))
            task = {
                "episode_id": "episode-1",
                "traj_path": "unused.json",
                "branch_config": BranchConfig("episode-1", "main", None),
            }
            result = BranchResult("main", "task_complete", 1, [], [])

            with patch("src.scheduler.BranchRunner.resume", return_value=result) as resume:
                scheduler._run_branch_worker(task, 1, 1)

            self.assertFalse(resume.call_args.kwargs["enable_phase2"])


if __name__ == "__main__":
    unittest.main()
