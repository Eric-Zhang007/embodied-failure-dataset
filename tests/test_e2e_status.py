import tempfile
import unittest
from pathlib import Path

from scripts import e2e_test
from src.episode_manager import EpisodeManager


class E2EMainStatusTest(unittest.TestCase):
    def test_loads_api_configuration_without_passing_a_key_on_the_command_line(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config_path = Path(output_dir) / "config.toml"
            config_path.write_text(
                '[api]\nbase_url = "https://example.test/v1"\napi_key = "test-key"\n'
                '[models]\nplanner = "planner-model"\n',
                encoding="utf-8",
            )

            config = e2e_test._load_api_config(str(config_path))

            self.assertEqual("test-key", config["api_key"])
            self.assertEqual("https://example.test/v1", config["api_base_url"])
            self.assertEqual("planner-model", config["planner_model"])

    def test_persists_completed_status_after_a_main_branch_result(self):
        with tempfile.TemporaryDirectory() as output_dir:
            episode = EpisodeManager(
                "ep", output_dir,
                {"task_goal": "test", "scene": "FloorPlan1", "task_type": "test"},
            )

            e2e_test._persist_main_result_status(episode.file_path, "task_complete")

            self.assertEqual("completed", EpisodeManager.load(episode.file_path).data["status"])


if __name__ == "__main__":
    unittest.main()
