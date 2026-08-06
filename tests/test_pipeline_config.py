import tempfile
import unittest
from pathlib import Path

from scripts.run_pipeline import _load_api_config


class PipelineConfigTest(unittest.TestCase):
    def test_config_reads_reasoning_effort_and_allows_key_from_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "collector.toml"
            config_path.write_text(
                """[api]
base_url = "https://example.invalid/v1"

[models]
planner = "planner-model"
executor = "executor-model"
oracle = "oracle-model"

[reasoning_effort]
planner = "high"
executor = "low"
oracle = "xhigh"
""",
                encoding="utf-8",
            )

            config = _load_api_config(str(config_path))

        self.assertEqual("", config["api_key"])
        self.assertEqual("high", config["planner_reasoning_effort"])
        self.assertEqual("low", config["executor_reasoning_effort"])
        self.assertEqual("xhigh", config["oracle_reasoning_effort"])


if __name__ == "__main__":
    unittest.main()
