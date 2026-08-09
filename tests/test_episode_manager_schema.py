import json
import os
import tempfile
import unittest

from src.episode_manager import EpisodeManager


def _meta():
    return {
        "task_goal": "Pick up the Mug and put it on the Table.",
        "scene": "FloorPlan1",
        "task_type": "pick_and_place",
        "alfred_task_type": "pick_and_place_simple",
        "alfred_task_id": "trial_1",
        "pddl_params": {"object_target": "Mug", "parent_target": "Table"},
        "alfred_scene": {"floor_plan": "FloorPlan1"},
        "initial_traps": [],
    }


class EpisodeManagerSchemaTest(unittest.TestCase):
    def test_new_episode_has_schema_version_and_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = EpisodeManager("ep1", tmp, _meta())

            self.assertEqual(ep.data["schema_version"], "1.0")
            self.assertEqual(ep.data["run_manifest"], {})

            with open(os.path.join(tmp, "ep1.json"), encoding="utf-8") as f:
                persisted = json.load(f)
            self.assertEqual(persisted["schema_version"], "1.0")
            self.assertEqual(persisted["run_manifest"], {})

    def test_run_manifest_carried_from_metadata(self):
        meta = _meta()
        meta["run_manifest"] = {"backend": "ai2thor", "seed": 7}
        with tempfile.TemporaryDirectory() as tmp:
            ep = EpisodeManager("ep2", tmp, meta)

            self.assertEqual(ep.data["run_manifest"], {"backend": "ai2thor", "seed": 7})

    def test_update_stage_persists_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = EpisodeManager("ep3", tmp, _meta())

            ep.update_stage(1, [True, False])

            self.assertEqual(ep.data["stage_index"], 1)
            self.assertEqual(ep.data["stage_progress"], [True, False])
            with open(os.path.join(tmp, "ep3.json"), encoding="utf-8") as f:
                persisted = json.load(f)
            self.assertEqual(persisted["stage_index"], 1)
            self.assertEqual(persisted["stage_progress"], [True, False])

    def test_update_stage_persists_start_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = EpisodeManager("ep4", tmp, _meta())

            ep.update_stage(1, [True, False], start_step=17)

            self.assertEqual(ep.data["stage_start_step"], 17)
            with open(os.path.join(tmp, "ep4.json"), encoding="utf-8") as f:
                persisted = json.load(f)
            self.assertEqual(persisted["stage_start_step"], 17)


if __name__ == "__main__":
    unittest.main()
