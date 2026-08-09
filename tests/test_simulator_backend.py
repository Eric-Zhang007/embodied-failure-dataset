import unittest

from src.simulator_backend import (
    ActionSpec,
    Ai2ThorBackend,
    ObsSpec,
    StateSnapshot,
    StepResult,
)


class FakeEnv:
    def __init__(self):
        self.steps = []
        self.reset_calls = []
        self.closed = False
        self.snapshot = {
            "frame": "frame0",
            "metadata": {"objects": []},
            "task_state": {"heated_objects": []},
        }

    def step(self, action, **params):
        self.steps.append((action, params))
        return {
            "success": True,
            "error": None,
            "frame": "frame1",
            "metadata": {"objects": []},
            "task_state": {"heated_objects": []},
        }

    def get_state_snapshot(self):
        return self.snapshot

    def reset_scene(self, scene=None):
        self.reset_calls.append(("scene", scene))

    def reset_to_alfred_scene(self, scene_state):
        self.reset_calls.append(("scene_state", scene_state))

    def close(self):
        self.closed = True


class Ai2ThorBackendTest(unittest.TestCase):
    def test_step_translates_to_step_result(self):
        env = FakeEnv()
        backend = Ai2ThorBackend(env)

        result = backend.step("MoveAhead", moveMagnitude=0.25)

        self.assertIsInstance(result, StepResult)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.frame, "frame1")
        self.assertEqual(result.metadata, {"objects": []})
        self.assertEqual(result.task_state, {"heated_objects": []})
        self.assertEqual(env.steps, [("MoveAhead", {"moveMagnitude": 0.25})])

    def test_state_snapshot_translates(self):
        env = FakeEnv()
        backend = Ai2ThorBackend(env)

        snapshot = backend.state_snapshot()

        self.assertIsInstance(snapshot, StateSnapshot)
        self.assertEqual(snapshot.frame, "frame0")
        self.assertEqual(snapshot.metadata, {"objects": []})
        self.assertEqual(snapshot.task_state, {"heated_objects": []})

    def test_reset_delegates_to_scene_and_scene_state(self):
        env = FakeEnv()
        backend = Ai2ThorBackend(env)

        backend.reset(scene="FloorPlan1")
        backend.reset(scene_state={"floor_plan": "FloorPlan2"})
        backend.reset()

        self.assertEqual(
            env.reset_calls,
            [("scene", "FloorPlan1"), ("scene_state", {"floor_plan": "FloorPlan2"}), ("scene", None)],
        )

    def test_close_delegates(self):
        env = FakeEnv()
        backend = Ai2ThorBackend(env)

        backend.close()

        self.assertTrue(env.closed)

    def test_spec_defaults(self):
        backend = Ai2ThorBackend(FakeEnv(), width=640, height=480)

        self.assertEqual(backend.backend_name, "ai2thor")
        self.assertEqual(backend.obs_spec, ObsSpec(width=640, height=480))
        self.assertEqual(backend.action_spec, ActionSpec())


if __name__ == "__main__":
    unittest.main()
