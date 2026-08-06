import unittest

import numpy as np

from src.branch_runner import BranchRunner


class _RotateEnv:
    def __init__(self):
        self.calls = []

    def step(self, action, **params):
        self.calls.append((action, params))
        return {
            "success": True,
            "error": None,
            "frame": np.zeros((2, 2, 3), dtype=np.uint8),
            "metadata": {"objects": [], "agent": {}},
        }


class RecoveryLookAroundTest(unittest.TestCase):
    def test_recovery_lookaround_completes_a_full_rotation(self):
        env = _RotateEnv()

        result = BranchRunner._execute_recovery_lookaround(env)

        self.assertTrue(result["success"])
        self.assertEqual([("RotateLeft", {})] * 4, env.calls)
