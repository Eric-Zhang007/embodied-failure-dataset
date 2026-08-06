import unittest

from src.branch_runner import _turn_to_scan_facing


class _RecordingEnv:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return {"success": True}


class ScanFacingTest(unittest.TestCase):
    def test_left_and_right_scan_views_use_matching_left_turn_counts(self):
        expected_turns = {"ahead": 0, "left": 1, "behind": 2, "right": 3}

        for direction, expected in expected_turns.items():
            with self.subTest(direction=direction):
                env = _RecordingEnv()
                actions = _turn_to_scan_facing(env, direction, "test")

                self.assertEqual(["RotateLeft"] * expected, env.actions)
                self.assertEqual(expected, len(actions))


if __name__ == "__main__":
    unittest.main()
