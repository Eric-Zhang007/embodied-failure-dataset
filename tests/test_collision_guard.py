import unittest

from src.collision_guard import CollisionGuard
from src.branch_runner import BranchRunner


def _metadata(x=0.0, z=0.0, yaw=0.0):
    return {
        "agent": {
            "position": {"x": x, "z": z},
            "rotation": {"y": yaw},
        },
    }


class CollisionGuardTest(unittest.TestCase):
    def test_blocks_the_same_navigation_action_at_the_confirmed_blocked_pose(self):
        guard = CollisionGuard()
        metadata = _metadata(x=1.0, z=2.0, yaw=90.0)

        recorded = guard.record_navigation_failure(
            "MoveBack",
            "KitchenIsland_1 is blocking Agent 0 from moving by (0, 0, 0.125).",
            metadata,
        )

        self.assertTrue(recorded)
        rejection = guard.blocked_action_reason("MoveBack", metadata)
        self.assertIn("MoveBack", rejection)
        self.assertIn("KitchenIsland_1", rejection)

    def test_allows_the_same_action_after_a_turn_changes_the_physical_direction(self):
        guard = CollisionGuard()
        guard.record_navigation_failure(
            "MoveBack",
            "KitchenIsland_1 is blocking Agent 0 from moving.",
            _metadata(yaw=0.0),
        )

        self.assertIsNone(guard.blocked_action_reason("MoveBack", _metadata(yaw=90.0)))

    def test_blocks_a_different_relative_action_that_repeats_the_world_direction(self):
        guard = CollisionGuard()
        guard.record_navigation_failure(
            "MoveBack",
            "KitchenIsland_1 is blocking Agent 0 from moving.",
            _metadata(yaw=0.0),
        )

        rejection = guard.blocked_action_reason("MoveRight", _metadata(yaw=90.0))

        self.assertIn("KitchenIsland_1", rejection)

    def test_blocks_a_sequence_when_its_first_effective_move_repeats_a_collision(self):
        guard = CollisionGuard()
        metadata = _metadata()
        guard.record_navigation_failure(
            "MoveAhead",
            "Cabinet_1 is blocking Agent 0 from moving.",
            metadata,
        )

        rejection = guard.blocked_sequence_reason([
            {"action": "LookDown"},
            {"action": "MoveAhead", "repeat": 4},
        ], metadata)

        self.assertIn("MoveAhead", rejection)
        self.assertIsNone(guard.blocked_sequence_reason([
            {"action": "RotateRight"},
            {"action": "MoveAhead", "repeat": 4},
        ], metadata))

    def test_ignores_non_collision_failures(self):
        guard = CollisionGuard()
        metadata = _metadata()

        self.assertFalse(guard.record_navigation_failure(
            "MoveAhead", "Target object is not visible.", metadata,
        ))
        self.assertIsNone(guard.blocked_action_reason("MoveAhead", metadata))

    def test_orients_toward_the_only_unblocked_direction(self):
        guard = CollisionGuard()
        metadata = _metadata(x=1.0, z=2.0, yaw=0.0)
        for action in ("MoveAhead", "MoveLeft", "MoveBack"):
            guard.record_navigation_failure(
                action, "Table_1 is blocking Agent 0 from moving.", metadata,
            )

        self.assertEqual(("RotateRight",), guard.reorientation_escape(metadata))
        self.assertIsNone(guard.reorientation_escape(metadata))

    def test_turns_left_or_around_when_those_are_the_only_open_routes(self):
        metadata = _metadata()
        cases = (
            (("MoveAhead", "MoveRight", "MoveBack"), ("RotateLeft",)),
            (("MoveAhead", "MoveLeft", "MoveRight"), ("RotateRight", "RotateRight")),
            (("MoveLeft", "MoveRight", "MoveBack"), ()),
        )
        for blocked_actions, expected_turns in cases:
            with self.subTest(blocked_actions=blocked_actions):
                guard = CollisionGuard()
                for action in blocked_actions:
                    guard.record_navigation_failure(
                        action, "Table_1 is blocking Agent 0 from moving.", metadata,
                    )
                self.assertEqual(expected_turns, guard.reorientation_escape(metadata))

    def test_escape_grants_one_extra_replan_when_the_first_choice_is_still_blocked(self):
        guard = CollisionGuard()
        metadata = _metadata()
        for action in ("MoveAhead", "MoveLeft", "MoveBack"):
            guard.record_navigation_failure(
                action, "Table_1 is blocking Agent 0 from moving.", metadata,
            )

        guard.reorientation_escape(metadata)

        self.assertTrue(guard.consume_escape_replan_credit())
        self.assertFalse(guard.consume_escape_replan_credit())

    def test_branch_runner_executes_the_safe_reorientation_escape(self):
        class TurningEnv:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return {"success": True, "frame": None, "metadata": _metadata(yaw=90.0)}

        guard = CollisionGuard()
        metadata = _metadata()
        for action in ("MoveAhead", "MoveLeft", "MoveBack"):
            guard.record_navigation_failure(
                action, "Table_1 is blocking Agent 0 from moving.", metadata,
            )

        result = BranchRunner.__new__(BranchRunner)._execute_stuck_reorientation(
            guard, TurningEnv(), metadata,
        )

        self.assertEqual("RotateRight", result[0][0])
        self.assertTrue(result[0][1]["success"])

    def test_branch_runner_checks_the_guard_before_a_move_sequence(self):
        guard = CollisionGuard()
        metadata = _metadata()
        guard.record_navigation_failure(
            "MoveLeft", "Cabinet_1 is blocking Agent 0 from moving.", metadata,
        )

        rejection = BranchRunner._collision_guard_error(
            guard,
            "MoveSequence",
            {"steps": [{"action": "MoveLeft", "repeat": 2}]},
            metadata,
        )

        self.assertIn("Cabinet_1", rejection)

    def test_branch_runner_checks_the_guard_before_a_recovery_move(self):
        guard = CollisionGuard()
        metadata = _metadata()
        guard.record_navigation_failure(
            "MoveBack", "Island_1 is blocking Agent 0 from moving.", metadata,
        )

        rejection = BranchRunner._collision_guard_error(
            guard, "MoveBack", {}, metadata,
        )

        self.assertIn("Island_1", rejection)

    def test_recovery_allows_a_valid_move_sequence(self):
        error = BranchRunner._validate_recovery_action(
            "MoveSequence",
            {"steps": [{"action": "MoveBack", "repeat": 8}, {"action": "RotateLeft"}]},
            camera_horizon=0.0,
        )

        self.assertIsNone(error)

    def test_recovery_allows_an_environment_confirmed_object_id(self):
        error = BranchRunner._validate_recovery_action(
            "OpenObject",
            {"objectId": "Cabinet|-00.62|+01.53|-02.10"},
            camera_horizon=0.0,
        )

        self.assertIsNone(error)
        self.assertIn(
            "objectType",
            BranchRunner._validate_standalone_action(
                "OpenObject",
                {"objectId": "Cabinet|-00.62|+01.53|-02.10"},
                camera_horizon=0.0,
            ),
        )
