import json
import unittest

from src.context_builder import (
    build_branch_history,
    build_eb_history_context,
    build_oracle_history_context,
)


def make_step(idx, branch="main", success=True, **extra):
    step = {
        "step_id": f"s{idx}",
        "branch_id": branch,
        "parent_step_id": f"s{idx - 1}" if idx > 0 else None,
        "step_index_in_branch": idx,
        "action": "MoveAhead",
        "action_params": {"moveMagnitude": 0.25, "objectId": "Hidden|0|0|0"},
        "success": success,
        "error_message": None if success else "blocked",
        "eb_reasoning": f"reason {idx}",
        "eb_diagnosis": "diag" if not success else None,
        "eb_recovery_reasoning": "recover" if not success else None,
        "eb_counterfactual": None,
        "eb_proposed_recovery_action": None,
        "oracle_injection_decision": {"decided_to_inject": True, "method": "occlude"} if idx == 3 else None,
        "oracle_diagnosis_correct": False if not success else None,
        "oracle_ground_truth": "real cause" if not success else None,
        "oracle_counterfactual_grade": "PA" if not success else None,
        "oracle_counterfactual_gold": "better action" if not success else None,
        "oracle_recovery_verdict": "recoverable" if not success else None,
        "fork_metadata": None,
    }
    step.update(extra)
    return step


def json_lines(context):
    return [json.loads(line) for line in context.splitlines() if line.startswith("{")]


class ContextBuilderTest(unittest.TestCase):
    def test_eb_history_uses_all_steps_without_oracle_fields(self):
        steps = [make_step(i) for i in range(12)]

        context = build_eb_history_context(steps)
        rows = json_lines(context)

        self.assertEqual(12, len(rows))
        self.assertEqual(0, rows[0]["step"])
        self.assertEqual(11, rows[-1]["step"])
        self.assertIn("reason 0", context)
        self.assertIn("reason 11", context)
        self.assertNotIn("oracle", context)
        self.assertNotIn("Hidden|0|0|0", context)

    def test_oracle_history_uses_all_steps_and_includes_oracle_fields(self):
        steps = [make_step(i, success=(i != 5)) for i in range(12)]

        context = build_oracle_history_context(steps)
        rows = json_lines(context)

        self.assertEqual(12, len(rows))
        self.assertEqual(0, rows[0]["step"])
        self.assertEqual(11, rows[-1]["step"])
        self.assertIn("oracle", context)
        self.assertIn("real cause", context)
        self.assertIn("recoverable", context)

    def test_oracle_history_keeps_fork_field_names(self):
        steps = [
            make_step(
                0,
                branch="fork_s1_main",
                fork_metadata={"is_fork_root": True},
                fork_decision={"should_fork": True},
            )
        ]

        row = json_lines(build_oracle_history_context(steps))[0]

        self.assertIn("fork_metadata", row["oracle"])
        self.assertIn("fork_decision", row["oracle"])

    def test_branch_history_includes_parent_shared_steps_before_fork_steps(self):
        episode_steps = [
            make_step(0, branch="main", step_id="main__s0"),
            make_step(1, branch="main", step_id="main__s1"),
            make_step(2, branch="main", step_id="main__s2"),
            make_step(0, branch="fork_s1_main", step_id="fork_s1_main__s0", action="RotateLeft"),
            make_step(1, branch="fork_s1_main", step_id="fork_s1_main__s1", action="MoveAhead"),
        ]

        history = build_branch_history(
            episode_steps,
            branch_id="fork_s1_main",
            parent_branch_id="main",
            shared_step_ids=["main__s0", "main__s1"],
        )

        self.assertEqual(
            [
                ("main", "main__s0"),
                ("main", "main__s1"),
                ("fork_s1_main", "fork_s1_main__s0"),
                ("fork_s1_main", "fork_s1_main__s1"),
            ],
            [(s["branch_id"], s["step_id"]) for s in history],
        )

    def test_nested_branch_history_keeps_the_complete_ordered_lineage(self):
        episode_steps = [
            make_step(0, branch="main", step_id="main__s0"),
            make_step(1, branch="main", step_id="main__s1"),
            make_step(0, branch="fork_s4_main", step_id="fork_s4_main__s0"),
            make_step(1, branch="fork_s4_main", step_id="fork_s4_main__s1"),
            make_step(
                0,
                branch="fork_s2_fork_s4_main",
                step_id="fork_s2_fork_s4_main__s0",
            ),
        ]

        history = build_branch_history(
            episode_steps,
            branch_id="fork_s2_fork_s4_main",
            parent_branch_id="fork_s4_main",
            shared_step_ids=["main__s0", "main__s1", "fork_s4_main__s0"],
        )

        self.assertEqual(
            [
                "main__s0",
                "main__s1",
                "fork_s4_main__s0",
                "fork_s2_fork_s4_main__s0",
            ],
            [step["step_id"] for step in history],
        )

    def test_current_failed_step_is_rendered_for_diagnosis(self):
        history = [make_step(0)]
        pending = make_step(
            1,
            success=False,
            action="PickupObject",
            action_params={"objectType": "Apple"},
            eb_reasoning="try pickup",
        )

        context = build_eb_history_context(history, current_step=pending)
        rows = json_lines(context)

        self.assertEqual(2, len(rows))
        self.assertEqual("PickupObject", rows[-1]["action"])
        self.assertFalse(rows[-1]["success"])
        self.assertEqual("blocked", rows[-1]["error"])


if __name__ == "__main__":
    unittest.main()
