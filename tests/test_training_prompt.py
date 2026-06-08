import unittest

from scripts.finetune_qwen3vl import build_input
from scripts.generate_ft_data import build_prompt


class TrainingPromptTest(unittest.TestCase):
    def test_training_input_reuses_eb_history_context(self):
        history = [
            {
                "step_id": "s0",
                "branch_id": "main",
                "step_index_in_branch": 0,
                "action": "MoveAhead",
                "action_params": {"moveMagnitude": 0.25, "objectId": "Apple|0|0|0"},
                "success": True,
                "error_message": None,
                "eb_reasoning": "moved closer",
                "oracle_ground_truth": "hidden supervisor field",
            }
        ]

        prompt = build_input("pick up the apple", history)

        self.assertIn("Full EB history:", prompt)
        self.assertIn("moved closer", prompt)
        self.assertNotIn("hidden supervisor field", prompt)
        self.assertNotIn("Apple|0|0|0", prompt)

    def test_reasoning_generation_prompt_does_not_request_pipe_format(self):
        history = [
            {
                "step_id": "s0",
                "branch_id": "main",
                "step_index_in_branch": 0,
                "action": "RotateLeft",
                "action_params": {},
                "success": True,
                "error_message": None,
                "eb_reasoning": "looked left",
            }
        ]

        prompt = build_prompt("find the mug", "MoveAhead", {}, history)

        self.assertIn("Full EB history:", prompt)
        self.assertNotIn("scene/goal/plan", prompt)
        self.assertNotIn(" | ", prompt)


if __name__ == "__main__":
    unittest.main()
