import unittest

import numpy as np

from src.eb_agent import EBAgent


class CapturingClient:
    def __init__(self):
        self.user_text = None

    def chat_with_image_json(self, *, user_text, **kwargs):
        self.user_text = user_text
        return {
            "diagnosis": "I failed.",
            "recovery_reasoning": "I will recover.",
            "counterfactual": None,
            "proposed_recovery_action": {"action": "MoveAhead", "params": {}},
        }

    def chat_with_images_json(self, *, user_text, **kwargs):
        self.user_text = user_text
        return {"action": "MoveAhead", "params": {}, "reasoning": "I move."}


def _visible_alarm_clock():
    return [
        {
            "objectId": "AlarmClock|1",
            "objectType": "AlarmClock",
            "visible": True,
            "pickupable": True,
            "isPickedUp": False,
            "position": {"x": 0, "y": 0, "z": 1},
        }
    ]


class EBAgentPromptTest(unittest.TestCase):
    def test_lookaround_prompt_includes_shared_task_context(self):
        client = CapturingClient()
        agent = EBAgent(client)

        agent.propose_action_lookaround(
            task_goal="move the alarm clock from one desk to another one",
            look_images=[("ahead", np.zeros((1, 1, 3))), ("left", np.zeros((1, 1, 3)))],
            visible_objects=_visible_alarm_clock(),
            action_history=[],
            last_error=None,
            inventory_objects=[{"objectId": "AlarmClock|1", "objectType": "AlarmClock"}],
            task_criteria="  - AlarmClock must be inside Desk",
        )

        self.assertIn("HAND STATUS: holding AlarmClock (AlarmClock|1)", client.user_text)
        self.assertIn("TASK COMPLETION CRITERIA", client.user_text)
        self.assertIn("AlarmClock must be inside Desk", client.user_text)

    def test_phase3_prompt_includes_shared_task_context(self):
        client = CapturingClient()
        agent = EBAgent(client)

        agent.diagnose_failure(
            task_goal="move the alarm clock from one desk to another one",
            error_message="InvalidOperationException",
            image=np.zeros((1, 1, 3)),
            action_history=[],
            cascade_level=1,
            visible_objects=_visible_alarm_clock(),
            inventory_objects=[{"objectId": "AlarmClock|1", "objectType": "AlarmClock"}],
            task_criteria="  - AlarmClock must be inside Desk",
        )

        self.assertIn("HAND STATUS: holding AlarmClock (AlarmClock|1)", client.user_text)
        self.assertIn("TASK COMPLETION CRITERIA", client.user_text)
        self.assertIn("AlarmClock must be inside Desk", client.user_text)


if __name__ == "__main__":
    unittest.main()
