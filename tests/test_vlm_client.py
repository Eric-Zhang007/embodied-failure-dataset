import unittest

import numpy as np

from src.vlm_client import VLMClient


class VLMClientTest(unittest.TestCase):
    def test_multi_image_request_labels_each_image_before_image_part(self):
        client = VLMClient("openai", "fake")
        captured = {}

        def fake_chat(messages, json_mode=False, temperature=0.2, max_tokens=2048):
            captured["messages"] = messages
            return "ok"

        client.chat = fake_chat

        client.chat_with_images(
            system_prompt="sys",
            user_text="scan prompt",
            images=[
                ("ahead", np.zeros((1, 1, 3), dtype=np.uint8)),
                ("left", np.zeros((1, 1, 3), dtype=np.uint8)),
            ],
        )

        content = captured["messages"][1]["content"]
        self.assertEqual("scan prompt", content[0]["text"])
        self.assertEqual("VIEW 1: ahead", content[1]["text"])
        self.assertEqual("image_url", content[2]["type"])
        self.assertEqual("VIEW 2: left", content[3]["text"])
        self.assertEqual("image_url", content[4]["type"])


if __name__ == "__main__":
    unittest.main()
