import json
import unittest

import numpy as np

from src.vlm_client import VLMClient


def _valid_200(content_text="Hello World"):
    return json.dumps({"choices": [{"message": {"content": content_text}}]})


def _200_with_reasoning(content_text="Hello", reasoning_text="I think..."):
    return json.dumps({
        "choices": [{"message": {
            "content": content_text,
            "reasoning_content": reasoning_text,
        }}]
    })


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


class ExtractContentTest(unittest.TestCase):
    def test_valid_response_returns_content(self):
        text = _valid_200("hello there")
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("hello there", content)
        self.assertIsNone(reasoning)
        self.assertIsNone(error)

    def test_valid_with_reasoning_content(self):
        text = _200_with_reasoning("result", "thinking...")
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("result", content)
        self.assertEqual("thinking...", reasoning)
        self.assertIsNone(error)

    def test_html_response_not_json(self):
        text = "<html><body>Internal Server Error</body></html>"
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)
        self.assertIn("not valid JSON", error)

    def test_missing_choices_key(self):
        text = '{"message": "ok"}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)
        self.assertIn("non-empty 'choices'", error)

    def test_empty_choices_list(self):
        text = '{"choices": []}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)
        self.assertIn("non-empty 'choices'", error)

    def test_choices_not_a_list(self):
        text = '{"choices": "string_not_list"}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)
        self.assertIn("non-empty 'choices'", error)

    def test_message_not_a_dict(self):
        text = '{"choices": [{"message": "not_dict"}]}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)
        self.assertIn("not a dict", error)

    def test_content_missing_from_message(self):
        text = '{"choices": [{"message": {}}]}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)
        self.assertIn("None", error)

    def test_content_is_none(self):
        text = '{"choices": [{"message": {"content": null}}]}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)
        self.assertIn("None", error)

    def test_content_is_empty_string(self):
        text = '{"choices": [{"message": {"content": ""}}]}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)

    def test_content_is_whitespace_only(self):
        text = '{"choices": [{"message": {"content": "   "}}]}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)

    def test_content_is_not_a_string(self):
        text = '{"choices": [{"message": {"content": 42}}]}'
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertIsNone(reasoning)
        self.assertIsNotNone(error)
        self.assertIn("not a string", error)

    def test_no_content_but_reasoning_content_present(self):
        text = json.dumps({
            "choices": [{"message": {
                "reasoning_content": "the model thought about it",
            }}]
        })
        content, reasoning, error = VLMClient._extract_content_from_200(text)
        self.assertEqual("", content)
        self.assertEqual("the model thought about it", reasoning)
        self.assertIsNotNone(error)
        self.assertIn("reasoning_content present", error)


if __name__ == "__main__":
    unittest.main()
