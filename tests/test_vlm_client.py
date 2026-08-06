import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from src.vlm_client import VLMClient, _openai_request_policy


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
    def test_root_base_url_uses_the_openai_v1_chat_endpoint(self):
        response = MagicMock(status_code=200, text=_valid_200("ok"))
        client = VLMClient(
            "openai",
            "test-model",
            base_url="https://api.example.invalid",
            api_key="test",
        )

        with patch("requests.post", return_value=response) as post:
            client.chat_text("system", "user")

        self.assertEqual(
            "https://api.example.invalid/v1/chat/completions",
            post.call_args.args[0],
        )

    def test_request_policy_accepts_collection_overrides(self):
        with patch.dict(os.environ, {
            "EFD_API_TIMEOUT_S": "20",
            "EFD_API_MAX_ATTEMPTS": "2",
        }, clear=False):
            self.assertEqual((20.0, 2), _openai_request_policy())

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

    def test_forbidden_response_waits_for_endpoint_recovery(self):
        forbidden = MagicMock(status_code=403, text='{"error":"insufficient balance"}')
        forbidden.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        recovered = MagicMock(status_code=200, text=_valid_200("recovered"))
        client = VLMClient("openai", "test-model", base_url="https://example.invalid/v1", api_key="test")

        with patch.dict(os.environ, {
            "EFD_API_MAX_ATTEMPTS": "1",
            "EFD_API_OUTAGE_BACKOFF_S": "15",
        }, clear=False), patch("requests.post", side_effect=[forbidden, recovered]) as post, patch(
            "src.vlm_client.time.sleep"
        ) as sleep, patch.object(VLMClient, "_dump_failure") as dump:
            content = client.chat_text("system", "user")

        self.assertEqual("recovered", content)
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(15.0)
        dump.assert_not_called()

    def test_disabled_outage_recovery_raises_after_short_retries(self):
        forbidden = MagicMock(status_code=403, text='{"error":"insufficient balance"}')
        forbidden.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        client = VLMClient(
            "openai",
            "test-model",
            base_url="https://example.invalid/v1",
            api_key="test",
            recover_api_outages=False,
        )

        with patch.dict(os.environ, {"EFD_API_MAX_ATTEMPTS": "1"}, clear=False), patch(
            "requests.post", return_value=forbidden
        ) as post, patch("src.vlm_client.time.sleep") as sleep:
            with self.assertRaises(requests.HTTPError):
                client.chat_text("system", "user")

        self.assertEqual(1, post.call_count)
        sleep.assert_not_called()

    def test_timeout_waits_for_endpoint_recovery(self):
        recovered = MagicMock(status_code=200, text=_valid_200("recovered"))
        client = VLMClient("openai", "test-model", base_url="https://example.invalid/v1", api_key="test")

        with patch.dict(os.environ, {
            "EFD_API_MAX_ATTEMPTS": "1",
            "EFD_API_OUTAGE_BACKOFF_S": "12",
        }, clear=False), patch("requests.post", side_effect=[requests.Timeout(), recovered]) as post, patch("src.vlm_client.time.sleep") as sleep:
            content = client.chat_text("system", "user")

        self.assertEqual("recovered", content)
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(12.0)

    def test_connection_error_waits_for_endpoint_recovery(self):
        recovered = MagicMock(status_code=200, text=_valid_200("recovered"))
        client = VLMClient("openai", "test-model", base_url="https://example.invalid/v1", api_key="test")

        with patch.dict(os.environ, {
            "EFD_API_MAX_ATTEMPTS": "1",
            "EFD_API_OUTAGE_BACKOFF_S": "11",
        }, clear=False), patch(
            "requests.post", side_effect=[requests.ConnectionError("network unavailable"), recovered]
        ) as post, patch("src.vlm_client.time.sleep") as sleep:
            content = client.chat_text("system", "user")

        self.assertEqual("recovered", content)
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(11.0)

    def test_rate_limit_waits_for_endpoint_recovery_after_short_retries(self):
        rate_limited = MagicMock(status_code=429, text='{"error":"rate limited"}')
        recovered = MagicMock(status_code=200, text=_valid_200("recovered"))
        client = VLMClient("openai", "test-model", base_url="https://example.invalid/v1", api_key="test")

        with patch.dict(os.environ, {
            "EFD_API_MAX_ATTEMPTS": "1",
            "EFD_API_OUTAGE_BACKOFF_S": "14",
        }, clear=False), patch("requests.post", side_effect=[rate_limited, recovered]) as post, patch(
            "src.vlm_client.time.sleep"
        ) as sleep:
            content = client.chat_text("system", "user")

        self.assertEqual("recovered", content)
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(14.0)

    def test_server_error_waits_for_endpoint_recovery_after_short_retries(self):
        unavailable = MagicMock(status_code=503, text='{"error":"service unavailable"}')
        unavailable.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
        recovered = MagicMock(status_code=200, text=_valid_200("recovered"))
        client = VLMClient("openai", "test-model", base_url="https://example.invalid/v1", api_key="test")

        with patch.dict(os.environ, {
            "EFD_API_MAX_ATTEMPTS": "1",
            "EFD_API_OUTAGE_BACKOFF_S": "13",
        }, clear=False), patch("requests.post", side_effect=[unavailable, recovered]) as post, patch(
            "src.vlm_client.time.sleep"
        ) as sleep:
            content = client.chat_text("system", "user")

        self.assertEqual("recovered", content)
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(13.0)


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
