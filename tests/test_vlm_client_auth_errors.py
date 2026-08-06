import os
import unittest
from unittest.mock import Mock, patch

import requests

from src.vlm_client import VLMClient


class VLMClientAuthErrorTest(unittest.TestCase):
    def test_invalid_credentials_fail_without_an_outage_backoff(self):
        response = Mock(status_code=401, text="invalid key")
        response.raise_for_status.side_effect = requests.HTTPError("401 invalid key")
        client = VLMClient.openai("test-model", "test-key", base_url="https://example.invalid/v1")

        with patch.dict(os.environ, {
            "EFD_API_MAX_ATTEMPTS": "1",
            "EFD_FAIL_FAST_AUTH_ERRORS": "1",
        }, clear=False), \
             patch("requests.post", return_value=response), \
             patch("src.vlm_client.time.sleep", side_effect=RuntimeError("must not sleep")) as sleep:
            with self.assertRaises(requests.HTTPError):
                client.chat_text("system", "user")

        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
