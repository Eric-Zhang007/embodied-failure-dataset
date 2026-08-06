import os
import unittest
from unittest.mock import patch

from src.env_controller import EnvController


class EnvControllerTest(unittest.TestCase):
    @patch.object(EnvController, "_start_xvfb")
    @patch("src.env_controller.Controller")
    def test_enables_instance_segmentation_for_visible_bounds(self, controller, _start_xvfb):
        env = EnvController("FloorPlan1")

        self.assertTrue(controller.call_args.kwargs["renderInstanceSegmentation"])
        env.close()

    @patch.object(EnvController, "_start_xvfb")
    @patch("src.env_controller.Controller")
    def test_uses_the_headless_xvfb_display_for_unity(self, controller, _start_xvfb):
        with patch.object(EnvController, "_xvfb_display", ":99"):
            env = EnvController("FloorPlan1")

        self.assertEqual(":99", controller.call_args.kwargs["x_display"])
        env.close()

    @patch("src.env_controller.time.sleep")
    @patch("src.env_controller.subprocess.Popen")
    def test_prefers_xvfb_over_an_available_wsldisplay(self, popen, _sleep):
        popen.return_value.poll.return_value = None
        with patch.object(EnvController, "_xvfb_proc", None), \
             patch.dict(os.environ, {"DISPLAY": ":0"}, clear=False):
            EnvController._start_xvfb()

        popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
