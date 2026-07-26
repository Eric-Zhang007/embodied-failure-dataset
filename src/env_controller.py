import atexit
import os
import subprocess
import threading
import time

import numpy as np
from ai2thor.controller import Controller

from src.alfred_scene import (
    apply_init_action,
    clean_sink_contents_after_faucet,
    empty_task_state,
    restore_alfred_scene,
    serializable_task_state,
    update_alfred_task_state,
)


class EnvController:
    _xvfb_proc = None
    _xvfb_display = ":99"
    _xvfb_lock = threading.Lock()

    def __init__(self, scene: str, width: int = 300, height: int = 300):
        self._start_xvfb()
        self._closed = False
        self.width = width
        self.height = height
        self.scene = scene
        self.alfred_task_state = empty_task_state()
        self.controller = Controller(
            scene=scene,
            width=width,
            height=height,
            makeAgentsVisible=False,
            renderInstanceSegmentation=True,
            x_display=self._xvfb_display,
        )

    def step(self, action: str, **params) -> dict:
        event = self.controller.step(action=action, **params)
        success = event.metadata["lastActionSuccess"]
        update_alfred_task_state(self.alfred_task_state, action, params, event.metadata)
        clean_sink_contents_after_faucet(self.controller, action, params, event.metadata)
        self._fix_visible_bounds(event)
        return {
            "success": success,
            "error": event.metadata.get("errorMessage") if not success else None,
            "frame": event.frame,
            "metadata": event.metadata,
            "task_state": serializable_task_state(self.alfred_task_state),
        }

    def get_state_snapshot(self) -> dict:
        """Return the current frame and full environment state."""
        event = self.controller.step(action="Pass")
        self._fix_visible_bounds(event)
        return {
            "frame": event.frame,
            "metadata": event.metadata,
            "task_state": serializable_task_state(self.alfred_task_state),
        }

    @staticmethod
    def _fix_visible_bounds(event):
        """Populate visibleBounds2D for AI2-THOR 5.0.0 events."""
        detections = event.instance_detections2D
        if detections is None:
            return
        for obj in event.metadata["objects"]:
            obj["visibleBounds2D"] = (
                obj.get("visible", False)
                and obj["objectId"] in detections
            )

    def reset_scene(self, scene: str = None):
        if scene is not None:
            self.scene = scene
        self.controller.reset(scene=self.scene)
        self.alfred_task_state = empty_task_state()

    def reset_to_alfred_scene(self, scene_state: dict):
        restore_alfred_scene(self.controller, scene_state)
        self.alfred_task_state = empty_task_state()
        apply_init_action(self.controller, scene_state.get("init_action"))

    def close(self):
        if self._closed:
            return
        self._closed = True
        controller, self.controller = self.controller, None
        if controller is not None:
            controller.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def _start_xvfb(cls):
        with cls._xvfb_lock:
            if cls._xvfb_proc is not None and cls._xvfb_proc.poll() is None:
                os.environ["DISPLAY"] = cls._xvfb_display
                return
            cls._xvfb_proc = subprocess.Popen(
                ["Xvfb", cls._xvfb_display, "-screen", "0", "1024x768x24", "-ac"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.environ["DISPLAY"] = cls._xvfb_display
            time.sleep(0.3)
            atexit.register(cls._stop_xvfb)

    @classmethod
    def _stop_xvfb(cls):
        with cls._xvfb_lock:
            proc, cls._xvfb_proc = cls._xvfb_proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait()
