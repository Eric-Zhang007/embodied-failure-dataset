import os
import subprocess
import time
import atexit

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
    def __init__(self, scene: str, width: int = 300, height: int = 300):
        self._start_xvfb()
        self.width = width
        self.height = height
        self.scene = scene
        self.alfred_task_state = empty_task_state()
        self.controller = Controller(
            scene=scene,
            width=width,
            height=height,
            makeAgentsVisible=False,
        )

    def step(self, action: str, **params) -> dict:
        event = self.controller.step(action=action, **params)
        success = event.metadata["lastActionSuccess"]
        update_alfred_task_state(self.alfred_task_state, action, params, event.metadata)
        clean_sink_contents_after_faucet(self.controller, action, params, event.metadata)
        return {
            "success": success,
            "error": event.metadata.get("errorMessage") if not success else None,
            "frame": event.frame,
            "metadata": event.metadata,
            "task_state": serializable_task_state(self.alfred_task_state),
        }

    def get_state_snapshot(self) -> dict:
        """返回当前帧和环境状态的完整快照。"""
        event = self.controller.step(action="Pass")
        return {
            "frame": event.frame,
            "metadata": event.metadata,
            "task_state": serializable_task_state(self.alfred_task_state),
        }

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
        self.controller.stop()

    # ------------------------------------------------------------------
    # Xvfb management
    # ------------------------------------------------------------------
    _xvfb_proc = None
    _xvfb_display = ":99"

    @classmethod
    def _start_xvfb(cls):
        if cls._xvfb_proc is not None:
            return
        # Check if DISPLAY already set
        if os.environ.get("DISPLAY"):
            return
        # Check if Xvfb already running on :99
        try:
            subprocess.check_call(
                ["xdpyinfo", "-display", cls._xvfb_display],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.environ["DISPLAY"] = cls._xvfb_display
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

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
        if cls._xvfb_proc is not None:
            cls._xvfb_proc.terminate()
            cls._xvfb_proc.wait()
            cls._xvfb_proc = None
