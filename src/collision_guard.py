"""Deterministic guard against retrying a confirmed navigation collision."""

from __future__ import annotations


_MOVEMENT_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
_ORIENTATION_ACTIONS = {"RotateLeft", "RotateRight"}
_CAMERA_ACTIONS = {"LookUp", "LookDown"}


class CollisionGuard:
    """Remember blocked moves only at the physical pose where they failed.

    A collision says nothing about the same relative action after the agent has
    moved or turned.  The key therefore includes position and yaw, which lets
    the executor reject low-value retries without preventing real rerouting.
    """

    def __init__(self):
        self._blocked: dict[tuple[float, float, float, str], str] = {}

    @staticmethod
    def _pose_key(metadata: dict, action: str) -> tuple[float, float, float, str] | None:
        agent = (metadata or {}).get("agent") or {}
        position = agent.get("position") or {}
        rotation = agent.get("rotation") or {}
        try:
            return (
                round(float(position["x"]), 3),
                round(float(position["z"]), 3),
                round(float(rotation.get("y", 0.0)) % 360.0, 1),
                action,
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _blocker(error_message: str | None) -> str | None:
        error = error_message or ""
        if " is blocking " not in error:
            return None
        blocker = error.split(" is blocking ", 1)[0].rsplit(": ", 1)[-1].strip()
        return blocker or None

    def record_navigation_failure(
        self, action: str, error_message: str | None, metadata: dict,
    ) -> bool:
        """Record a movement collision at its post-failure (unchanged) pose."""
        if action not in _MOVEMENT_ACTIONS:
            return False
        blocker = self._blocker(error_message)
        pose_key = self._pose_key(metadata, action)
        if blocker is None or pose_key is None:
            return False
        self._blocked[pose_key] = blocker
        return True

    def blocked_action_reason(self, action: str, metadata: dict) -> str | None:
        """Explain why this exact navigation retry must be replanned."""
        pose_key = self._pose_key(metadata, action)
        blocker = self._blocked.get(pose_key) if pose_key else None
        if blocker is None:
            return None
        return (
            f"{action} was not executed because {blocker} already blocked that "
            "same movement at the current pose. Rotate or choose a different "
            "movement direction before trying again."
        )

    def blocked_sequence_reason(self, steps: list[dict] | None, metadata: dict) -> str | None:
        """Check only the first movement before a heading-changing action.

        Camera tilts do not change movement direction, whereas a rotation does;
        later moves after a rotation are intentionally left to the environment.
        """
        if not isinstance(steps, list):
            return None
        for step in steps:
            if not isinstance(step, dict):
                return None
            action = step.get("action")
            if action in _CAMERA_ACTIONS:
                continue
            if action in _MOVEMENT_ACTIONS:
                return self.blocked_action_reason(action, metadata)
            if action in _ORIENTATION_ACTIONS:
                return None
            return None
        return None
