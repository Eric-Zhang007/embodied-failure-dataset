"""Deterministic guard against retrying a confirmed navigation collision."""

from __future__ import annotations


_MOVEMENT_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
_ORIENTATION_ACTIONS = {"RotateLeft", "RotateRight"}
_CAMERA_ACTIONS = {"LookUp", "LookDown"}
_MOVEMENT_YAW_OFFSETS = {
    "MoveAhead": 0.0,
    "MoveRight": 90.0,
    "MoveBack": 180.0,
    "MoveLeft": 270.0,
}
_TURNS_TO_FACE_ACTION = {
    "MoveAhead": (),
    "MoveRight": ("RotateRight",),
    "MoveBack": ("RotateRight", "RotateRight"),
    "MoveLeft": ("RotateLeft",),
}


class CollisionGuard:
    """Remember blocked moves only at the physical pose where they failed.

    A collision says nothing about a different physical direction after the
    agent moves or turns.  We normalize a relative action to its world heading
    so a turn cannot disguise a retry of the same blocked route.
    """

    def __init__(self):
        self._blocked: dict[tuple[float, float, float], str] = {}
        self._reorientation_attempted: set[tuple[float, float]] = set()
        self._escape_replan_credit = 0

    @staticmethod
    def _position_key(metadata: dict) -> tuple[float, float] | None:
        agent = (metadata or {}).get("agent") or {}
        position = agent.get("position") or {}
        try:
            return round(float(position["x"]), 3), round(float(position["z"]), 3)
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _yaw(metadata: dict) -> float | None:
        rotation = ((metadata or {}).get("agent") or {}).get("rotation") or {}
        try:
            return round(float(rotation.get("y", 0.0)) % 360.0, 1)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _movement_heading(yaw: float, action: str) -> float:
        return round((yaw + _MOVEMENT_YAW_OFFSETS[action]) % 360.0, 1)

    @classmethod
    def _movement_key(
        cls, metadata: dict, action: str,
    ) -> tuple[float, float, float] | None:
        position_key = cls._position_key(metadata)
        yaw = cls._yaw(metadata)
        if position_key is None or yaw is None or action not in _MOVEMENT_YAW_OFFSETS:
            return None
        return (*position_key, cls._movement_heading(yaw, action))

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
        movement_key = self._movement_key(metadata, action)
        if blocker is None or movement_key is None:
            return False
        self._blocked[movement_key] = blocker
        return True

    def blocked_action_reason(self, action: str, metadata: dict) -> str | None:
        """Explain why this exact navigation retry must be replanned."""
        movement_key = self._movement_key(metadata, action)
        blocker = self._blocked.get(movement_key) if movement_key else None
        if blocker is None:
            return None
        return (
            f"{action} was not executed because {blocker} already blocked that "
            "physical direction at the current position. Choose a different "
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

    def reorientation_escape(self, metadata: dict) -> tuple[str, ...] | None:
        """Face the only physical direction not yet blocked at this position."""
        position_key = self._position_key(metadata)
        yaw = self._yaw(metadata)
        if (
            position_key is None
            or yaw is None
            or position_key in self._reorientation_attempted
        ):
            return None
        blocked_headings = {
            heading
            for x, z, heading in self._blocked
            if (x, z) == position_key
        }
        remaining_actions = [
            action
            for action in _MOVEMENT_YAW_OFFSETS
            if self._movement_heading(yaw, action) not in blocked_headings
        ]
        if len(remaining_actions) != 1:
            return None
        self._reorientation_attempted.add(position_key)
        self._escape_replan_credit = 1
        return _TURNS_TO_FACE_ACTION[remaining_actions[0]]

    def consume_escape_replan_credit(self) -> bool:
        """Allow one correction after the automatic escape changes the view."""
        if self._escape_replan_credit <= 0:
            return False
        self._escape_replan_credit -= 1
        return True
