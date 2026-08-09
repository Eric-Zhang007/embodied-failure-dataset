"""Simulator backend contract for the platform evolution.

The agent/eval layers should depend on this protocol instead of concrete
simulators (AI2-THOR today; RoboTHOR / MolmoSpaces later). The first adapter
wraps ``EnvController`` without changing its public API, so the existing
pipeline keeps working while call sites migrate gradually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ObsSpec:
    """What observations a backend produces."""

    width: int
    height: int
    channels: int = 3
    # rgb, depth, segmentation, proprio, ...
    sensors: tuple[str, ...] = ("rgb",)


@dataclass(frozen=True)
class ActionSpec:
    """What actions a backend accepts.

    ``discrete_actions`` covers high-level interaction actions (AI2-THOR
    style). ``continuous_action_dim`` covers low-level end-effector/joint
    control needed by VLA policies (0 when unused).
    """

    discrete_actions: tuple[str, ...] = ()
    continuous_action_dim: int = 0


@dataclass
class StateSnapshot:
    """A normalized state observation after a reset or a pass step."""

    frame: Any
    metadata: dict
    task_state: dict


@dataclass
class StepResult:
    """The result of executing one action."""

    success: bool
    error: str | None
    frame: Any
    metadata: dict
    task_state: dict


@runtime_checkable
class SimulatorBackend(Protocol):
    """Minimal contract shared by all simulator backends."""

    backend_name: str
    obs_spec: ObsSpec
    action_spec: ActionSpec

    def reset(
        self,
        *,
        scene: str | None = None,
        scene_state: dict | None = None,
    ) -> StateSnapshot: ...

    def step(self, action: str, **params) -> StepResult: ...

    def state_snapshot(self) -> StateSnapshot: ...

    def close(self) -> None: ...


class EnvControllerLike(Protocol):
    """Duck-typed subset of EnvController needed by the AI2-THOR adapter."""

    def step(self, action: str, **params) -> dict: ...

    def get_state_snapshot(self) -> dict: ...

    def reset_scene(self, scene: str | None = None) -> None: ...

    def reset_to_alfred_scene(self, scene_state: dict) -> None: ...

    def close(self) -> None: ...


class Ai2ThorBackend:
    """SimulatorBackend adapter over the existing EnvController."""

    backend_name = "ai2thor"

    def __init__(
        self,
        env: EnvControllerLike,
        *,
        width: int = 300,
        height: int = 300,
    ):
        self._env = env
        self.obs_spec = ObsSpec(width=width, height=height)
        # Populated as the action contract formalizes; the existing pipeline
        # still calls EnvController directly today.
        self.action_spec = ActionSpec()

    def reset(
        self,
        *,
        scene: str | None = None,
        scene_state: dict | None = None,
    ) -> StateSnapshot:
        if scene_state is not None:
            self._env.reset_to_alfred_scene(scene_state)
        elif scene is not None:
            self._env.reset_scene(scene)
        else:
            self._env.reset_scene()
        return self.state_snapshot()

    def step(self, action: str, **params) -> StepResult:
        result = self._env.step(action, **params)
        return StepResult(
            success=bool(result.get("success")),
            error=result.get("error"),
            frame=result.get("frame"),
            metadata=result.get("metadata") or {},
            task_state=result.get("task_state") or {},
        )

    def state_snapshot(self) -> StateSnapshot:
        snap = self._env.get_state_snapshot()
        return StateSnapshot(
            frame=snap.get("frame"),
            metadata=snap.get("metadata") or {},
            task_state=snap.get("task_state") or {},
        )

    def close(self) -> None:
        self._env.close()
