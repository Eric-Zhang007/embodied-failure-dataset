"""Composable task/goal abstractions for long-horizon evaluation.

M0 foundation for composing small ALFRED-style tasks into long-horizon
tasks. Leaf goals bridge to the existing instance-aware completion
checkers in ``task_conditions``; composite goals aggregate children.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.task_conditions import check_task_complete

GoalKind = Literal["place", "toggle", "hold", "state", "composite"]
CompositeOp = Literal["sequence", "and", "or"]

STATE_TO_TASK_TYPE = {
    "heated": "pick_heat_then_place_in_recep",
    "cooled": "pick_cool_then_place_in_recep",
    "cleaned": "pick_clean_then_place_in_recep",
}


@dataclass(frozen=True)
class Goal:
    """One leaf or composite sub-goal.

    Leaf kinds:
    - place: exact ``target_object_id`` inside exact ``receptacle_id``
    - toggle: exact ``target_object_id`` held while exact
      ``toggle_target_id`` is on and visible
    - hold: exact ``target_object_id`` held in hand
    - state: exact object with ``state`` (heated/cooled/cleaned) inside
      exact ``receptacle_id``
    """

    kind: GoalKind
    target_object_id: str | None = None
    receptacle_id: str | None = None
    state: str | None = None
    toggle_target_id: str | None = None
    op: CompositeOp = "sequence"
    children: tuple["Goal", ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    """Versioned description of a runnable (possibly long-horizon) task.

    ``budget`` supports step-based limits: ``max_steps_per_stage`` and
    ``max_steps_total``. Exceeding either terminates the branch with
    ``stage_budget_exceeded`` / ``step_budget_exceeded``.
    """

    task_id: str
    backend: str
    scene: Any
    goals: tuple[Goal, ...]
    schema_version: str = "1.0"
    seed: int = 0
    budget: dict | None = None
    source: str | None = None


@dataclass(frozen=True)
class GoalStatus:
    satisfied: bool
    reason: str = ""


@dataclass(frozen=True)
class TaskProgress:
    """Progress of a (possibly long-horizon) task against its goals."""

    all_complete: bool
    stage_index: int
    stage_statuses: tuple[GoalStatus, ...]
    reason: str = ""


def goal_from_dict(data: dict | None) -> Goal | None:
    """Deserialize a Goal (returns None for malformed input)."""
    if not isinstance(data, dict) or "kind" not in data:
        return None
    children = tuple(
        child
        for child in (goal_from_dict(item) for item in data.get("children") or [])
        if child is not None
    )
    return Goal(
        kind=data["kind"],
        target_object_id=data.get("target_object_id"),
        receptacle_id=data.get("receptacle_id"),
        state=data.get("state"),
        toggle_target_id=data.get("toggle_target_id"),
        op=data.get("op", "sequence"),
        children=children,
    )


def task_spec_from_dict(data: dict | None) -> TaskSpec | None:
    """Deserialize a TaskSpec (returns None when absent or malformed)."""
    if not isinstance(data, dict) or "task_id" not in data or "goals" not in data:
        return None
    goals = tuple(
        goal
        for goal in (goal_from_dict(item) for item in data["goals"])
        if goal is not None
    )
    return TaskSpec(
        task_id=data["task_id"],
        backend=data.get("backend", "ai2thor"),
        scene=data.get("scene"),
        goals=goals,
        schema_version=data.get("schema_version", "1.0"),
        seed=data.get("seed", 0),
        budget=data.get("budget"),
        source=data.get("source"),
    )


def evaluate_goal(metadata: dict, task_state: dict, goal: Goal) -> GoalStatus:
    """Evaluate one goal against the current normalized state."""
    if goal.kind == "composite":
        return _evaluate_composite(metadata, task_state, goal)
    if goal.kind == "hold":
        return _evaluate_hold(metadata, goal)
    return _evaluate_leaf(metadata, task_state, goal)


def evaluate_task_progress(spec: TaskSpec, metadata: dict, task_state: dict) -> TaskProgress:
    """Evaluate all stages in order; stage_index is the first unsatisfied stage."""
    statuses = tuple(evaluate_goal(metadata, task_state, goal) for goal in spec.goals)
    if all(status.satisfied for status in statuses):
        return TaskProgress(
            all_complete=True,
            stage_index=len(statuses),
            stage_statuses=statuses,
        )
    first = next(index for index, status in enumerate(statuses) if not status.satisfied)
    return TaskProgress(
        all_complete=False,
        stage_index=first,
        stage_statuses=statuses,
        reason=statuses[first].reason,
    )


def evaluate_episode_progress(ep_data: dict, metadata: dict, task_state: dict) -> TaskProgress:
    """Progress for an episode: composed tasks use TaskSpec goals, otherwise
    the legacy single-task completion check."""
    spec = task_spec_from_dict(ep_data.get("task_spec"))
    if spec is not None:
        return evaluate_task_progress(spec, metadata, task_state)
    ok, reason = check_task_complete(metadata, ep_data, task_state)
    return TaskProgress(
        all_complete=ok,
        stage_index=0 if not ok else 1,
        stage_statuses=(GoalStatus(ok, reason),),
        reason=reason,
    )


def goal_text(goal: Goal) -> str:
    """Human-readable description of one goal (instance-level)."""
    if goal.kind == "composite":
        separator = " OR " if goal.op == "or" else " AND "
        return separator.join(goal_text(child) for child in goal.children) if goal.children else "(empty)"
    if goal.kind == "place":
        return f"{goal.target_object_id} must be inside {goal.receptacle_id}"
    if goal.kind == "hold":
        return f"{goal.target_object_id} must be held in hand"
    if goal.kind == "toggle":
        return f"{goal.target_object_id} must be held while {goal.toggle_target_id} is on and visible"
    if goal.kind == "state":
        return f"{goal.target_object_id} must be {goal.state} and inside {goal.receptacle_id}"
    return str(goal)


def task_spec_criteria_text(spec: TaskSpec) -> str:
    """Multi-stage completion criteria for long-horizon prompts."""
    lines = [f"Long-horizon task with {len(spec.goals)} stage(s):"]
    for index, goal in enumerate(spec.goals, 1):
        lines.append(f"Stage {index}: {goal_text(goal)}")
    return "\n".join(lines)


def _evaluate_composite(metadata: dict, task_state: dict, goal: Goal) -> GoalStatus:
    if goal.op == "or":
        reasons = []
        for child in goal.children:
            status = evaluate_goal(metadata, task_state, child)
            if status.satisfied:
                return GoalStatus(True)
            if status.reason:
                reasons.append(status.reason)
        return GoalStatus(False, " or ".join(reasons) if reasons else "no child satisfied")

    # sequence and and: every child must be satisfied.
    # Stage/order tracking belongs to the runner, not to this predicate.
    for child in goal.children:
        status = evaluate_goal(metadata, task_state, child)
        if not status.satisfied:
            return GoalStatus(False, status.reason or f"child goal not satisfied: {child}")
    return GoalStatus(True)


def _evaluate_hold(metadata: dict, goal: Goal) -> GoalStatus:
    inventory = metadata.get("inventoryObjects") or []
    if inventory and inventory[0].get("objectId") == goal.target_object_id:
        return GoalStatus(True)
    return GoalStatus(False, f"{goal.target_object_id} must be held")


def _evaluate_leaf(metadata: dict, task_state: dict, goal: Goal) -> GoalStatus:
    if goal.kind == "place":
        ep_data = _leaf_ep_data(
            task_type="pick_and_place_simple",
            pddl={
                "object_target": _object_type(metadata, goal.target_object_id),
                "parent_target": _object_type(metadata, goal.receptacle_id),
                "object_sliced": False,
                "mrecep_target": "",
                "toggle_target": "",
            },
            goal_instances={
                "final_put": {
                    "objectId": goal.target_object_id,
                    "receptacleObjectId": goal.receptacle_id,
                }
            },
            goal=goal,
        )
    elif goal.kind == "toggle":
        ep_data = _leaf_ep_data(
            task_type="look_at_obj_in_light",
            pddl={
                "object_target": _object_type(metadata, goal.target_object_id),
                "toggle_target": _object_type(metadata, goal.toggle_target_id),
            },
            goal_instances={
                "pickup_object_ids": [goal.target_object_id],
                "toggle_on_object_ids": [goal.toggle_target_id],
            },
            goal=goal,
        )
    elif goal.kind == "state":
        task_type = STATE_TO_TASK_TYPE.get(goal.state or "")
        ep_data = _leaf_ep_data(
            task_type=task_type,
            pddl={
                "object_target": _object_type(metadata, goal.target_object_id),
                "parent_target": _object_type(metadata, goal.receptacle_id),
                "object_sliced": False,
                "mrecep_target": "",
                "toggle_target": "",
            },
            goal_instances={
                "final_put": {
                    "objectId": goal.target_object_id,
                    "receptacleObjectId": goal.receptacle_id,
                }
            },
            goal=goal,
        )
    else:
        return GoalStatus(False, f"unsupported goal kind: {goal.kind}")

    if ep_data is None:
        return GoalStatus(False, f"incomplete goal spec: {goal}")
    ok, reason = check_task_complete(metadata, ep_data, task_state)
    return GoalStatus(ok, "" if ok else reason)


def _leaf_ep_data(task_type: str | None, pddl: dict, goal_instances: dict, goal: Goal) -> dict | None:
    if task_type is None:
        return None
    if pddl.get("object_target") is None or pddl.get("parent_target") is None:
        if goal.kind != "toggle":
            return None
    if goal.kind == "toggle" and (pddl.get("object_target") is None or pddl.get("toggle_target") is None):
        return None
    return {
        "task_type": task_type,
        "pddl_params": pddl,
        "goal_instances": goal_instances,
    }


def _object_type(metadata: dict, object_id: str | None) -> str | None:
    if not object_id:
        return None
    for obj in metadata.get("objects", []):
        if obj.get("objectId") == object_id:
            return obj.get("objectType")
    return None
