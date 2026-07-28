from types import SimpleNamespace

import pytest

from src.branch_runner import BranchConfig, BranchRunner


class CapturingEpisode:
    episode_id = "episode-1"

    def __init__(self):
        self.pending = []

    def add_pending_fork(self, task):
        self.pending.append(task)


def _history(branch_id: str) -> list[dict]:
    return [{
        "step_id": f"{branch_id}__s0",
        "parent_step_id": None,
        "step_index_in_branch": 0,
        "action": "MoveAhead",
    }]


@pytest.mark.parametrize(
    ("parent_depth", "expected_depth"),
    [(0, 1), (1, 2), (2, 3)],
)
def test_generated_fork_persists_parent_depth_plus_one(
    parent_depth, expected_depth,
):
    runner = BranchRunner.__new__(BranchRunner)
    branch_id = "main" if parent_depth == 0 else f"opaque-parent-{parent_depth}"
    config = BranchConfig(
        episode_id="episode-1",
        branch_id=branch_id,
        parent_branch_id=None if parent_depth == 0 else "parent",
        fork_depth=parent_depth,
    )
    episode = CapturingEpisode()

    task = runner._build_fork_task(
        config=config,
        current_step_idx=1,
        counterfactual={
            "target_step": 0,
            "alternative_action": {"action": "MoveBack", "params": {}},
            "reasoning": "avoid the blocked move",
        },
        fallback_recovery={},
        history=_history(branch_id),
        ep=episode,
    )

    assert task["fork_depth"] == expected_depth
    assert episode.pending == [task]
    assert BranchConfig(**task).fork_depth == expected_depth
