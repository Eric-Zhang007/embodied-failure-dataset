import tempfile

import pytest

from src.branch_runner import BranchConfig, BranchRunner
from src.episode_manager import EpisodeManager
from src.scheduler import InvalidForkLineage, Scheduler, SchedulerConfig


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


def _metadata() -> dict:
    return {
        "task_goal": "Put an Apple in a Bowl.",
        "scene": "FloorPlan1",
        "task_type": "pick_and_place_simple",
    }


def _legacy_task(branch_id: str, parent_branch_id: str) -> dict:
    return {
        "episode_id": "episode-1",
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "shared_context_step_ids": [],
        "diverges_at_step_id": None,
        "fork_config": {},
    }


def _scheduler(output_dir: str) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(output_dir=output_dir, api_key="test-key")
    scheduler.stats = {"fork_rejected": 0}
    return scheduler


def test_episode_manager_backfills_depth_in_registry_and_nested_task():
    with tempfile.TemporaryDirectory() as output_dir:
        manager = EpisodeManager("episode-1", output_dir, _metadata())
        manager.add_pending_fork(_legacy_task("opaque-child", "main"))

        manager.set_pending_fork_depth("opaque-child", 1)

        entry = EpisodeManager.load(manager.file_path).data["pending_forks"][0]
        assert entry["fork_depth"] == 1
        assert entry["task"]["fork_depth"] == 1


def test_episode_manager_rejects_pending_fork_without_removing_audit_record():
    with tempfile.TemporaryDirectory() as output_dir:
        manager = EpisodeManager("episode-1", output_dir, _metadata())
        manager.add_pending_fork(_legacy_task("opaque-too-deep", "main"))

        manager.reject_pending_fork(
            "opaque-too-deep",
            termination_reason="fork_depth_limit",
            fork_depth=4,
            diagnostic="maximum fork depth is 3",
        )

        reloaded = EpisodeManager.load(manager.file_path)
        entry = reloaded.data["pending_forks"][0]
        assert entry["state"] == "rejected"
        assert entry["termination_reason"] == "fork_depth_limit"
        assert entry["fork_depth"] == 4
        assert entry["diagnostic"] == "maximum fork depth is 3"
        assert reloaded.get_pending_fork_tasks() == []
        assert reloaded.data["final_outcome"] is None


def test_legacy_depth_comes_from_parent_links_not_branch_name():
    with tempfile.TemporaryDirectory() as output_dir:
        manager = EpisodeManager("episode-1", output_dir, _metadata())
        first = _legacy_task("not-a-fork-name", "main")
        second = _legacy_task("also-opaque", first["branch_id"])
        third = _legacy_task("still-opaque", second["branch_id"])
        for task in (first, second, third):
            manager.add_pending_fork(task)

        scheduler = _scheduler(output_dir)

        assert scheduler._resolve_fork_depth(manager, first) == 1
        assert scheduler._resolve_fork_depth(manager, second) == 2
        assert scheduler._resolve_fork_depth(manager, third) == 3


def test_legacy_depth_rejects_missing_parent():
    with tempfile.TemporaryDirectory() as output_dir:
        manager = EpisodeManager("episode-1", output_dir, _metadata())
        task = _legacy_task("child", "missing-parent")
        manager.add_pending_fork(task)

        with pytest.raises(InvalidForkLineage, match="missing parent"):
            _scheduler(output_dir)._resolve_fork_depth(manager, task)


def test_legacy_depth_rejects_parent_cycle():
    with tempfile.TemporaryDirectory() as output_dir:
        manager = EpisodeManager("episode-1", output_dir, _metadata())
        first = _legacy_task("first", "second")
        second = _legacy_task("second", "first")
        manager.add_pending_fork(first)
        manager.add_pending_fork(second)

        with pytest.raises(InvalidForkLineage, match="cycle"):
            _scheduler(output_dir)._resolve_fork_depth(manager, first)


def test_prepare_backfills_valid_legacy_depth():
    with tempfile.TemporaryDirectory() as output_dir:
        manager = EpisodeManager("episode-1", output_dir, _metadata())
        task = _legacy_task("opaque-child", "main")
        manager.add_pending_fork(task)

        prepared = _scheduler(output_dir)._prepare_fork_task(manager, task)

        assert prepared["fork_depth"] == 1
        persisted = EpisodeManager.load(manager.file_path).get_pending_fork_tasks()[0]
        assert persisted["fork_depth"] == 1


def test_prepare_rejects_depth_four_before_worker_admission():
    with tempfile.TemporaryDirectory() as output_dir:
        manager = EpisodeManager("episode-1", output_dir, _metadata())
        parent = "main"
        tasks = []
        for depth in range(1, 5):
            task = _legacy_task(f"opaque-{depth}", parent)
            manager.add_pending_fork(task)
            tasks.append(task)
            parent = task["branch_id"]

        scheduler = _scheduler(output_dir)

        assert scheduler._prepare_fork_task(manager, tasks[-1]) is None
        entry = EpisodeManager.load(manager.file_path).data["pending_forks"][-1]
        assert entry["state"] == "rejected"
        assert entry["termination_reason"] == "fork_depth_limit"
        assert entry["fork_depth"] == 4
        assert scheduler.stats["fork_rejected"] == 1


def test_prepare_rejects_invalid_lineage_as_scheduler_outcome():
    with tempfile.TemporaryDirectory() as output_dir:
        manager = EpisodeManager("episode-1", output_dir, _metadata())
        task = _legacy_task("child", "missing-parent")
        manager.add_pending_fork(task)
        scheduler = _scheduler(output_dir)

        assert scheduler._prepare_fork_task(manager, task) is None
        entry = EpisodeManager.load(manager.file_path).data["pending_forks"][0]
        assert entry["state"] == "rejected"
        assert entry["termination_reason"] == "invalid_fork_lineage"
        assert "missing parent" in entry["diagnostic"]
        assert scheduler.stats["fork_rejected"] == 1
