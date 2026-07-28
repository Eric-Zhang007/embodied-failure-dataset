from types import SimpleNamespace

import pytest

from src.fair_branch_queue import FairBranchQueue


def _task(name: str, fork_depth: int = 0) -> dict:
    branch_id = "main" if fork_depth == 0 else name
    return {
        "name": name,
        "branch_config": SimpleNamespace(
            branch_id=branch_id,
            fork_depth=fork_depth,
        ),
    }


def test_dispatches_main_then_two_forks_when_both_classes_have_work():
    queue = FairBranchQueue()
    for task in (
        _task("main-1"),
        _task("main-2"),
        _task("fork-1a", 1),
        _task("fork-1b", 1),
        _task("fork-2a", 2),
    ):
        queue.append(task)

    assert [queue.popleft()["name"] for _ in range(5)] == [
        "main-1",
        "fork-1a",
        "fork-1b",
        "main-2",
        "fork-2a",
    ]


@pytest.mark.parametrize(
    ("tasks", "expected"),
    [
        ([_task("main-1"), _task("main-2")], ["main-1", "main-2"]),
        ([_task("fork-1", 1), _task("fork-2", 1)], ["fork-1", "fork-2"]),
    ],
)
def test_nonempty_class_fills_slots_without_idling(tasks, expected):
    queue = FairBranchQueue()
    for task in tasks:
        queue.append(task)

    assert [queue.popleft()["name"] for _ in tasks] == expected
    assert not queue


def test_main_arriving_after_fork_only_work_waits_until_next_main_slot():
    queue = FairBranchQueue()
    queue.append(_task("fork-1", 1))
    queue.append(_task("fork-2", 1))
    queue.append(_task("fork-3", 1))

    assert queue.popleft()["name"] == "fork-1"
    queue.append(_task("main-1"))

    assert queue.popleft()["name"] == "fork-2"
    assert queue.popleft()["name"] == "fork-3"
    assert queue.popleft()["name"] == "main-1"


def test_forks_are_shallowest_first_and_fifo_within_depth():
    queue = FairBranchQueue()
    queue.append(_task("depth-3", 3))
    queue.append(_task("depth-2a", 2))
    queue.append(_task("depth-1a", 1))
    queue.append(_task("depth-2b", 2))
    queue.append(_task("depth-1b", 1))

    assert [queue.popleft()["name"] for _ in range(5)] == [
        "depth-1a",
        "depth-1b",
        "depth-2a",
        "depth-2b",
        "depth-3",
    ]


def test_new_child_stays_behind_existing_work_at_the_same_depth():
    queue = FairBranchQueue()
    queue.append(_task("older-a", 2))
    queue.append(_task("older-b", 2))

    assert queue.popleft()["name"] == "older-a"
    queue.append(_task("new-child", 2))

    assert [queue.popleft()["name"] for _ in range(2)] == [
        "older-b",
        "new-child",
    ]


def test_main_retry_goes_to_main_head_but_remains_weight_gated():
    queue = FairBranchQueue()
    queue.append(_task("main-old"))
    queue.append(_task("fork-a", 1))
    queue.append(_task("fork-b", 1))

    assert queue.popleft()["name"] == "main-old"
    queue.append(_task("main-later"))
    queue.append_main_retry(_task("main-retry"))

    assert queue.popleft()["name"] == "fork-a"
    assert queue.popleft()["name"] == "fork-b"
    assert queue.popleft()["name"] == "main-retry"
    assert queue.popleft()["name"] == "main-later"


def test_queues_have_independent_fairness_cursors():
    first = FairBranchQueue()
    second = FairBranchQueue()
    for queue, suffix in ((first, "a"), (second, "b")):
        queue.append(_task(f"main-{suffix}"))
        queue.append(_task(f"fork-{suffix}", 1))

    assert first.popleft()["name"] == "main-a"
    assert second.popleft()["name"] == "main-b"
    assert first.popleft()["name"] == "fork-a"
    assert second.popleft()["name"] == "fork-b"


def test_counts_report_main_and_each_fork_depth():
    queue = FairBranchQueue()
    for task in (
        _task("main-a"),
        _task("main-b"),
        _task("fork-1a", 1),
        _task("fork-1b", 1),
        _task("fork-1c", 1),
        _task("fork-2", 2),
    ):
        queue.append(task)

    assert queue.counts() == {
        "main": 2,
        "fork_depth_1": 3,
        "fork_depth_2": 1,
        "fork_depth_3": 0,
    }
    assert len(queue) == 6


@pytest.mark.parametrize("depth", [-1, 4])
def test_rejects_fork_depths_outside_the_scheduler_contract(depth):
    queue = FairBranchQueue()

    with pytest.raises(ValueError, match="fork depth"):
        queue.append(_task("invalid", depth))


def test_hundred_dispatch_canary_keeps_one_to_two_weight_and_bounded_depth():
    queue = FairBranchQueue()
    for index in range(3):
        queue.append(_task(f"main-{index}"))
        queue.append(_task(f"fork-{index}", (index % 3) + 1))

    selected_classes = []
    for index in range(100):
        selected = queue.popleft()
        config = selected["branch_config"]
        selected_classes.append("main" if config.branch_id == "main" else "fork")
        assert 0 <= config.fork_depth <= FairBranchQueue.MAX_FORK_DEPTH

        if config.branch_id == "main":
            queue.append(_task(f"main-refill-{index}"))
        else:
            queue.append(_task(f"fork-refill-{index}", config.fork_depth))

    assert selected_classes.count("main") == 34
    assert selected_classes.count("fork") == 66
