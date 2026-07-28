"""Weighted fair queue for main episodes and bounded-depth fork work."""

from collections import deque


class FairBranchQueue:
    """Dispatch main and fork work in a 1:2 ratio when both are queued."""

    MAX_FORK_DEPTH = 3
    _SCHEDULE = ("main", "fork", "fork")

    def __init__(self):
        self._main = deque()
        self._forks = {
            depth: deque()
            for depth in range(1, self.MAX_FORK_DEPTH + 1)
        }
        self._cursor = 0

    def append(self, task: dict) -> None:
        config = task["branch_config"]
        depth = config.fork_depth
        if config.branch_id == "main":
            if depth != 0:
                raise ValueError(f"main branch must have fork depth 0, got {depth}")
            self._main.append(task)
            return
        if depth not in self._forks:
            raise ValueError(
                f"fork depth must be between 1 and {self.MAX_FORK_DEPTH}, got {depth}"
            )
        self._forks[depth].append(task)

    def append_main_retry(self, task: dict) -> None:
        config = task["branch_config"]
        if config.branch_id != "main" or config.fork_depth != 0:
            raise ValueError("only a depth-0 main task can be retried")
        self._main.appendleft(task)

    def popleft(self) -> dict:
        if not self:
            raise IndexError("pop from an empty FairBranchQueue")

        scheduled_class = self._SCHEDULE[self._cursor]
        if scheduled_class == "main" and self._main:
            task = self._main.popleft()
        elif scheduled_class == "fork" and self._has_forks():
            task = self._popleft_fork()
        elif self._main:
            task = self._main.popleft()
        else:
            task = self._popleft_fork()

        self._cursor = (self._cursor + 1) % len(self._SCHEDULE)
        return task

    def counts(self) -> dict[str, int]:
        counts = {"main": len(self._main)}
        counts.update({
            f"fork_depth_{depth}": len(queue)
            for depth, queue in self._forks.items()
        })
        return counts

    def __bool__(self) -> bool:
        return bool(self._main) or self._has_forks()

    def __len__(self) -> int:
        return len(self._main) + sum(len(queue) for queue in self._forks.values())

    def _has_forks(self) -> bool:
        return any(self._forks.values())

    def _popleft_fork(self) -> dict:
        for depth in range(1, self.MAX_FORK_DEPTH + 1):
            if self._forks[depth]:
                return self._forks[depth].popleft()
        raise IndexError("pop fork from an empty FairBranchQueue")
