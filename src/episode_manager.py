import json
import os
import threading


class EpisodeManager:
    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(
        self,
        episode_id: str,
        output_dir: str,
        metadata: dict,
    ):
        self.episode_id = episode_id
        self.file_path = os.path.join(output_dir, f"{episode_id}.json")

        self.data = {
            "episode_id": episode_id,
            "task_goal": metadata["task_goal"],
            "scene": metadata["scene"],
            "task_type": metadata["task_type"],
            "alfred_task_type": metadata.get("alfred_task_type"),
            "alfred_task_id": metadata.get("alfred_task_id"),
            "pddl_params": metadata.get("pddl_params", {}),
            "alfred_scene": metadata.get("alfred_scene"),
            "initial_traps": metadata.get("initial_traps", []),
            "runtime_traps": [],
            "steps": [],
            "final_outcome": None,
            "status": "pending",
            "pid": None,
        }

        os.makedirs(output_dir, exist_ok=True)
        self._flush()

    @classmethod
    def _lock_for(cls, file_path: str) -> threading.Lock:
        key = os.path.abspath(file_path)
        with cls._locks_guard:
            if key not in cls._locks:
                cls._locks[key] = threading.Lock()
            return cls._locks[key]

    def _mutate(self, mutation):
        lock = self._lock_for(self.file_path)
        with lock:
            if os.path.exists(self.file_path):
                self._reload_unlocked()
            mutation(self.data)
            self._flush_unlocked()

    def add_step(self, step: dict):
        def append_step(data):
            data["steps"].append(step)

        self._mutate(append_step)

    def add_initial_trap(self, trap: dict):
        def append_trap(data):
            data["initial_traps"].append(trap)

        self._mutate(append_trap)

    def add_runtime_trap(self, trap: dict):
        def append_trap(data):
            data["runtime_traps"].append(trap)

        self._mutate(append_trap)

    def set_final_outcome(self, outcome: dict):
        def set_outcome(data):
            data["final_outcome"] = outcome

        self._mutate(set_outcome)

    def update_final_outcome(self, branch_entry: dict, dedup_stats: dict = None,
                              is_main: bool = False, fork_source_step_id: str = "",
                              counterfactual_verified: bool = False):
        """Atomically read-modify-write final_outcome to prevent races
        when multiple forks update the same episode concurrently."""
        def mutation(data):
            outcome = data.get("final_outcome") or {"main_branch": None, "forks": []}
            if is_main:
                outcome["main_branch"] = branch_entry
                if dedup_stats:
                    outcome["dedup_stats"] = dedup_stats
            else:
                branch_entry["fork_source_step_id"] = fork_source_step_id
                branch_entry["counterfactual_verified"] = counterfactual_verified
                outcome["forks"].append(branch_entry)
            data["final_outcome"] = outcome
        self._mutate(mutation)

    def set_status(self, status: str, pid: int | None = None):
        if status not in {"pending", "running", "completed", "failed", "interrupted"}:
            raise ValueError(f"Invalid episode status: {status}")

        def update_status(data):
            data["status"] = status
            data["pid"] = pid

        self._mutate(update_status)

    def get_steps_for_branch(self, branch_id: str) -> list[dict]:
        return [s for s in self.data["steps"] if s["branch_id"] == branch_id]

    def get_shared_context_step_ids(self, until_step_id: str) -> list[str]:
        """
        返回直到（含）某个 step 为止的所有 step_id，用于 fork 创建。
        仅返回 main 分支上的 step。
        """
        ids = []
        for s in self.data["steps"]:
            if s["branch_id"] != "main":
                continue
            ids.append(s["step_id"])
            if s["step_id"] == until_step_id:
                break
        return ids

    def _flush(self):
        lock = self._lock_for(self.file_path)
        with lock:
            self._flush_unlocked()

    def _flush_unlocked(self):
        temp_path = f"{self.file_path}.{threading.get_ident()}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.file_path)

    def _reload_unlocked(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self._ensure_status_fields()

    def _ensure_status_fields(self):
        if "status" not in self.data:
            self.data["status"] = "completed" if self.data.get("final_outcome") else "interrupted"
        self.data.setdefault("pid", None)

    # ------------------------------------------------------------------
    # 静态工厂方法：从已有文件恢复
    # ------------------------------------------------------------------
    @staticmethod
    def load(file_path: str) -> "EpisodeManager":
        lock = EpisodeManager._lock_for(file_path)
        with lock:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        mgr = EpisodeManager.__new__(EpisodeManager)
        mgr.episode_id = data["episode_id"]
        mgr.file_path = file_path
        mgr.data = data
        mgr._ensure_status_fields()
        return mgr
