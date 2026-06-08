import json
import os


class EpisodeManager:
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
        }

        os.makedirs(output_dir, exist_ok=True)
        self._flush()

    def add_step(self, step: dict):
        self.data["steps"].append(step)
        self._flush()

    def add_initial_trap(self, trap: dict):
        self.data["initial_traps"].append(trap)
        self._flush()

    def add_runtime_trap(self, trap: dict):
        self.data["runtime_traps"].append(trap)
        self._flush()

    def set_final_outcome(self, outcome: dict):
        self.data["final_outcome"] = outcome
        self._flush()

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
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 静态工厂方法：从已有文件恢复
    # ------------------------------------------------------------------
    @staticmethod
    def load(file_path: str) -> "EpisodeManager":
        with open(file_path, "r") as f:
            data = json.load(f)

        mgr = EpisodeManager.__new__(EpisodeManager)
        mgr.episode_id = data["episode_id"]
        mgr.file_path = file_path
        mgr.data = data
        return mgr
