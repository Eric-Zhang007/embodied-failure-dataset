import os

import numpy as np
from PIL import Image


class StepRecorder:
    @staticmethod
    def build_step(
        step_id: str,
        branch_id: str,
        parent_step_id: str | None,
        step_index: int,
        action: str,
        action_params: dict,
        result: dict,
        image_dir: str,
    ) -> dict:
        image_path = os.path.join(image_dir, f"{step_id}.png")
        frame = result.get("frame")
        if frame is not None:
            StepRecorder.save_frame(frame, image_path)

        return {
            "step_id": step_id,
            "branch_id": branch_id,
            "parent_step_id": parent_step_id,
            "step_index_in_branch": step_index,
            "action": action,
            "action_params": action_params,
            "success": result["success"],
            "error_message": result.get("error"),
            "image_path": image_path,
            # Agent 相关字段：底座阶段填 None，Agent 接入后再填充
            "eb_reasoning": None,
            "eb_diagnosis": None,
            "eb_recovery_reasoning": None,
            "eb_counterfactual": None,
            "eb_proposed_recovery_action": None,
            # 外部 Agent 相关字段
            "oracle_injection_decision": None,
            "oracle_diagnosis_correct": None,
            "oracle_ground_truth": None,
            "oracle_counterfactual_grade": None,
            "oracle_counterfactual_gold": None,
            "oracle_recovery_verdict": None,
            # Fork 相关
            "fork_decision": None,
            "fork_metadata": None,
        }

    @staticmethod
    def save_frame(frame: np.ndarray, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(frame).save(path)
