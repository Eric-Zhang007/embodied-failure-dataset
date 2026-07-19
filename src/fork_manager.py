"""
Fork 管理器。仅保留 _rewrite_reasoning 工具方法。
"""

import numpy as np

from src.vlm_client import VLMClient
from src.oracle_agent import OracleAgent
from src.context_builder import build_eb_history_context


class ForkManager:
    def __init__(self, eb_agent, oracle_agent: OracleAgent, branch_runner, output_dir: str):
        self.eb_agent = eb_agent
        self.oracle_agent = oracle_agent
        self.branch_runner = branch_runner
        self.output_dir = output_dir

    def _rewrite_reasoning(
        self,
        shared_steps: list[dict],
        original_action: str,
        original_reasoning: str,
        alternative_action: dict,
        counterfactual_text: str,
    ) -> str:
        """让外部 Agent 在分岔点重写 EB 的推理文本。"""
        history_text = build_eb_history_context(shared_steps)

        prompt = f"""The following is the shared history before a decision point:

{history_text}

In the original path, the agent chose:
  Action: {original_action}
  Reasoning: "{original_reasoning}"

A counterfactual analysis suggests the agent should have chosen:
  Action: {alternative_action.get('action', '?')}
  Reason: "{counterfactual_text}"

Your task: Write a 2-4 sentence reasoning that makes it sound like the agent naturally chose the alternative action based on the shared history.

Constraints:
1. Must be consistent with the shared history
2. Must NOT mention "counterfactual", "original path", "alternative", "fork", or "correction"
3. Must read like the agent autonomously arrived at this decision"""

        return self.oracle_agent.client.chat_with_image(
            system_prompt="You are rewriting an agent's reasoning to make it self-consistent. Respond with ONLY the rewritten reasoning text, no JSON, no quotes.",
            user_text=prompt,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            json_mode=False,
            temperature=0.3,
            max_tokens=256,
        )
