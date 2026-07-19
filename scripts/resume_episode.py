"""Resume an incomplete episode from its saved JSON."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlm_client import VLMClient
from eb_agent import EBAgent
from oracle_agent import OracleAgent
from executor import ExecutorAgent
from branch_runner import BranchRunner


def main():
    parser = argparse.ArgumentParser(description="Resume a paused/failed episode")
    parser.add_argument("episode_json", help="Path to episode JSON file")
    parser.add_argument("--branch-id", default="main", help="Branch to resume")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base-url", default="https://cdn.9527code.com/v1")
    parser.add_argument("--planner-model", default="gpt-5.5")
    parser.add_argument("--executor-model", default="gpt-5.5")
    parser.add_argument("--oracle-model", default="gpt-5.5")
    parser.add_argument("--planner-reasoning-effort", default="medium")
    parser.add_argument("--executor-reasoning-effort", default="medium")
    parser.add_argument("--oracle-reasoning-effort", default="medium")
    parser.add_argument("--output", default="")
    parser.add_argument("--no-traps", action="store_true", help="Disable Phase 2 trap injection")
    parser.add_argument("--memory", default="semantic", choices=["semantic", "geometric"],
                        help="Memory mode: semantic (receptacle-grouped) or geometric (flat list)")
    args = parser.parse_args()

    planner_client = VLMClient.openai(
        args.planner_model, args.api_key,
        base_url=args.api_base_url,
        reasoning_effort=args.planner_reasoning_effort,
    )
    executor_client = VLMClient.openai(
        args.executor_model, args.api_key,
        base_url=args.api_base_url,
        reasoning_effort=args.executor_reasoning_effort,
    )
    oracle_client = VLMClient.openai(
        args.oracle_model, args.api_key,
        base_url=args.api_base_url,
        reasoning_effort=args.oracle_reasoning_effort,
    )

    eb_agent = EBAgent(planner_client)
    oracle_agent = OracleAgent(oracle_client)
    executor_agent = ExecutorAgent(executor_client)

    output_dir = args.output or os.path.dirname(args.episode_json)
    VLMClient.set_api_log_dir(output_dir)

    episode_path = os.path.abspath(args.episode_json)
    print(f"Resuming {episode_path} branch={args.branch_id} → {output_dir}/")
    result = BranchRunner.resume(
        episode_path=episode_path,
        branch_id=args.branch_id,
        eb_agent=eb_agent,
        oracle_agent=oracle_agent,
        executor_agent=executor_agent,
        output_dir=output_dir,
        enable_fork=False,
        enable_phase2=not args.no_traps,
        memory_mode=args.memory,
    )
    print(f"Result: {result.termination_reason} ({result.total_steps} steps)")


if __name__ == "__main__":
    main()
