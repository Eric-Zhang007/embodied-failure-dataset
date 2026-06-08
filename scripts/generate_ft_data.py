"""
Generate fine-tuning reasoning from existing episode data.
Reads output/ JSONs + screenshots, calls 32B for each step, writes back.
Usage: uv run python scripts/generate_ft_data.py --api-key sk-xxx --max 1000 --workers 100
"""
import argparse, glob, json, os, sys, time, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from src.vlm_client import VLMClient
from src.context_builder import build_eb_history_context

C_G = "\033[32m"
C_Y = "\033[33m"
C_R = "\033[31m"
C_C = "\033[36m"
C_X = "\033[0m"
C_B = "\033[1m"

class _CF(logging.Formatter):
    MAP = {logging.DEBUG: C_C, logging.INFO: "", logging.WARNING: C_Y, logging.ERROR: C_R, logging.CRITICAL: C_R + C_B}
    def format(self, record):
        c = self.MAP.get(record.levelno, "")
        record.levelname = c + record.levelname + C_X
        if c:
            record.msg = c + str(record.msg) + C_X
        return super().format(record)

logger = logging.getLogger("ft")
h = logging.StreamHandler()
h.setFormatter(_CF("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers = [h]
logger.setLevel(logging.INFO)

RS = "You are an expert embodied agent in a 3D household. "
RS += "The image is your first-person view. "
RS += "The action taken at this step is shown. Write the concise reasoning that should accompany that action. "
RS += "Output ONLY a JSON object: "
RS += '{"reasoning": "<1-2 sentences explaining why this action follows from the image, task, and history>"} '
RS += "Be specific about object types and positions. Do not use pipe-separated sections."


def build_prompt(task_goal, action, params, history):
    lines = [f"Task goal: {task_goal}"]
    lines.append(build_eb_history_context(history))
    lines.append(f"Action taken now: {action}({json.dumps(params, ensure_ascii=False)})")
    lines.append("Output JSON with one concise reasoning string for this action.")
    return "\n".join(lines)


def process_episode(ep_path, client):
    """Process all steps in one episode. Returns (modified, total_steps)."""
    with open(ep_path) as f:
        ep = json.load(f)

    steps = ep["steps"]
    task_goal = ep["task_goal"]

    for i, step in enumerate(steps):
        if step.get("eb_reasoning"):
            continue
        img_path = step["image_path"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image for {ep_path} step {step.get('step_id')}: {img_path}")

        image = np.array(Image.open(img_path))
        history = []
        for j, s in enumerate(steps[:i]):
            history.append({
                "step_id": s.get("step_id", f"s{j}"),
                "branch_id": s.get("branch_id", "main"),
                "parent_step_id": s.get("parent_step_id"),
                "step_index_in_branch": s.get("step_index_in_branch", j),
                "action": s["action"],
                "action_params": s.get("action_params", {}),
                "success": s.get("success", True),
                "error_message": s.get("error_message"),
                "eb_reasoning": s.get("eb_reasoning"),
                "eb_diagnosis": s.get("eb_diagnosis"),
                "eb_recovery_reasoning": s.get("eb_recovery_reasoning"),
                "eb_counterfactual": s.get("eb_counterfactual"),
                "eb_proposed_recovery_action": s.get("eb_proposed_recovery_action"),
            })

        prompt = build_prompt(task_goal, step["action"], step.get("action_params", {}), history)
        resp = client.chat_with_image_json(
            system_prompt=RS,
            user_text=prompt,
            image=image,
            max_tokens=512,
            required_fields=("reasoning",),
        )

        r = resp.get("reasoning", "")
        reasoning = r.strip() if isinstance(r, str) and r.strip() else None
        if reasoning is None:
            raise ValueError(f"Empty reasoning for {ep_path} step {step.get('step_id')}")

        step["eb_reasoning"] = reasoning
        # Flush after each step so partial progress is never lost
        with open(ep_path, "w") as f:
            json.dump(ep, f, indent=2, ensure_ascii=False)

    return any(s.get("eb_reasoning") for s in steps), len(steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    p.add_argument("--data-dir", default="output")
    p.add_argument("--max", type=int, default=100)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    ep_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if not ep_files:
        print("No episodes in " + args.data_dir + "/")
        sys.exit(1)

    total_steps = 0
    done_steps = 0
    for fp in ep_files[:args.max]:
        with open(fp) as f:
            ep = json.load(f)
        for s in ep.get("steps", []):
            total_steps += 1
            if s.get("eb_reasoning"):
                done_steps += 1

    print(f"{len(ep_files)} episodes total, processing {min(len(ep_files), args.max)}")
    print(f"Steps with reasoning: {done_steps}/{total_steps} ({100*done_steps//max(1,total_steps)}%)")
    print(f"Workers: {args.workers}\n")

    ep_files = ep_files[:args.max]
    stats = {"ok": 0, "skip": 0, "crash": 0, "steps": 0}
    t_start = time.time()

    def worker(ep_path):
        c = VLMClient.siliconflow(args.model, args.api_key)
        return process_episode(ep_path, c)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, f): f for f in ep_files}
        done = 0
        for future in as_completed(futures):
            ep_path = futures[future]
            modified, n_steps = future.result()
            stats["episodes"] = stats.get("episodes", 0) + 1
            stats["ok"] += 1
            eid = os.path.basename(ep_path)
            done += 1
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            with open(ep_path) as f:
                with_reasoning = sum(1 for s in json.load(f).get("steps", []) if s.get("eb_reasoning"))
            tag = C_G + "OK" + C_X + " (" + str(with_reasoning) + "/" + str(n_steps) + ")" if with_reasoning > 0 else C_Y + "SKIP" + C_X
            print(f"[{done}/{len(ep_files)}] {eid[:50]} {tag} [{rate:.1f}ep/s]")

    elapsed = time.time() - t_start
    print(f"\n{C_B}Done{C_X} {elapsed:.0f}s. {C_G}OK={stats['ok']}{C_X} {C_Y}SKIP={stats['skip']}{C_X} {C_R}CRASH={stats['crash']}{C_X} STEPS={stats['steps']}")

if __name__ == "__main__":
    main()
