"""
硅基流动 API 推理速度测试。
分别测试 EB 模型 (Qwen3-VL-8B) 和 Oracle 模型 (Qwen3-VL-32B) 的响应时间。

运行（WSL2 中）：
    cd ~/embodied-failure-dataset
    uv run python scripts/bench_api.py --api-key sk-xxx
"""

import argparse
import time
import numpy as np
from PIL import Image

from src.vlm_client import VLMClient


def bench_text(client: VLMClient, label: str, rounds: int = 5):
    """测试纯文本推理速度。"""
    system = "You are a helpful assistant. Respond concisely."
    times = []
    for i in range(rounds):
        t0 = time.time()
        resp = client.chat_text(
            system_prompt=system,
            user_text="What is the capital of France? Reply in one word.",
            json_mode=False,
            max_tokens=50,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  [{label}] text round {i+1}/{rounds}: {elapsed:.1f}s, resp={resp.strip()[:60]}")
    avg = sum(times) / len(times)
    print(f"  [{label}] text avg: {avg:.1f}s (min={min(times):.1f}, max={max(times):.1f})\n")
    return times


def bench_text_json(client: VLMClient, label: str, rounds: int = 5):
    """测试带 JSON 约束（仅 prompt 约束，无 response_format）的推理速度。"""
    system = "You are a JSON-only assistant. Output ONLY valid JSON, no markdown, no explanation."
    times = []
    for i in range(rounds):
        t0 = time.time()
        resp = client.chat_text(
            system_prompt=system,
            user_text='Reply with: {"answer": "Paris"}',
            json_mode=False,  # ★ 不用 response_format
            max_tokens=150,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  [{label}] json-prompt round {i+1}/{rounds}: {elapsed:.1f}s, resp={resp.strip()[:80]}")
    avg = sum(times) / len(times)
    print(f"  [{label}] json-prompt avg: {avg:.1f}s (min={min(times):.1f}, max={max(times):.1f})\n")
    return times


def bench_image(client: VLMClient, label: str, rounds: int = 3):
    """测试带图像的推理速度（模拟 Phase 1 真实场景）。"""
    # 生成一张模拟 AI2-THOR 截图的纯色图像
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :100] = [128, 64, 32]   # 模拟墙壁
    img[:, 100:200] = [200, 180, 150]  # 模拟地板
    img[:, 200:] = [100, 140, 200]  # 模拟家具

    system = "You are an embodied agent. Output JSON: {action, params, reasoning}"
    user_text = """Task goal: Pick up the alarm clock and turn on the lamp.

Objects in view:
  AlarmClock|+00.50|+01.00|-00.80 (pickupable)
  DeskLamp|-00.87|+00.90|-02.44 (toggleable, off)
  Desk|-00.87|-00.01|-02.44

Recent history:
  Step 0: MoveAhead -> OK
  Step 1: RotateRight -> OK

Decide your next action. Output JSON."""

    times = []
    for i in range(rounds):
        t0 = time.time()
        resp = client.chat_with_image(
            system_prompt=system,
            user_text=user_text,
            image=img,
            json_mode=False,  # ★ 不用 response_format
            max_tokens=200,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  [{label}] image round {i+1}/{rounds}: {elapsed:.1f}s, resp={resp.strip()[:100]}")
    avg = sum(times) / len(times)
    print(f"  [{label}] image avg: {avg:.1f}s (min={min(times):.1f}, max={max(times):.1f})\n")
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--eb-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--oracle-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    eb = VLMClient.siliconflow(args.eb_model, args.api_key)
    oracle = VLMClient.siliconflow(args.oracle_model, args.api_key)

    print("=" * 60)
    print(f"EB model: {args.eb_model}")
    print(f"Oracle model: {args.oracle_model}")
    print(f"Rounds per test: {args.rounds}")
    print("=" * 60)
    print()

    # 1. 纯文本测试
    print("--- 1. Pure text (both models) ---")
    bench_text(eb, "EB-8B", args.rounds)
    bench_text(oracle, "Oracle-32B", args.rounds)

    # 2. JSON prompt 约束测试（模拟 Phase 3/4）
    print("--- 2. JSON via prompt constraint (both models) ---")
    bench_text_json(eb, "EB-8B", args.rounds)
    bench_text_json(oracle, "Oracle-32B", args.rounds)

    # 3. 图像推理测试（模拟 Phase 1/2）
    print("--- 3. Image + text (both models) ---")
    bench_image(eb, "EB-8B", args.rounds)
    bench_image(oracle, "Oracle-32B", args.rounds)

    print("Done.")


if __name__ == "__main__":
    main()
