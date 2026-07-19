#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL-8B on expert trajectory data.

Converts episode JSONs -> LLaMA-Factory sharegpt format -> trains with QLoRA.
Default: model learns to output (action, params, reasoning).
Use --reasoning_only to train only reasoning.

Usage:
    # Generate data + config
    python finetune_qwen3vl.py --data_dir output --max_episodes 1000

    # Train
    llamafactory-cli train ft_data/training/train_config.yaml

    # Resume from checkpoint
    llamafactory-cli train ft_data/training/train_config.yaml --resume_from_checkpoint ft_model/checkpoint-500

    # Merge and export
    llamafactory-cli export --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \\
        --adapter_name_or_path ft_model --export_dir ft_model/merged
"""
import argparse, json, glob, os, random, sys
from pathlib import Path

from src.context_builder import build_eb_history_context

# ============================================================
# Data conversion
# ============================================================

def build_input(task_goal, history):
    """Build user prompt from task context and action history."""
    lines = [f"<image>\nTask goal: {task_goal}\n"]
    lines.append(build_eb_history_context(history))
    return "\n".join(lines)


def build_history_step(step, fallback_index):
    """Keep only EB-visible fields needed by the runtime prompt."""
    return {
        "step_id": step.get("step_id", f"s{fallback_index}"),
        "branch_id": step.get("branch_id", "main"),
        "parent_step_id": step.get("parent_step_id"),
        "step_index_in_branch": step.get("step_index_in_branch", fallback_index),
        "action": step["action"],
        "action_params": step.get("action_params", {}),
        "success": step.get("success", True),
        "error_message": step.get("error_message"),
        "eb_reasoning": step.get("eb_reasoning"),
        "eb_diagnosis": step.get("eb_diagnosis"),
        "eb_recovery_reasoning": step.get("eb_recovery_reasoning"),
        "eb_counterfactual": step.get("eb_counterfactual"),
        "eb_proposed_recovery_action": step.get("eb_proposed_recovery_action"),
    }


def build_output(action, params, reasoning, reasoning_only=False):
    """Build assistant output. reasoning_only skips action/params."""
    # Use objectType not objectId (our resolver handles full IDs at runtime)
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, str) and "|" in v:
            clean_params[k] = v.split("|")[0]  # keep only type name
        elif k not in ("forceAction", "moveMagnitude", "placeStationary"):
            clean_params[k] = v

    if reasoning_only:
        return json.dumps({"reasoning": reasoning}, ensure_ascii=False)
    else:
        return json.dumps({
            "action": action,
            "params": clean_params,
            "reasoning": reasoning,
        }, ensure_ascii=False)


def convert_episodes(data_dir, output_file, max_episodes=None, val_split=0.05, reasoning_only=False):
    """Convert episode JSONs to LLaMA-Factory sharegpt format."""
    ep_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if max_episodes:
        ep_files = ep_files[:max_episodes]

    samples = []
    skipped_no_reasoning = 0
    skipped_no_image = 0
    total_steps = 0

    for ep_path in ep_files:
        with open(ep_path) as f:
            ep = json.load(f)
        task_goal = ep.get("task_goal", "")
        steps = ep.get("steps", [])
        if not steps:
            continue

        history = []
        for s in steps:
            total_steps += 1
            reasoning = s.get("eb_reasoning", "")

            if not reasoning:
                skipped_no_reasoning += 1
                history.append(build_history_step(s, len(history)))
                continue

            img_path = s.get("image_path", "")
            if not img_path or not os.path.exists(img_path):
                skipped_no_image += 1
                history.append(build_history_step(s, len(history)))
                continue

            user_text = build_input(task_goal, history)
            assistant_text = build_output(
                s["action"], s.get("action_params", {}), reasoning, reasoning_only
            )

            samples.append({
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "images": [img_path],
                "_split": ep.get("data_split", "train"),
            })

            history.append(build_history_step(s, len(history)))

    if not samples:
        print(f"ERROR: No valid samples! {total_steps} steps total, "
              f"{skipped_no_reasoning} missing reasoning, {skipped_no_image} missing images.")
        print("Run generate_ft_data.py first to fill reasoning fields.")
        sys.exit(1)

    print(f"  {len(ep_files)} episodes, {total_steps} steps")
    print(f"  {len(samples)} valid samples, {skipped_no_reasoning} no-reasoning, {skipped_no_image} no-image")

    # Split by ALFRED split: train / valid_seen(val) / valid_unseen(test)
    train_samples = [s for s in samples if s.pop("_split", "train") == "train"]
    val_samples = [s for s in samples if s.pop("_split", "train") == "valid_seen"]
    test_samples = [s for s in samples if s.pop("_split", "train") == "valid_unseen"]
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    if not val_samples:
        # Fallback: random split if no ALFRED validation data
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * val_split))
        val_samples = samples[:n_val]
        train_samples = samples[n_val:]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    train_file = output_file.replace(".json", "_train.json")
    val_file = output_file.replace(".json", "_val.json")
    test_file = output_file.replace(".json", "_test.json")

    for path, data in [(train_file, train_samples), (val_file, val_samples), (test_file, test_samples)]:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  {os.path.basename(path)}: {len(data)} samples, {os.path.getsize(path)//1024}KB")

    return train_file, val_file


# ============================================================
# Training config generation
# ============================================================

def generate_config(model_name, train_file, val_file, output_dir, config_path,
                    lora_rank=64, lora_alpha=128, batch_size=2, grad_accum=8,
                    learning_rate=2e-4, epochs=3, cutoff_len=2048):
    """Generate LLaMA-Factory YAML config and dataset_info.json."""

    # dataset_info.json for sharegpt format
    dataset_info = {
        "ft_train": {
            "file_name": os.path.basename(train_file),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        },
        "ft_val": {
            "file_name": os.path.basename(val_file),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        },
    }
    info_path = os.path.join(os.path.dirname(train_file), "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"  dataset_info.json written")

    # Main training config
    effective_bs = batch_size * grad_accum
    config = f"""# LLaMA-Factory QLoRA config for Qwen3-VL-8B
# Effective batch size: {batch_size} x {grad_accum} = {effective_bs}

### model
model_name_or_path: {model_name}
trust_remote_code: true
flash_attn: fa3

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {lora_rank}
lora_alpha: {lora_alpha}
lora_target: all
lora_dropout: 0.05

### dataset
dataset: ft_train,ft_val
template: qwen2_vl
cutoff_len: {cutoff_len}
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 4

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 500
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
resume_from_checkpoint: null

### train
per_device_train_batch_size: {batch_size}
gradient_accumulation_steps: {grad_accum}
learning_rate: {learning_rate}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
gradient_checkpointing: true
optim: adamw_torch
weight_decay: 0.01
max_grad_norm: 1.0

### eval
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 500
load_best_model_at_end: true
metric_for_best_model: eval_loss

### data paths
dataset_dir: {os.path.dirname(train_file)}
"""

    with open(config_path, "w") as f:
        f.write(config)
    print(f"  train_config.yaml written")
    print(f"  Effective batch size: {effective_bs}")
    print(f"  LoRA rank={lora_rank} alpha={lora_alpha}")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL-8B on expert trajectory data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data + config (action+reasoning, default)
  python finetune_qwen3vl.py --data_dir output --max_episodes 1000

  # Reasoning only (ablation)
  python finetune_qwen3vl.py --data_dir output --max_episodes 1000 --reasoning_only

  # Generate only config (skip data conversion)
  python finetune_qwen3vl.py --config_only
""")
    p.add_argument("--data_dir", default="output", help="Directory with episode JSONs")
    p.add_argument("--output_dir", default="ft_model", help="Output dir for trained LoRA weights")
    p.add_argument("--max_episodes", type=int, default=1000, help="Max episodes for training")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="HF model name")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--data_output", default="ft_data/training", help="Where to write converted data")
    p.add_argument("--reasoning_only", action="store_true",
                   help="Only train reasoning, not action/params (ablation)")
    p.add_argument("--config_only", action="store_true",
                   help="Skip data conversion, only regenerate config")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    data_json = os.path.join(args.data_output, "ft_data.json")
    config_yaml = os.path.join(args.data_output, "train_config.yaml")

    print("=" * 60)
    print("Qwen3-VL-8B Fine-Tuning Setup")
    print("=" * 60)
    print(f"  Mode: {'reasoning only' if args.reasoning_only else 'action + reasoning'}")
    print(f"  Max episodes: {args.max_episodes}")
    print(f"  Output: {args.output_dir}")
    print()

    if not args.config_only:
        print("[1/2] Converting episode data ...")
        train_file, val_file = convert_episodes(
            args.data_dir, data_json,
            max_episodes=args.max_episodes,
            val_split=args.val_split,
            reasoning_only=args.reasoning_only,
        )
        print()

    print("[2/2] Generating training config ...")
    generate_config(
        args.model, data_json.replace(".json", "_train.json"),
        data_json.replace(".json", "_val.json"),
        args.output_dir, config_yaml,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        learning_rate=args.learning_rate, epochs=args.epochs,
    )

    print(f"""
{'=' * 60}
Ready. Next steps:

  # Train (single GPU)
  llamafactory-cli train {config_yaml}

  # Train (multi-GPU)
  torchrun --nproc_per_node=2 src/train.py {config_yaml}

  # Resume from checkpoint
  llamafactory-cli train {config_yaml} \\
      --resume_from_checkpoint {args.output_dir}/checkpoint-500

  # Merge LoRA -> full model
  llamafactory-cli export \\
      --model_name_or_path {args.model} \\
      --adapter_name_or_path {args.output_dir} \\
      --template qwen2_vl \\
      --finetuning_type lora \\
      --export_dir {args.output_dir}/merged

  # Test the fine-tuned model
  # (set EB model to {args.output_dir}/merged in e2e_test.py)
{'=' * 60}
""")


if __name__ == "__main__":
    main()
