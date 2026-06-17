from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .model_accuracy_test import evaluate_model_task_accuracy, load_examples, resolve_dtype
except ImportError:
    from model_accuracy_test import evaluate_model_task_accuracy, load_examples, resolve_dtype


DEFAULT_CHECKPOINT = "/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
DEFAULT_DATASET = "ShuoZheLi/MetaMathQA-math-500"
DEFAULT_OUTPUT = "/data/shuozhe/verl/value_decoding/output/minimal_actor_responses.jsonl"

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")


def has_hf_checkpoint(path: Path) -> bool:
    return (path / "config.json").is_file() and any((path / name).is_file() for name in WEIGHT_FILES)


def resolve_actor_hf_dir(checkpoint_dir: str | Path, *, skip_merge: bool = False) -> Path:
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    candidates = [
        checkpoint_dir / "merged_hf" / "actor",
        checkpoint_dir / "actor",
        checkpoint_dir,
    ]
    for candidate in candidates:
        if has_hf_checkpoint(candidate):
            return candidate

    actor_fsdp_dir = checkpoint_dir / "actor"
    if not any(actor_fsdp_dir.glob("model_world_size_*_rank_*.pt")):
        tried = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No actor HF checkpoint or FSDP shards found. Tried:\n{tried}")
    if skip_merge:
        raise FileNotFoundError(f"Actor checkpoint needs merging, but --skip_merge was set: {actor_fsdp_dir}")

    target_dir = checkpoint_dir / "merged_hf" / "actor"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_fsdp_dir),
        "--target_dir",
        str(target_dir),
    ]
    hf_config_dir = actor_fsdp_dir / "huggingface"
    if (hf_config_dir / "config.json").is_file():
        cmd.extend(["--hf_model_config_path", str(hf_config_dir)])
    subprocess.run(cmd, check=True)
    if not has_hf_checkpoint(target_dir):
        raise RuntimeError(f"Merge completed but no HF actor checkpoint was found at {target_dir}")
    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load one actor checkpoint and generate responses for a parquet dataset.")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default=None, help="Optional dataset column containing ground-truth answers.")
    parser.add_argument("--reward_score_dir", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=500, help="Use -1 for all examples.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    actor_dir = resolve_actor_hf_dir(args.checkpoint_dir, skip_merge=args.skip_merge)
    tokenizer = AutoTokenizer.from_pretrained(actor_dir, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        actor_dir,
        dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    examples = load_examples(
        args.dataset_path,
        tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=args.start_index,
        max_examples=args.max_examples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    metrics = evaluate_model_task_accuracy(
        model,
        tokenizer,
        examples,
        args,
        output_path=args.output_path,
        reward_score_dir=args.reward_score_dir,
    )

    print(f"Wrote {len(examples)} responses to {Path(args.output_path).expanduser()}")
    if metrics["num_scored"]:
        print(json.dumps(metrics, indent=2))
    else:
        print("pass@1 was not computed because no ground-truth answers were found.")


if __name__ == "__main__":
    main()
