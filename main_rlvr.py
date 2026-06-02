import argparse
import os
from pathlib import Path

import torch

from utils.entry_utils import add_common_prune_args, run_prune_args
from utils.model_utils import safe_hf_login, set_seed


DEFAULT_CHECKPOINT = "llm_weights/rlvr_ppo_qwen2.5_0.5B_metamath_global_step_800"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")


def has_hf_checkpoint(path):
    return (path / "config.json").is_file() and any((path / name).is_file() for name in WEIGHT_FILES)


def resolve_actor_hf_dir(checkpoint_dir, skip_merge=False):
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

    raise FileNotFoundError(
        f"Actor checkpoint needs merging before pruning. Run load_sample.py without --skip_merge first: {actor_fsdp_dir}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    add_common_prune_args(
        parser,
        require_model=False,
        default_model=DEFAULT_CHECKPOINT,
        model_help="Local RLVR checkpoint root or merged HF actor path.",
    )
    parser.add_argument("--skip_merge", action="store_true", help="Require an existing merged HF actor checkpoint.")
    return parser


def resolve_model_arg(args):
    model_path = Path(args.model).expanduser()
    if model_path.exists():
        actor_dir = resolve_actor_hf_dir(model_path, skip_merge=args.skip_merge)
        args.model = str(actor_dir)
        print(f"Resolved RLVR actor model: {args.model}")


def main():
    safe_hf_login(os.environ.get("HF_TOKEN"))
    print("# of gpus: ", torch.cuda.device_count())

    args = build_parser().parse_args()
    set_seed(args.seed)
    resolve_model_arg(args)
    run_prune_args(args)


if __name__ == "__main__":
    main()
