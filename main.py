import argparse
import os
import tempfile

import torch

from model_utils import safe_hf_login, set_seed
from run_utils import run_score_eval, score_save_dir


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model name or local path.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for calibration sampling.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument("--prune_method", type=str, choices=["wanda", "magnitude", "sparsegpt"], required=True)
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        nargs="+",
        default=[0.0],
        help="One or more sparsity levels for prune eval.",
    )
    parser.add_argument(
        "--score_order",
        type=str,
        nargs="+",
        choices=["global", "local", "per_op"],
        default=["global", "local", "per_op"],
        help="Score ordering for prune eval: global all layers/ops, local per layer, or per op.",
    )
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--save", type=str, default="scores", help="Directory to save outputs.")
    parser.add_argument(
        "--model_device",
        type=str,
        default="auto_free",
        help="Device map for model load. Use auto_free to pick the GPU with most free memory.",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="c4",
        choices=["c4", "wikitext2", "metamathqa_math_500", "MetaMathQA-math-500", "math_500"],
        help="Calibration data for score calculation.",
    )
    parser.add_argument("--seqlen", type=int, default=1024, help="Calibration sequence length.")
    parser.add_argument(
        "--pp_seqlen",
        type=int,
        nargs="*",
        default=[],
        help="Perplexity eval sequence lengths. Defaults to --seqlen.",
    )
    parser.add_argument("--skip_pp_eval", action="store_true", help="Only save scores; skip PPL eval.")
    parser.add_argument(
        "--save_score_pkl",
        dest="save_score_pkl",
        action="store_true",
        default=True,
        help="Persist per-layer score PKLs.",
    )
    parser.add_argument(
        "--no_save_score_pkl",
        dest="save_score_pkl",
        action="store_false",
        help="Use a temporary score directory and remove PKLs after the run.",
    )
    return parser


def main():
    safe_hf_login(os.environ.get("HF_TOKEN"))
    print("# of gpus: ", torch.cuda.device_count())

    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.save_score_pkl:
        run_score_eval(args, score_save_dir(args))
    else:
        with tempfile.TemporaryDirectory(prefix=f"{args.prune_method}_scores_") as tmp_dir:
            run_score_eval(args, tmp_dir)


if __name__ == "__main__":
    main()
