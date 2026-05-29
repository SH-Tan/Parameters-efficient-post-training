import argparse
import os
import sys

from result_utils import draw_dataset_comparisons

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/xdg-cache-{os.environ.get('USER', 'user')}")


def _dataset_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use DATASET=RUN_ROOT, for example c4=out/qwen2.5_0.5b/20260529_145310")
    dataset, run_root = value.split("=", 1)
    if not dataset or not run_root:
        raise argparse.ArgumentTypeError("Use DATASET=RUN_ROOT with non-empty values.")
    return dataset, run_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_run",
        action="append",
        type=_dataset_run,
        required=True,
        help="Dataset/run pair as DATASET=RUN_ROOT. Repeat for each dataset.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--pp_seq_len", type=int, default=1024)
    parser.add_argument("--max_sparsity", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["wanda", "magnitude", "sparsegpt"],
        choices=["wanda", "magnitude", "sparsegpt"],
    )
    parser.add_argument(
        "--score_orders",
        nargs="+",
        default=["global", "local", "per_op"],
        choices=["global", "local", "per_op"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    drawn_paths = draw_dataset_comparisons(
        args.dataset_run,
        args.output_dir,
        args.seq_len,
        methods=args.methods,
        score_orders=args.score_orders,
        max_sparsity=args.max_sparsity,
        pp_seq_len=args.pp_seq_len,
    )
    if not drawn_paths:
        print("No dataset comparison plots were drawn. Check --dataset_run, --methods, and --score_orders.")
        sys.exit(1)
    for path in drawn_paths:
        print(path)


if __name__ == "__main__":
    main()
