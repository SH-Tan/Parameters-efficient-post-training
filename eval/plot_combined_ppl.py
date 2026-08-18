import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/xdg-cache-{os.environ.get('USER', 'user')}")
sys.path.insert(0, os.path.dirname(__file__))

from result_utils import draw_combined_ppl_comparisons


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_roots", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_sparsity", type=float, default=0.5)
    parser.add_argument("--pp_seq_len", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["wanda", "magnitude", "sparsegpt", "random"],
        choices=["wanda", "magnitude", "sparsegpt", "random"],
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
    drawn_paths = draw_combined_ppl_comparisons(
        args.run_roots,
        args.output_dir,
        methods=args.methods,
        score_orders=args.score_orders,
        max_sparsity=args.max_sparsity,
        pp_seq_len=args.pp_seq_len,
    )
    if not drawn_paths:
        print("No combined PPL plots were drawn. Check --run_roots, --methods, and --score_orders.")
        sys.exit(1)
    for path in drawn_paths:
        print(path)


if __name__ == "__main__":
    main()
