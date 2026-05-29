import argparse
import os
import sys

from result_utils import draw_run_comparison_plots

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/xdg-cache-{os.environ.get('USER', 'user')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--calib_data", default="c4")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--pp_seq_len", type=int, default=1024)
    parser.add_argument("--max_sparsity", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["wanda", "magnitude", "sparsegpt"],
        choices=["wanda", "magnitude", "sparsegpt"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    drawn_paths = draw_run_comparison_plots(
        args.run_root,
        args.calib_data,
        args.seq_len,
        methods=args.methods,
        max_sparsity=args.max_sparsity,
        pp_seq_len=args.pp_seq_len,
    )
    if not drawn_paths:
        print("No comparison plots were drawn. Check --run_root, --calib_data, --seq_len, and --methods.")
        sys.exit(1)
    for path in drawn_paths:
        print(path)


if __name__ == "__main__":
    main()
