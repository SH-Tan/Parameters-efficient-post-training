import argparse
import os

from score_utils import build_selection, save_selection, summarize_selection, write_summary_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Load saved score PKLs and build reusable selection masks.")
    parser.add_argument("--score_dir", required=True, help="Directory containing layer_XXX.pkl score files.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--side", choices=["low", "high"], default="low")
    parser.add_argument("--order", choices=["global", "local", "per_op"], default="global")
    parser.add_argument("--name", default=None, help="Output file prefix.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ratio < 0 or args.ratio > 1:
        raise ValueError(f"ratio must be in [0, 1], got {args.ratio}")

    selection = build_selection(args.score_dir, ratio=args.ratio, side=args.side, order=args.order)
    rows, selected, total = summarize_selection(selection)

    name = args.name or f"{args.order}_{args.side}_{args.ratio:g}"
    mask_path = os.path.join(args.output_dir, f"{name}_mask.pkl")
    csv_path = os.path.join(args.output_dir, f"{name}_summary.csv")

    save_selection(selection, mask_path, args.score_dir, args.ratio, args.side, args.order)
    write_summary_csv(rows, csv_path)

    print(f"Saved mask: {mask_path}")
    print(f"Saved summary: {csv_path}")
    print(f"Selected {selected}/{total} ({0.0 if total == 0 else selected / total:.6f})")


if __name__ == "__main__":
    main()
