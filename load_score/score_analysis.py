import argparse
import csv
import os

from score_utils import build_selection, jaccard, overlap, selection_flat_bool, summarize_selection


def parse_score_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Expected NAME=DIR, got {spec}")
    name, score_dir = spec.split("=", 1)
    return name, score_dir


def draw_heatmap(matrix, labels, output_path, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping heatmap.")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(max(5, len(labels) * 1.2), max(4, len(labels) * 0.9)), dpi=180)
    plt.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            plt.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def draw_layer_op_heatmap(rows, output_path, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping layer/op heatmap.")
        return None

    layers = sorted({int(row["layer"]) for row in rows})
    ops = sorted({row["op"] for row in rows})
    layer_to_idx = {layer: idx for idx, layer in enumerate(layers)}
    op_to_idx = {op: idx for idx, op in enumerate(ops)}
    matrix = [[0.0 for _ in ops] for _ in layers]
    for row in rows:
        matrix[layer_to_idx[int(row["layer"])]][op_to_idx[row["op"]]] = float(row["ratio"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(max(7, len(ops) * 1.4), max(5, len(layers) * 0.25)), dpi=180)
    plt.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="magma")
    plt.colorbar(label="selected ratio")
    plt.xticks(range(len(ops)), ops, rotation=45, ha="right")
    plt.yticks(range(len(layers)), layers)
    plt.xlabel("operation")
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def write_matrix_csv(matrix, labels, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.8f}" for value in row])
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare saved score selections across methods/models.")
    parser.add_argument("--score", action="append", required=True, help="NAME=score_dir. Repeat for each method/model.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--side", choices=["low", "high"], default="low")
    parser.add_argument("--order", choices=["global", "local", "per_op"], default="global")
    parser.add_argument("--metric", choices=["jaccard", "overlap"], default="jaccard")
    parser.add_argument("--layer_heatmaps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    specs = [parse_score_spec(spec) for spec in args.score]
    selections = []
    for name, score_dir in specs:
        selection = build_selection(score_dir, ratio=args.ratio, side=args.side, order=args.order)
        selections.append((name, selection, selection_flat_bool(selection)))

        if args.layer_heatmaps:
            rows, _, _ = summarize_selection(selection)
            draw_layer_op_heatmap(
                rows,
                os.path.join(args.output_dir, f"{name}_{args.order}_{args.side}_{args.ratio:g}_layer_op.png"),
                f"{name}: {args.order} {args.side} {args.ratio:g}",
            )

    labels = [name for name, _, _ in selections]
    matrix = []
    for _, _, mask_a in selections:
        row = []
        for _, _, mask_b in selections:
            if args.metric == "jaccard":
                row.append(jaccard(mask_a, mask_b))
            else:
                row.append(overlap(mask_a, mask_b))
        matrix.append(row)

    prefix = f"{args.metric}_{args.order}_{args.side}_{args.ratio:g}"
    csv_path = write_matrix_csv(matrix, labels, os.path.join(args.output_dir, f"{prefix}.csv"))
    png_path = draw_heatmap(matrix, labels, os.path.join(args.output_dir, f"{prefix}.png"), prefix)
    print(f"Saved matrix: {csv_path}")
    if png_path is not None:
        print(f"Saved heatmap: {png_path}")


if __name__ == "__main__":
    main()
