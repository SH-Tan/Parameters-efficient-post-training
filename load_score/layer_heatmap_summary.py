import argparse
import os
import pickle
import re

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/xdg-cache-{os.environ.get('USER', 'user')}")

OPS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_pickle_file(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def layer_paths(score_dir):
    if not os.path.isdir(score_dir):
        print(f"warning: missing score_dir, skip: {score_dir}")
        return []
    names = [
        name for name in os.listdir(score_dir)
        if name.startswith("layer_") and name.endswith(".pkl")
    ]
    return [os.path.join(score_dir, name) for name in sorted(names)]


def extract_op(name):
    name = str(name)
    parts = [part for part in re.split(r"[./:]", name) if part]
    for op in OPS:
        if op in parts or name.endswith(op):
            return op
    return str(parts[-1] if parts else name)


def flatten_score(score):
    if hasattr(score, "detach"):
        score = score.detach().cpu().numpy()
    return np.asarray(score).reshape(-1)


def finite_float(score):
    score = flatten_score(score).astype(np.float32, copy=False)
    return score[np.isfinite(score)]


def layer_id(payload, path):
    if "layer_idx" in payload:
        return int(payload["layer_idx"])
    match = re.search(r"layer_(\d+)\.pkl$", os.path.basename(path))
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer layer id from {path}")


def iter_layer_scores(score_dir):
    for path in layer_paths(score_dir):
        payload = load_pickle_file(path)
        layer = layer_id(payload, path)
        for name, score in payload.get("scores", {}).items():
            op = extract_op(name)
            yield layer, op, score


def load_layer_ops(path):
    payload = load_pickle_file(path)
    layer = layer_id(payload, path)
    return layer, {
        extract_op(name): score
        for name, score in payload.get("scores", {}).items()
    }


def layer_path_map(score_dir):
    mapping = {}
    for path in layer_paths(score_dir):
        payload = load_pickle_file(path)
        mapping[layer_id(payload, path)] = path
        del payload
    return mapping


def summarize_score_dir(score_dir):
    values = {}
    counts = {}
    for layer, op, score in iter_layer_scores(score_dir):
        score = finite_float(score)
        if score.size == 0:
            values[(layer, op)] = np.nan
            counts[(layer, op)] = 0
            continue
        values[(layer, op)] = float(np.mean(np.abs(score)))
        counts[(layer, op)] = int(score.size)
    return values, counts


def draw_heatmap(values, output_path, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    layers = sorted({layer for layer, _ in values})
    present_ops = {op for _, op in values}
    ops = [op for op in OPS if op in present_ops]
    if not layers or not ops:
        return None

    matrix = np.full((len(layers), len(ops)), np.nan, dtype=np.float32)
    layer_to_idx = {layer: idx for idx, layer in enumerate(layers)}
    op_to_idx = {op: idx for idx, op in enumerate(ops)}
    for (layer, op), value in values.items():
        if op in op_to_idx:
            matrix[layer_to_idx[layer], op_to_idx[op]] = value

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(3.0, 2.2), dpi=120)
    image = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(image, fraction=0.046, pad=0.04, label="mean |score|")
    plt.xticks(range(len(ops)), ops, rotation=45, ha="right", fontsize=5)
    step = max(1, len(layers) // 8)
    shown = list(range(0, len(layers), step))
    plt.yticks(shown, [layers[idx] for idx in shown], fontsize=5)
    plt.xlabel("Operation", fontsize=6)
    plt.ylabel("Layer", fontsize=6)
    plt.title(title, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return output_path


def top_mask(score, ratio):
    score = finite_float(score)
    if score.size == 0:
        return np.zeros(0, dtype=bool)
    keep = max(1, int(score.size * ratio))
    idx = np.argpartition(score, -keep)[-keep:]
    mask = np.zeros(score.size, dtype=bool)
    mask[idx] = True
    return mask


def compare_score_dirs(left_dir, right_dir, ratio):
    if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
        print(f"warning: missing compare score_dir, skip: {left_dir} vs {right_dir}")
        return []
    right_paths = layer_path_map(right_dir)
    rows = []
    for left_path in layer_paths(left_dir):
        layer, left_ops = load_layer_ops(left_path)
        if layer not in right_paths:
            continue
        _, right_ops = load_layer_ops(right_paths[layer])
        for op in [op for op in OPS if op in left_ops and op in right_ops]:
            left_mask = top_mask(left_ops[op], ratio)
            right_mask = top_mask(right_ops[op], ratio)
            size = min(left_mask.size, right_mask.size)
            if size == 0:
                continue
            if left_mask.size != right_mask.size:
                left_mask = left_mask[:size]
                right_mask = right_mask[:size]
            intersection = np.logical_and(left_mask, right_mask).sum()
            union = np.logical_or(left_mask, right_mask).sum()
            shift = np.logical_xor(left_mask, right_mask).sum() / size
            overlap = intersection / max(1, min(left_mask.sum(), right_mask.sum()))
            jaccard = 0.0 if union == 0 else intersection / union
            rows.append({
                "layer": layer,
                "op": op,
                "num_scores": size,
                "overlap": float(overlap),
                "jaccard": float(jaccard),
                "shift": float(shift),
            })
        del left_ops, right_ops
    return rows


def write_report(path, comparisons):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for name, rows in comparisons:
            handle.write(f"{name}\n")
            handle.write("=" * len(name) + "\n")
            if not rows:
                handle.write("No matching layer/op scores.\n\n")
                continue
            total = sum(row["num_scores"] for row in rows)
            avg_overlap = sum(row["overlap"] * row["num_scores"] for row in rows) / total
            avg_jaccard = sum(row["jaccard"] * row["num_scores"] for row in rows) / total
            avg_shift = sum(row["shift"] * row["num_scores"] for row in rows) / total
            handle.write(f"valid_layer_ops: {len(rows)}\n")
            handle.write(f"num_scores: {total}\n")
            handle.write(f"top50_overlap_weighted: {avg_overlap:.6f}\n")
            handle.write(f"top50_jaccard_weighted: {avg_jaccard:.6f}\n")
            handle.write(f"top50_shift_weighted: {avg_shift:.6f}\n")
            handle.write("layer,op,num_scores,top50_overlap,top50_jaccard,top50_shift\n")
            for row in rows:
                handle.write(
                    f"{row['layer']},{row['op']},{row['num_scores']},"
                    f"{row['overlap']:.6f},{row['jaccard']:.6f},{row['shift']:.6f}\n"
                )
            handle.write("\n")
    return path


def parse_item(spec):
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected NAME=DIR, got {spec}")
    return parts[0], parts[1]


def parse_compare(spec):
    name, rest = spec.split("=", 1)
    left, right = rest.split(",", 1)
    return name, left, right


def main():
    parser = argparse.ArgumentParser(description="Small per-layer score heatmaps and top-50 similarity reports.")
    parser.add_argument("--score", action="append", required=True, help="NAME=score_dir")
    parser.add_argument("--compare", action="append", default=[], help="NAME=left_score_dir,right_score_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_ratio", type=float, default=0.5)
    args = parser.parse_args()

    plots = []
    for name, score_dir in [parse_item(spec) for spec in args.score]:
        values, _ = summarize_score_dir(score_dir)
        path = draw_heatmap(values, os.path.join(args.output_dir, f"heatmap_{name}.png"), name)
        if path is not None:
            plots.append(path)

    comparisons = []
    for spec in args.compare:
        name, left_dir, right_dir = parse_compare(spec)
        comparisons.append((name, compare_score_dirs(left_dir, right_dir, args.top_ratio)))
    report_path = write_report(os.path.join(args.output_dir, "similarity_shift_report.txt"), comparisons)

    print(f"Output directory: {args.output_dir}")
    print(f"Report: {report_path}")
    print("Plots:")
    for path in plots:
        print(path)


if __name__ == "__main__":
    main()
