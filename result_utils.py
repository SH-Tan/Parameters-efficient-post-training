import csv
import os


def result_dir(args):
    path = os.path.join(args.save, "results", args.calib_data, f"seq_len_{args.seqlen}")
    os.makedirs(path, exist_ok=True)
    return path


def append_pp_result(log_path, method, score_order, target_sparsity, actual_sparsity, pp_seq_len, ppl):
    with open(log_path, "a+", encoding="utf-8") as f:
        print(
            f"{method:<12}{score_order:<12}{target_sparsity:<18.4f}"
            f"{actual_sparsity:<18.4f}{pp_seq_len:<12d}{ppl:<12.4f}",
            file=f,
            flush=True,
        )


def append_result_csv(csv_path, row):
    has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    fieldnames = [
        "method",
        "score_order",
        "target_sparsity",
        "actual_sparsity",
        "pp_seq_len",
        "ppl_test",
    ]
    with open(csv_path, "a+", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def load_result_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def draw_pp_vs_sparsity(csv_path, plot_path):
    rows = load_result_rows(csv_path)
    if not rows:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return None

    series = {}
    for row in rows:
        key = (row["method"], row["score_order"], row["pp_seq_len"])
        series.setdefault(key, []).append(
            (float(row["target_sparsity"]), float(row["ppl_test"]))
        )

    plt.figure(figsize=(8, 5), dpi=150)
    for (method, score_order, pp_seq_len), values in sorted(series.items()):
        values = sorted(values)
        plt.plot(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            linewidth=1.5,
            label=f"{method}-{score_order}-seq{pp_seq_len}",
        )

    plt.xlabel("Sparsity")
    plt.ylabel("Perplexity")
    plt.title("PP vs Sparsity")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path
