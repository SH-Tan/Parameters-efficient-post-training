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


def load_result_rows_many(csv_paths):
    rows = []
    for csv_path in csv_paths:
        rows.extend(load_result_rows(csv_path))
    return rows


def method_result_csv(run_root, method, calib_data, seq_len):
    return os.path.join(
        run_root,
        method,
        "results",
        calib_data,
        f"seq_len_{seq_len}",
        "pp_eval_results.csv",
    )


def existing_method_result_csvs(run_root, methods, calib_data, seq_len):
    csv_paths = []
    for method in methods:
        csv_path = method_result_csv(run_root, method, calib_data, seq_len)
        if os.path.exists(csv_path):
            csv_paths.append(csv_path)
    return csv_paths


def _float_row_value(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _int_row_value(row, key):
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _filtered_points(rows, method=None, score_order=None, pp_seq_len=None, max_sparsity=None):
    points = []
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        if score_order is not None and row.get("score_order") != score_order:
            continue
        if pp_seq_len is not None and _int_row_value(row, "pp_seq_len") != int(pp_seq_len):
            continue

        sparsity = _float_row_value(row, "target_sparsity")
        ppl = _float_row_value(row, "ppl_test")
        actual_sparsity = _float_row_value(row, "actual_sparsity")
        if sparsity is None or ppl is None:
            continue
        if max_sparsity is not None and sparsity > max_sparsity:
            continue
        points.append((sparsity, ppl, actual_sparsity))
    return sorted(points)


def _draw_series(plot_path, title, series, y_log=True, annotate_actual=True):
    series = [(label, points) for label, points in series if points]
    if not series:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return None

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(7.5, 4.8), dpi=180)
    for label, points in series:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        actuals = [point[2] for point in points]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=label)
        if annotate_actual:
            for x, y, actual in zip(xs, ys, actuals):
                if actual is not None:
                    plt.annotate(
                        f"act {actual:.4f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=7,
                    )

    if y_log:
        plt.yscale("log")
    plt.xlabel("Target sparsity")
    plt.ylabel("Perplexity" + (", log scale" if y_log else ""))
    plt.title(title)
    plt.grid(True, which="both", linewidth=0.4, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def draw_score_order_comparisons(
    csv_paths,
    output_dir,
    max_sparsity=0.5,
    pp_seq_len=1024,
    score_orders=("global", "local", "per_op"),
):
    rows = load_result_rows_many(csv_paths)
    methods = sorted({row.get("method") for row in rows if row.get("method")})
    drawn_paths = []
    for method in methods:
        series = [
            (
                score_order,
                _filtered_points(
                    rows,
                    method=method,
                    score_order=score_order,
                    pp_seq_len=pp_seq_len,
                    max_sparsity=max_sparsity,
                ),
            )
            for score_order in score_orders
        ]
        plot_path = os.path.join(output_dir, f"{method}_score_order_compare_to_{max_sparsity:g}.png")
        drawn_path = _draw_series(
            plot_path,
            f"{method} score-order comparison, seq_len={pp_seq_len}",
            series,
        )
        if drawn_path is not None:
            drawn_paths.append(drawn_path)
    return drawn_paths


def draw_method_comparisons(
    csv_paths,
    output_dir,
    max_sparsity=0.5,
    pp_seq_len=1024,
    score_orders=("global", "local", "per_op"),
):
    rows = load_result_rows_many(csv_paths)
    methods = sorted({row.get("method") for row in rows if row.get("method")})
    drawn_paths = []
    for score_order in score_orders:
        series = [
            (
                method,
                _filtered_points(
                    rows,
                    method=method,
                    score_order=score_order,
                    pp_seq_len=pp_seq_len,
                    max_sparsity=max_sparsity,
                ),
            )
            for method in methods
        ]
        plot_path = os.path.join(output_dir, f"{score_order}_method_compare_to_{max_sparsity:g}.png")
        drawn_path = _draw_series(
            plot_path,
            f"{score_order} method comparison, seq_len={pp_seq_len}",
            series,
        )
        if drawn_path is not None:
            drawn_paths.append(drawn_path)
    return drawn_paths


def draw_run_comparison_plots(
    run_root,
    calib_data,
    seq_len,
    methods=("wanda", "magnitude", "sparsegpt"),
    max_sparsity=0.5,
    pp_seq_len=1024,
):
    csv_paths = existing_method_result_csvs(run_root, methods, calib_data, seq_len)
    if not csv_paths:
        return []
    output_dir = os.path.join(run_root, "plots", calib_data, f"seq_len_{seq_len}")
    drawn_paths = []
    drawn_paths.extend(
        draw_score_order_comparisons(
            csv_paths,
            output_dir,
            max_sparsity=max_sparsity,
            pp_seq_len=pp_seq_len,
        )
    )
    drawn_paths.extend(
        draw_method_comparisons(
            csv_paths,
            output_dir,
            max_sparsity=max_sparsity,
            pp_seq_len=pp_seq_len,
        )
    )
    return drawn_paths


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
