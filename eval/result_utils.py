import csv
import glob
import os


def safe_result_component(value):
    text = str(value).strip().replace(os.sep, "__")
    if os.altsep:
        text = text.replace(os.altsep, "__")
    text = text.replace(":", "_")
    return text.strip("._") or "dataset"


def result_dir(args):
    path = os.path.join(args.save, "results", safe_result_component(args.calib_data), f"seq_len_{args.seqlen}")
    os.makedirs(path, exist_ok=True)
    return path


def append_result_csv(csv_path, row):
    has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    fieldnames = [
        "seed",
        "method",
        "score_order",
        "target_sparsity",
        "actual_sparsity",
        "pp_eval_data",
        "pp_seq_len",
        "ppl_test",
    ]
    with open(csv_path, "a+", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def append_downstream_result_csv(csv_path, row):
    has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    fieldnames = [
        "seed",
        "method",
        "score_order",
        "target_sparsity",
        "actual_sparsity",
        "task_data",
        "backend",
        "pruned_model_path",
        "num_examples",
        "num_scored",
        "num_unscored",
        "accuracy",
        "pass@1",
        "mean_score",
        "score_sum",
        "num_correct",
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
        safe_result_component(calib_data),
        f"seq_len_{seq_len}",
        "pp_eval_results.csv",
    )


def method_downstream_csv(run_root, method, calib_data, seq_len):
    return os.path.join(
        run_root,
        method,
        "results",
        safe_result_component(calib_data),
        f"seq_len_{seq_len}",
        "downstream_task_results.csv",
    )


def existing_method_result_csvs(run_root, methods, calib_data, seq_len):
    csv_paths = []
    for method in methods:
        csv_path = method_result_csv(run_root, method, calib_data, seq_len)
        if os.path.exists(csv_path):
            csv_paths.append(csv_path)
    return csv_paths


def existing_method_downstream_csvs(run_root, methods, calib_data, seq_len):
    csv_paths = []
    for method in methods:
        csv_path = method_downstream_csv(run_root, method, calib_data, seq_len)
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


def _label_from_result_csv(csv_path):
    parts = csv_path.split(os.sep)
    try:
        result_index = parts.index("results")
    except ValueError:
        return os.path.basename(os.path.dirname(csv_path))
    if result_index + 1 >= len(parts):
        return "dataset"
    label = parts[result_index + 1]
    if label.startswith("dataset__"):
        label = label[len("dataset__") :]
    if label.endswith(".parquet"):
        label = label[: -len(".parquet")]
    return label.replace("__", "/")


def _filtered_accuracy_points(rows, method=None, score_order=None, max_sparsity=None):
    points = []
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        if score_order is not None and row.get("score_order") != score_order:
            continue

        sparsity = _float_row_value(row, "target_sparsity")
        accuracy = _float_row_value(row, "accuracy")
        actual_sparsity = _float_row_value(row, "actual_sparsity")
        if sparsity is None or accuracy is None:
            continue
        if max_sparsity is not None and sparsity > max_sparsity:
            continue
        points.append((sparsity, accuracy, actual_sparsity))
    return sorted(points)


def _draw_series(plot_path, title, series, y_log=True, annotate_actual=True, ylabel="Perplexity"):
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
    plt.ylabel(ylabel + (", log scale" if y_log else ""))
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


def draw_combined_ppl_comparisons(
    run_roots,
    output_dir,
    methods=("wanda", "magnitude", "sparsegpt", "random"),
    score_orders=("global", "local", "per_op"),
    max_sparsity=0.5,
    pp_seq_len=None,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return []

    dataset_rows = []
    for run_root in run_roots:
        for csv_path in glob.glob(os.path.join(run_root, "*", "results", "*", "seq_len_*", "pp_eval_results.csv")):
            rows = load_result_rows(csv_path)
            if rows:
                dataset_rows.append((_label_from_result_csv(csv_path), rows))

    if not dataset_rows:
        return []

    method_colors = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, method in enumerate(methods):
        method_colors[method] = color_cycle[index % len(color_cycle)] if color_cycle else None

    dataset_labels = sorted({label for label, _ in dataset_rows})
    linestyles = ["-", "--", "-.", ":"]
    dataset_styles = {
        label: linestyles[index % len(linestyles)]
        for index, label in enumerate(dataset_labels)
    }

    os.makedirs(output_dir, exist_ok=True)
    drawn_paths = []
    for score_order in score_orders:
        plotted = False
        plt.figure(figsize=(8.2, 5.0), dpi=180)
        for dataset_label, rows in dataset_rows:
            for method in methods:
                points = _filtered_points(
                    rows,
                    method=method,
                    score_order=score_order,
                    pp_seq_len=pp_seq_len,
                    max_sparsity=max_sparsity,
                )
                if not points:
                    continue
                plotted = True
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                plt.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    color=method_colors.get(method),
                    linestyle=dataset_styles[dataset_label],
                    label=f"{method} - {dataset_label}",
                )

        if not plotted:
            plt.close()
            continue

        method_handles = [
            Line2D([0], [0], color=method_colors.get(method), marker="o", linewidth=1.8, label=method)
            for method in methods
            if method_colors.get(method) is not None
        ]
        dataset_handles = [
            Line2D([0], [0], color="black", linestyle=dataset_styles[label], linewidth=1.8, label=label)
            for label in dataset_labels
        ]

        plt.yscale("log")
        plt.xlabel("Target sparsity")
        plt.ylabel("Perplexity, log scale")
        title_seq = f", pp_seq_len={pp_seq_len}" if pp_seq_len is not None else ""
        plt.title(f"{score_order} PPL comparison{title_seq}")
        plt.grid(True, which="both", linewidth=0.4, alpha=0.35)
        first_legend = plt.legend(handles=method_handles, title="Method", fontsize=8, title_fontsize=8, loc="upper left")
        plt.gca().add_artist(first_legend)
        plt.legend(handles=dataset_handles, title="Dataset", fontsize=8, title_fontsize=8, loc="upper right")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{score_order}_combined_ppl_to_{max_sparsity:g}.png")
        plt.savefig(plot_path)
        plt.close()
        drawn_paths.append(plot_path)
    return drawn_paths


def draw_run_comparison_plots(
    run_root,
    calib_data,
    seq_len,
    methods=("wanda", "magnitude", "sparsegpt", "random"),
    max_sparsity=0.5,
    pp_seq_len=1024,
):
    csv_paths = existing_method_result_csvs(run_root, methods, calib_data, seq_len)
    output_dir = os.path.join(run_root, "plots", calib_data, f"seq_len_{seq_len}")
    drawn_paths = []
    if csv_paths:
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
    downstream_csv_paths = existing_method_downstream_csvs(run_root, methods, calib_data, seq_len)
    if downstream_csv_paths:
        drawn_paths.extend(
            draw_accuracy_method_comparisons(
                downstream_csv_paths,
                output_dir,
                max_sparsity=max_sparsity,
            )
        )
    return drawn_paths


def draw_accuracy_method_comparisons(
    csv_paths,
    output_dir,
    max_sparsity=0.5,
    score_orders=("global", "local", "per_op"),
):
    rows = load_result_rows_many(csv_paths)
    methods = sorted({row.get("method") for row in rows if row.get("method")})
    drawn_paths = []
    for score_order in score_orders:
        series = [
            (
                method,
                _filtered_accuracy_points(
                    rows,
                    method=method,
                    score_order=score_order,
                    max_sparsity=max_sparsity,
                ),
            )
            for method in methods
        ]
        plot_path = os.path.join(output_dir, f"{score_order}_accuracy_vs_sparsity_to_{max_sparsity:g}.png")
        drawn_path = _draw_series(
            plot_path,
            f"{score_order} accuracy comparison",
            series,
            y_log=False,
            ylabel="Accuracy",
        )
        if drawn_path is not None:
            drawn_paths.append(drawn_path)
    return drawn_paths


def draw_dataset_comparisons(
    dataset_runs,
    output_dir,
    seq_len,
    methods=("wanda", "magnitude", "sparsegpt", "random"),
    score_orders=("global", "local", "per_op"),
    max_sparsity=0.5,
    pp_seq_len=1024,
):
    drawn_paths = []
    for method in methods:
        rows_by_dataset = []
        for calib_data, run_root in dataset_runs:
            csv_path = method_result_csv(run_root, method, calib_data, seq_len)
            rows = load_result_rows(csv_path)
            if rows:
                rows_by_dataset.append((calib_data, rows))

        if len(rows_by_dataset) < 2:
            continue

        for score_order in score_orders:
            series = [
                (
                    calib_data,
                    _filtered_points(
                        rows,
                        method=method,
                        score_order=score_order,
                        pp_seq_len=pp_seq_len,
                        max_sparsity=max_sparsity,
                    ),
                )
                for calib_data, rows in rows_by_dataset
            ]
            plot_path = os.path.join(
                output_dir,
                f"{method}_{score_order}_dataset_compare_to_{max_sparsity:g}.png",
            )
            drawn_path = _draw_series(
                plot_path,
                f"{method} {score_order} dataset comparison, seq_len={pp_seq_len}",
                series,
            )
            if drawn_path is not None:
                drawn_paths.append(drawn_path)
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

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
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


def draw_accuracy_vs_sparsity(csv_path, plot_path):
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
        accuracy = _float_row_value(row, "accuracy")
        if accuracy is None:
            continue
        key = (row["method"], row["score_order"])
        series.setdefault(key, []).append((float(row["target_sparsity"]), accuracy))
    if not series:
        return None

    merged_series = {}
    for key, values in series.items():
        signature = tuple(sorted(values))
        merged_series.setdefault(signature, []).append(key)

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=150)
    for signature, keys in sorted(merged_series.items(), key=lambda item: item[1]):
        values = list(signature)
        label = " / ".join(f"{method}-{score_order}" for method, score_order in keys)
        plt.plot(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            linewidth=1.5,
            label=label,
        )

    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Sparsity")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path
