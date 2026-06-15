#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def normalize_metric_name(metric_name):
    return metric_name.strip().replace("arith_inst_retired_", "")


def load_metric_data(csv_path, focus=None, exclude=None):
    focus = set(focus or [])
    exclude = set(exclude or [])
    metric_values = defaultdict(list)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        required_cols = {"rank", "hostname", "metric", "value"}
        missing_cols = required_cols.difference(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                "CSV must contain header: rank,hostname,metric,value. "
                f"Missing: {sorted(missing_cols)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            metric_name = (row.get("metric") or "").strip()
            metric_name = normalize_metric_name(metric_name)
            raw_value = (row.get("value") or "").strip()

            if not metric_name or not raw_value:
                continue

            if metric_name in exclude:
                continue

            if focus and metric_name not in focus:
                continue

            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value '{raw_value}' at CSV line {row_idx}"
                ) from exc

            metric_values[metric_name].append(value)

    if not metric_values:
        raise ValueError("No metric data found in CSV")

    return metric_values


def load_ground_truth_data(csv_path, focus=None, exclude=None):
    focus = set(focus or [])
    exclude = set(exclude or [])
    ground_truth = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Ground-truth CSV has no header row")

        required_cols = {"metric", "value"}
        missing_cols = required_cols.difference(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                "Ground-truth CSV must contain header: metric,value. "
                f"Missing: {sorted(missing_cols)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            metric_name = normalize_metric_name(row.get("metric") or "")
            raw_value = (row.get("value") or "").strip()

            if not metric_name or not raw_value:
                continue

            if metric_name in exclude:
                continue

            if focus and metric_name not in focus:
                continue

            try:
                ground_truth[metric_name] = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ground-truth value '{raw_value}' at CSV line {row_idx}"
                ) from exc

    return ground_truth


def infer_ground_truth_path(csv_path):
    candidates = []
    if csv_path.name.startswith("perf_metrics_"):
        candidates.append(
            csv_path.with_name(csv_path.name.replace("perf_metrics_", "perf_ground_truth_", 1))
        )
    candidates.append(csv_path.with_name(f"{csv_path.stem}_ground_truth{csv_path.suffix}"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def resolve_ground_truth_path(csv_path, ground_truth_arg):
    if ground_truth_arg == "none":
        return None
    if ground_truth_arg == "auto":
        return infer_ground_truth_path(csv_path)
    return Path(ground_truth_arg)


def plot_metrics_boxplot(metric_values, output_path, logscale=False, ground_truth=None):
    metric_names = sorted(metric_values.keys())
    series = [metric_values[name] for name in metric_names]

    fig_width = max(10, len(metric_names) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    ax.boxplot(
        series,
        tick_labels=metric_names,
        patch_artist=True,
        showfliers=True,
        whis=(1, 99),
    )

    if ground_truth:
        truth_x = []
        truth_y = []
        for idx, metric_name in enumerate(metric_names, start=1):
            if metric_name not in ground_truth:
                continue
            value = ground_truth[metric_name]
            if logscale and value <= 0.0:
                continue
            truth_x.append(idx)
            truth_y.append(value)

        if truth_x:
            ax.scatter(
                truth_x,
                truth_y,
                color="green",
                edgecolors="black",
                marker="o",
                s=70,
                zorder=4,
                label="Ground truth",
            )
            ax.legend()

    ax.set_xlabel("Metric")
    ax.set_ylabel("Metric Value")
    if logscale:
        ax.set_yscale("log")
    ax.set_title("Performance Counter Distributions by Metric")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def arg_list(values):
    return [value.strip() for value in values.split(",") if value.strip()] if values else []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-metric box-and-whisker distributions from a perf CSV."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input CSV containing metric name and value columns",
    )
    parser.add_argument(
        "-l", "--logscale", action="store_true",
        help="Use logarithmic scale for y-axis (default: False)",
    )
    parser.add_argument(
        "-e", "--exclude", type=arg_list, default=[],
        help="Metric names to exclude from the plot (default: none)",
    )
    parser.add_argument(
        "-f", "--focus", type=arg_list, default=[],
        help="Metric names to focus on in the plot (default: none)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="perf_plot.png",
        help="Output plot path (default: perf_plot.png)",
    )
    parser.add_argument(
        "--ground-truth",
        default="auto",
        help=(
            "Ground-truth CSV path with metric,value columns. Use 'auto' to look for "
            "perf_ground_truth_<benchmark>.csv next to the input, or 'none' to disable "
            "the overlay (default: auto)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    metric_values = load_metric_data(csv_path, focus=args.focus, exclude=args.exclude)
    ground_truth_path = resolve_ground_truth_path(csv_path, args.ground_truth)
    if ground_truth_path and not ground_truth_path.is_file():
        raise FileNotFoundError(f"Ground-truth CSV not found: {ground_truth_path}")

    ground_truth = None
    if ground_truth_path:
        ground_truth = load_ground_truth_data(
            ground_truth_path,
            focus=args.focus,
            exclude=args.exclude,
        )

    plot_metrics_boxplot(
        metric_values,
        Path(args.output),
        logscale=args.logscale,
        ground_truth=ground_truth,
    )


if __name__ == "__main__":
    main()
