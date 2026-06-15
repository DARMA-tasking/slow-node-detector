#!/usr/bin/env python3

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SUBDIR_PATTERN = re.compile(r"^perf_(mp|nomp)_(\d+)r_(\d+)l$")
VALID_TYPES = {"double", "complex"}


def normalize_metric_name(metric_name):
    return metric_name.strip().replace("arith_inst_retired_", "")


def int_list(values):
    return [int(value.strip()) for value in values.split(",") if value.strip()] if values else []


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired mp/nomp boxplots for one metric across loop counts "
            "from perf benchmark output directories, plus a percent-difference plot."
        )
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        required=True,
        help="Root data directory containing perf_[no]mp_[num_ranks]r_[num_loops]l subdirectories",
    )
    parser.add_argument(
        "-m",
        "--metric",
        required=True,
        help="Metric name to plot",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="double",
        choices=sorted(VALID_TYPES),
        help="Benchmark type: double or complex",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        type=int_list,
        default=[],
        help="Comma-separated loop counts to exclude from the plots (default: none)",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default=None,
        help="Output filename prefix (default: metric_<type>_<metric>)",
    )
    parser.add_argument(
        "-l",
        "--logscale",
        action="store_true",
        help="Use logarithmic scale for the boxplot y-axis",
    )
    return parser.parse_args()


def load_metric_series(csv_path, target_metric):
    values = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        required_cols = {"rank", "hostname", "metric", "value"}
        missing_cols = required_cols.difference(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"CSV {csv_path} must contain header rank,hostname,metric,value. "
                f"Missing: {sorted(missing_cols)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            metric_name = normalize_metric_name(row.get("metric") or "")
            raw_value = (row.get("value") or "").strip()

            if not metric_name or not raw_value:
                continue

            if metric_name != target_metric:
                continue

            try:
                values.append(float(raw_value))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value '{raw_value}' at line {row_idx} in {csv_path}"
                ) from exc

    return values


def load_ground_truth_value(csv_path, target_metric):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Ground-truth CSV has no header row: {csv_path}")

        required_cols = {"metric", "value"}
        missing_cols = required_cols.difference(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"Ground-truth CSV {csv_path} must contain header metric,value. "
                f"Missing: {sorted(missing_cols)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            metric_name = normalize_metric_name(row.get("metric") or "")
            raw_value = (row.get("value") or "").strip()

            if not metric_name or not raw_value:
                continue

            if metric_name != target_metric:
                continue

            try:
                return float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ground-truth value '{raw_value}' at line {row_idx} in {csv_path}"
                ) from exc

    return None


def collect_data(data_dir, bench_type, target_metric, excluded_loops=None):
    excluded_loops = set(excluded_loops or [])
    grouped = defaultdict(dict)

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue

        match = SUBDIR_PATTERN.match(subdir.name)
        if not match:
            continue

        mode, num_ranks_str, num_loops_str = match.groups()
        num_ranks = int(num_ranks_str)
        num_loops = int(num_loops_str)

        if num_loops in excluded_loops:
            continue

        metrics_path = subdir / f"perf_metrics_{bench_type}.csv"
        truth_path = subdir / f"perf_ground_truth_{bench_type}.csv"

        if not metrics_path.is_file():
            raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")
        if not truth_path.is_file():
            raise FileNotFoundError(f"Ground-truth CSV not found: {truth_path}")

        metric_values = load_metric_series(metrics_path, target_metric)
        ground_truth = load_ground_truth_value(truth_path, target_metric)

        grouped[num_loops][mode] = {
            "num_ranks": num_ranks,
            "values": metric_values,
            "ground_truth": ground_truth,
            "source_dir": subdir,
        }

    if not grouped:
        raise ValueError(
            f"No matching perf_* subdirectories found under {data_dir} "
            f"after excluding loop counts {sorted(excluded_loops)}"
        )

    return dict(sorted(grouped.items()))


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def plot_grouped_boxplots(grouped_data, metric_name, bench_type, output_path, logscale=False):
    loops_sorted = sorted(grouped_data.keys())

    positions = []
    series = []
    colors = []
    tick_positions = []
    tick_labels = []
    truth_x = []
    truth_y = []

    width = 0.32
    group_gap = 1.25
    current_center = 1.0

    for num_loops in loops_sorted:
        group = grouped_data[num_loops]

        mp_pos = current_center - 0.22
        nomp_pos = current_center + 0.22

        has_any = False

        if "mp" in group and group["mp"]["values"]:
            positions.append(mp_pos)
            series.append(group["mp"]["values"])
            colors.append("#4C72B0")
            has_any = True

        if "nomp" in group and group["nomp"]["values"]:
            positions.append(nomp_pos)
            series.append(group["nomp"]["values"])
            colors.append("#DD8452")
            has_any = True

        tick_positions.append(current_center)
        tick_labels.append(str(num_loops))

        gt_value = None
        for mode in ("mp", "nomp"):
            if mode in group and group[mode]["ground_truth"] is not None:
                gt_value = group[mode]["ground_truth"]
                break

        if gt_value is not None and (not logscale or gt_value > 0.0):
            truth_x.append(current_center)
            truth_y.append(gt_value)

        if has_any:
            current_center += group_gap

    if not series:
        raise ValueError(f"No data found for metric '{metric_name}'")

    fig_width = max(10, len(loops_sorted) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    bp = ax.boxplot(
        series,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
        whis=(1, 99),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)

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

    legend_handles = [
        plt.Line2D([0], [0], color="#4C72B0", lw=8, label="mp"),
        plt.Line2D([0], [0], color="#DD8452", lw=8, label="nomp"),
    ]
    if truth_x:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markeredgecolor="black",
                markersize=8,
                linestyle="None",
                label="Ground truth",
            )
        )
    ax.legend(handles=legend_handles)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Number of loops")
    ax.set_ylabel("Metric Value")
    if logscale:
        ax.set_yscale("log")
    ax.set_title(f"Metric Distribution for {metric_name} ({bench_type})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Boxplot saved to {output_path}")


def plot_percent_difference(grouped_data, metric_name, bench_type, output_path):
    loops_sorted = sorted(grouped_data.keys())

    mp_positions = []
    mp_values = []
    nomp_positions = []
    nomp_values = []
    tick_positions = []
    tick_labels = []

    bar_width = 0.32
    group_gap = 1.25
    current_center = 1.0

    for num_loops in loops_sorted:
        group = grouped_data[num_loops]
        mp_pos = current_center - 0.22
        nomp_pos = current_center + 0.22

        tick_positions.append(current_center)
        tick_labels.append(str(num_loops))

        if "mp" in group:
            gt = group["mp"]["ground_truth"]
            avg = mean(group["mp"]["values"])
            if gt not in (None, 0.0) and avg is not None:
                mp_positions.append(mp_pos)
                mp_values.append(100.0 * (avg - gt) / gt)

        if "nomp" in group:
            gt = group["nomp"]["ground_truth"]
            avg = mean(group["nomp"]["values"])
            if gt not in (None, 0.0) and avg is not None:
                nomp_positions.append(nomp_pos)
                nomp_values.append(100.0 * (avg - gt) / gt)

        current_center += group_gap

    if not mp_values and not nomp_values:
        raise ValueError(
            f"No valid percent-difference data found for metric '{metric_name}'"
        )

    fig_width = max(10, len(loops_sorted) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    if mp_values:
        ax.bar(
            mp_positions,
            mp_values,
            width=bar_width,
            color="#4C72B0",
            alpha=0.85,
            label="mp",
        )
    if nomp_values:
        ax.bar(
            nomp_positions,
            nomp_values,
            width=bar_width,
            color="#DD8452",
            alpha=0.85,
            label="nomp",
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Number of loops")
    ax.set_ylabel("Percent Difference from Ground Truth (%)")
    ax.set_title(f"Percent Difference from Ground Truth for {metric_name} ({bench_type})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Percent-difference plot saved to {output_path}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    metric_name = normalize_metric_name(args.metric)

    output_prefix = args.output_prefix or f"metric_{args.type}_{metric_name}"
    boxplot_path = Path(f"{output_prefix}_boxplot.png")
    pctdiff_path = Path(f"{output_prefix}_pctdiff.png")

    grouped_data = collect_data(
        data_dir,
        args.type,
        metric_name,
        excluded_loops=args.exclude,
    )

    plot_grouped_boxplots(
        grouped_data,
        metric_name,
        args.type,
        boxplot_path,
        logscale=args.logscale,
    )

    plot_percent_difference(
        grouped_data,
        metric_name,
        args.type,
        pctdiff_path,
    )


if __name__ == "__main__":
    main()
