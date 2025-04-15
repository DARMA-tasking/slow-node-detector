import os
import numpy as np
import matplotlib.pyplot as plt

from detection.utils.Helpers import getNodeNumber

def plotData(x_data, y_data, title, xlabel, save_dir, threshold_pct=0.05, highlights=[]):
    """
    Plots y_data vs. x_data and highlights outliers.
    Saves plots to the same directory as the input file.
    """
    x_size = len(x_data)
    y_size = len(y_data)
    assert x_size == y_size

    # Calculate average
    avg = np.mean(y_data)

    # Determine x-ticks
    n_ticks = 10
    skip = round(x_size/n_ticks) if x_size > n_ticks else 1
    x_ticks = [x_data[i] for i in range(x_size) if i % skip == 0]
    if x_data[-1] not in x_ticks:
        x_ticks.append(x_data[-1])

    # Generate plot
    plt.figure()
    plt.plot(x_data, y_data, zorder=2, label="Data")
    plt.plot(x_data, [avg] * y_size, label="Average", color="tab:green", zorder=1)

    if threshold_pct is not None:
        plt.plot(x_data, [avg * (1 + threshold_pct)] * y_size, label="Threshold", color="tab:purple", zorder=1)

    if len(highlights) > 0:
        x_vals = []
        for i in range(y_size):
            if y_data[i] in highlights:
                x_vals.append(x_data[i])
        s = 's' if len(x_vals) != 1 else ''
        plt.scatter(x_vals, highlights, label=f"Outlier{s}", color="r", marker="*", zorder=3)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(x_ticks)
    plt.ylabel("Time (s)")
    plt.legend()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_name = title.lower().replace(" ", "_")
    save_path = os.path.join(save_dir, f"{save_name}.png")
    plt.savefig(save_path)
    plt.close()

def plotNodes(x_data, y_data, y_mins, y_maxes, title, save_dir, dropped_nodes=[], show_all_ranges=False, hide_all_ranges=False):
    """
    Plots y_data vs. x_data and highlights outliers.
    Saves plots to the same directory as the input file.
    This function is specifically intended to plot nodes,
    as opposed to rank IDs.
    """
    x_size = len(x_data)
    y_size = len(y_data)
    assert x_size == y_size

    # Calculate average and error
    avg = np.mean(y_data)

    # Generate plot
    plt.figure()
    plt.scatter(x_data, y_data, label='Data', zorder=3, s=10)
    plt.plot(x_data, [avg] * y_size, label="Average", color="tab:green", zorder=1)

    if show_all_ranges and not hide_all_ranges:
        all_mins, all_maxes = [], []
        for n_id in x_data:
            idx = x_data.index(n_id)
            n_time = y_data[idx]
            n_min = y_mins[idx]
            n_max = y_maxes[idx]
            all_mins.append(n_time - n_min)
            all_maxes.append(n_max - n_time)
        ranges = [all_mins, all_maxes]
        plt.errorbar(x_data, y_data, yerr=ranges, fmt='none', color='black', elinewidth=0.5, capsize=3, capthick=0.5, zorder=1)

    if len(dropped_nodes) > 0:
        dropped_node_times = []
        mins, maxes = [], []
        for n_id in dropped_nodes:
            idx = x_data.index(n_id)
            n_time = y_data[idx]
            n_min = y_mins[idx]
            n_max = y_maxes[idx]
            dropped_node_times.append(n_time)
            plt.text(n_id - 60, n_time, n_id, ha='center', va='bottom', fontsize=10, color="red")
            mins.append(n_time - n_min)
            maxes.append(n_max - n_time)
        s = "" if len(dropped_node_times) == 1 else "s"
        plt.scatter(dropped_nodes, dropped_node_times, label=f"Dropped Node{s}", color="r", marker="*", zorder=4)
        errors = [mins, maxes]
        if not hide_all_ranges:
            plt.errorbar(dropped_nodes, dropped_node_times, yerr=errors, fmt='none', color='red', elinewidth=1.0 if show_all_ranges else 0.5, capsize=3, capthick=0.5, zorder=4)

    plt.title(title)
    plt.xlabel("Node ID")
    plt.ylabel("Time (s)")
    plt.legend()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_name = title.lower().replace(" ", "_")
    save_path = os.path.join(save_dir, f"{save_name}.png")
    plt.savefig(save_path)
    plt.close()

def plotDroppedNodes(rank_times, rank_to_node_map, dropped_nodes, output_filepath, show_all_ranges=False, hide_all_ranges=False):
    if hide_all_ranges:
        show_all_ranges = False

    # Gather all times on this node
    all_node_data = {}
    for r_id, r_time in rank_times.items():
        node_name = rank_to_node_map[r_id]
        if node_name not in all_node_data:
            all_node_data[node_name] = {
                "all": []
            }
        all_node_data[node_name]["all"].append(r_time)

    # Save any possible y-axis data
    all_node_ids = []
    all_node_ydata = {
        "avg": [],
        "sum": [],
        "max": [],
        "min": [],
        "std": []
    }
    for node_name in all_node_data:
        all_node_ids.append(getNodeNumber(node_name))
        all_node_times = all_node_data[node_name]["all"]
        all_node_ydata["avg"].append(np.mean(all_node_times))
        all_node_ydata["sum"].append(np.sum(all_node_times))
        all_node_ydata["max"].append(np.max(all_node_times))
        all_node_ydata["min"].append(np.min(all_node_times))
        all_node_ydata["std"].append(np.std(all_node_times))

    # Plot data
    plotNodes(
        all_node_ids,          # x: all node names
        all_node_ydata["avg"], # y: average across all ranks on the node
        all_node_ydata["min"],
        all_node_ydata["max"],
        "Average Time Across All Ranks on All Nodes",
        output_filepath,
        dropped_nodes=[getNodeNumber(n_id) for n_id in dropped_nodes],
        show_all_ranges=show_all_ranges,
        hide_all_ranges=hide_all_ranges
    )
