import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

class SlowRankDetector:

    def __init__(self, output_filepath, threshold_pct=0.05):
        self.__filepath = output_filepath
        self.__node_times = {}
        self.__node_breakdowns = {}
        self.__node_to_proc_map = {}
        self.__threshold_pct = threshold_pct

        # Initialize outliers
        self.__outlying_nodes = {}
        self.__outlying_iterations = {}

        # Initialize plots directory
        self.__plots_dir = os.path.join(
            os.path.dirname(output_filepath),
            "plots"
        )
        os.makedirs(self.__plots_dir, exist_ok=True)

    def __parse_output(self):
        """Parses text output from slow_node.cc"""
        with open(self.__filepath, "r") as output:
            for line in output:
                # Parse each line the starts with "gather"
                if line.startswith("gather"):
                    splits = line.split(":")
                    raw_node_info = splits[1].strip()
                    # If the processor name is included in the output
                    if "(" in raw_node_info:
                        node_info = re.findall(
                            r"(\d+)\s+\(([^)]+)\)",
                            raw_node_info
                        )[0]
                        node_id = int(node_info[0])
                        proc_name = node_info[1]
                        self.__node_to_proc_map[node_id] = proc_name
                    else:
                        # Just to prevent errors, map node_id to node_id
                        node_id = int(raw_node_info)
                        self.__node_to_proc_map[node_id] = node_id
                    total_time =  float(splits[2].strip())
                    breakdown = splits[-1].strip()
                    breakdown_list = [float(t) for t in breakdown.split(" ")]

                    # Populate node data dicts
                    self.__node_times[node_id] = total_time
                    self.__node_breakdowns[node_id] = breakdown_list

    def __plot_data(self, x_data, y_data, title, xlabel, highlights=[]):
        """
        Plots y_data vs. x_data (and possibly the avg as well).
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
        plt.plot(x_data, [avg * (1 + self.__threshold_pct)] * y_size, label="Threshold", color="tab:purple", zorder=1)
        if len(highlights) > 0:
            indices = []
            for i in range(y_size):
                if y_data[i] in highlights:
                    indices.append(i)
            plt.scatter(indices, highlights, label="Outlier(s)", color="r", marker="*", zorder=3)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.xticks(x_ticks)
        plt.ylabel("Time (s)")
        plt.legend()

        # Save plot
        save_name = title.lower().replace(" ", "_")
        save_path = os.path.join(self.__plots_dir, f"{save_name}.png")
        plt.savefig(save_path)

    def __find_outliers(self, data):
        avg = np.mean(data)
        threshold = avg * (1.0 + self.__threshold_pct)
        outliers = [elt for elt in data if elt > threshold]
        return outliers

    def __analyze_across_nodes(self):
        """Compares the total execution time across all nodes."""
        node_ids, total_times = zip(*self.__node_times.items())
        avg_time = np.mean(total_times)
        outliers = self.__find_outliers(total_times)

        self.__plot_data(node_ids, total_times, "Across-Node Comparison", "Node ID", outliers)

        for n_id, time in self.__node_times.items():
            if time in outliers:
                self.__outlying_nodes[n_id] = time

    def __analyze_within_nodes(self):
        for node_id, breakdown in self.__node_breakdowns.items():

            avg_time = np.mean(breakdown)
            std_dev = np.std(breakdown)
            outliers = self.__find_outliers(breakdown)
            n_iterations = len(breakdown)
            iters = list(range(n_iterations))

            self.__plot_data(
                iters, breakdown,
                f"Node {node_id} Breakdown", "Iteration",
                outliers)

            if len(outliers) > 0:
                self.__outlying_iterations[node_id] = []

            for t in outliers:
                idx = breakdown.index(t)
                self.__outlying_iterations[node_id].append((idx,t))

    def detect(self):
        self.__parse_output()
        self.__analyze_across_nodes()
        self.__analyze_within_nodes()

        # Gather results
        node_ids, total_times = zip(*self.__node_times.items())
        outlying_nodes = list(self.__outlying_nodes.keys())
        nodes_with_outlying_iterations = list(self.__outlying_iterations.keys())

        node_with_slowest_iteration = -1
        slowest_iteration = -1
        slowest_time = -np.inf
        if len(nodes_with_outlying_iterations) > 0:
            for n_id, (iter, t) in self.__outlying_iterations.items():
                slowest_time = max(slowest_time, t)
                if t == slowest_time:
                    slowest_iteration = iter
                    node_with_slowest_iteration = n_id
        else:
            for n_id, breakdown in self.__node_breakdowns.items():
                slowest_time = max(np.max(breakdown), slowest_time)
                if slowest_time in breakdown:
                    slowest_iteration = np.argmax(breakdown)
                    node_with_slowest_iteration = n_id

        # Print results
        s = 's' if len(outlying_nodes) != 1 else ''
        print("\n----------------------------------------------------------")
        print("Results from Across-Node Analysis")
        print()
        print(f"    {len(outlying_nodes)} Outlier Node{s} (at least {self.__threshold_pct:.0%} slower than the mean): {outlying_nodes}")
        if len(outlying_nodes) > 0:
            print()
            print(f"    Node-to-Processor Mapping for Slow Node{s}: ")
            for node in outlying_nodes:
                print(f"        {node}: {self.__node_to_proc_map[node]}")
            print()
        print(f"    Slowest Node: {node_ids[np.argmax(total_times)]} ({np.max(total_times)}s)")
        print(f"    Fastest Node: {node_ids[np.argmin(total_times)]} ({np.min(total_times)}s)")
        print(f"    Avg Time Across All Nodes: {np.mean(total_times)}s")
        print(f"    Std Dev Across All Nodes: {np.std(total_times)}")
        print()

        print("----------------------------------------------------------")
        print("Results from Intra-Node Analysis")
        print()
        print(f"    {len(nodes_with_outlying_iterations)} Node{s} With Outlying Iterations: {nodes_with_outlying_iterations}")
        print(f"    Slowest Iteration: {slowest_iteration} on Node {node_with_slowest_iteration} ({self.__node_to_proc_map[node_with_slowest_iteration]}) - {slowest_time}s")
        print()

        print(f"View generated plots in {self.__plots_dir}.\n")


def main():
    parser = argparse.ArgumentParser(description='Slow Node Detector script.')
    parser.add_argument('-f', '--filepath', help='Absolute or relative path to the output file from running slow_node executable', required=True)
    args = parser.parse_args()

    filepath = args.filepath if os.path.isabs(args.filepath) else os.path.join(os.getcwd(), args.filepath)

    slowRankDetector = SlowRankDetector(filepath)
    slowRankDetector.detect()

main()
