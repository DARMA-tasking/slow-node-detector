import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

class SlowRankDetector:

    def __init__(self, datafile, threshold_percentage, sockets_per_node, ranks_per_node):
        # Create empty dicts for data
        self.__rank_times = {}
        self.__rank_breakdowns = {}
        self.__rank_to_proc_map = {}

        # Initialize variables
        self.__n_ranks = 0
        self.__filepath = datafile
        self.__threshold_pct = threshold_percentage
        self.__spn = int(sockets_per_node)
        self.__rpn = int(ranks_per_node)
        self.__rps = self.__rpn / self.__spn

        # Initialize outliers
        self.__slow_ranks = {}
        self.__slow_proc_names = []
        self.__slow_iterations = {}

        # Initialize (and create) directories
        self.__output_dir = os.path.join(
            os.path.dirname(datafile),
            "output")
        self.__plots_dir = os.path.join(
            self.__output_dir,
            "plots")
        os.makedirs(self.__plots_dir, exist_ok=True)

    def __get_n_slow_ranks_on_node(self, node_name):
        """
        Returns the number of ranks in self.__slow_ranks that
        belong to the given node.
        """
        return sum(1 for r_id in self.__slow_ranks if self.__rank_to_proc_map[r_id] == node_name)

    def __is_slow_node(self, node_name):
        """
        Returns True if all of the ranks on one socket of the node
        are considered slow.

        For example, if there are two sockets per node, and half of
        the ranks on a node are "slow," the function will return True.
        """
        # Exit early if possible
        if len(self.__slow_ranks) < self.__rps:
            return False

        # Determine how many slow ranks are on this node
        n_slow_ranks = self.__get_n_slow_ranks_on_node(node_name)

        return n_slow_ranks >= self.__rps

    def get_n_ranks(self):
        """Returns the number of ranks found in the data."""
        return self.__n_ranks

    def __parse_output(self):
        """Parses text output from slow_node.cc"""
        with open(self.__filepath, "r") as output:
            for line in output:
                # Parse each line the starts with "gather"
                if line.startswith("gather"):
                    # splits: ['gather', rank_info, total_time, 'breakdown', [times]]
                    splits = line.split(":")

                    # 1. Determine the Rank ID (and processor name, if present)
                    raw_rank_info = splits[1].strip()
                    # raw_rank_info = 'rank_id (proc)'
                    rank_info = re.findall(
                        r"(\d+)\s+\(([^)]+)\)",
                        raw_rank_info
                    )[0]
                    rank_id = int(rank_info[0])
                    proc_name = rank_info[1]
                    self.__rank_to_proc_map[rank_id] = proc_name

                    # 2. Get the total time for the current rank
                    total_time =  float(splits[2].strip())

                    # 3. Isolate the times for each iteration on the current rank
                    breakdown = splits[4].strip()
                    breakdown_list = [float(t) for t in breakdown.split(" ")]

                    # Populate rank data dicts
                    self.__rank_times[rank_id] = total_time
                    self.__rank_breakdowns[rank_id] = breakdown_list

        # Set the number of ranks
        self.__n_ranks = len(self.__rank_times)

    def __plot_data(self, x_data, y_data, title, xlabel, highlights=[]):
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
        plt.plot(x_data, [avg * (1 + self.__threshold_pct)] * y_size, label="Threshold", color="tab:purple", zorder=1)
        if len(highlights) > 0:
            indices = []
            for i in range(y_size):
                if y_data[i] in highlights:
                    indices.append(i)
            s = 's' if len(indices) != 1 else ''
            plt.scatter(indices, highlights, label=f"Outlier{s}", color="r", marker="*", zorder=3)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.xticks(x_ticks)
        plt.ylabel("Time (s)")
        plt.legend()

        # Save plot
        save_name = title.lower().replace(" ", "_")
        save_path = os.path.join(self.__plots_dir, f"{save_name}.png")
        plt.savefig(save_path)

    def __find_slow_outliers(self, data):
        """
        Finds data points that are some percentage (given by self.__threshold_pct)
        higher than the mean of the data.
        """
        avg = np.mean(data)
        threshold = avg * (1.0 + self.__threshold_pct)
        outliers = [elt for elt in data if elt > threshold]
        return outliers

    def __analyze_across_ranks(self):
        """
        Compares the total execution time across all ranks to
        find any slow (self.__threshold_pct slower than the mean) ranks.
        """
        rank_ids, total_times = zip(*self.__rank_times.items())
        outliers = self.__find_slow_outliers(total_times)

        self.__plot_data(rank_ids, total_times, "Across-Rank Comparison", "Rank ID", outliers)

        for r_id, time in self.__rank_times.items():
            if time in outliers:
                self.__slow_ranks[r_id] = time

        for r_id in self.__slow_ranks.keys():
            node_name = self.__rank_to_proc_map[r_id]
            if self.__is_slow_node(node_name) and node_name not in self.__slow_proc_names:
                self.__slow_proc_names.append(node_name)

    def __analyze_within_ranks(self):
        """
        Compares the execution of each iteration on a single rank to
        find any slow (self.__threshold_pct slower than the mean) iterations.
        """
        for rank_id, breakdown in self.__rank_breakdowns.items():
            outliers = self.__find_slow_outliers(breakdown)
            n_iterations = len(breakdown)
            iters = list(range(n_iterations))

            self.__plot_data(
                iters, breakdown,
                f"Rank {rank_id} Breakdown", "Iteration",
                outliers)

            if len(outliers) > 0:
                self.__slow_iterations[rank_id] = []

            for t in outliers:
                idx = breakdown.index(t)
                self.__slow_iterations[rank_id].append((idx,t))

    def detect(self):
        """
        Main function of the SlowRankDetector class.
        Parses the output file from the slow_node executable
        and identifies any slow ranks or iterations.

        Plots are generated in the same directory as the output
        file.
        """
        self.__parse_output()
        self.__analyze_across_ranks()
        self.__analyze_within_ranks()

        # Gather results
        rank_ids, total_times = zip(*self.__rank_times.items())
        slow_rank_ids = list(self.__slow_ranks.keys())
        ranks_with_outlying_iterations = list(self.__slow_iterations.keys())

        rank_with_slowest_iteration = -1
        slowest_iteration = -1
        slowest_time = -np.inf
        if len(ranks_with_outlying_iterations) > 0:
            for r_id, (iter, t) in self.__slow_iterations.items():
                slowest_time = max(slowest_time, t)
                if t == slowest_time:
                    slowest_iteration = iter
                    rank_with_slowest_iteration = r_id
        else:
            for r_id, breakdown in self.__rank_breakdowns.items():
                slowest_time = max(np.max(breakdown), slowest_time)
                if slowest_time in breakdown:
                    slowest_iteration = np.argmax(breakdown)
                    rank_with_slowest_iteration = r_id

        # Print results
        s = "s" if len(slow_rank_ids) != 1 else ""
        print("\n----------------------------------------------------------")
        print("Results from Across-Rank Analysis")
        print()
        print(f"    {len(slow_rank_ids)} Outlier Ranks{s} (at least {self.__threshold_pct:.0%} slower than the mean): {slow_rank_ids}")
        if len(slow_rank_ids) > 0:
            print()
            print(f"    Rank-to-Processor Mapping for Slow Rank{s}: ")
            for rank in slow_rank_ids:
                print(f"        {rank}: {self.__rank_to_proc_map[rank]}")
            print()
        print(f"    Slowest Rank: {rank_ids[np.argmax(total_times)]} ({np.max(total_times)}s)")
        print(f"    Fastest Rank: {rank_ids[np.argmin(total_times)]} ({np.min(total_times)}s)")
        print(f"    Avg Time Across All Ranks: {np.mean(total_times)}s")
        print(f"    Std Dev Across All Ranks: {np.std(total_times)}")
        print()
        if len(self.__slow_proc_names) > 0:
            print(f"    Slow Nodes:")
            for proc_name in self.__slow_proc_names:
                print(f"        {proc_name} ({self.__get_n_slow_ranks_on_node(proc_name)} slow ranks)")
            print(f"    These nodes will be excluded from the hostfile.")
        else:
            print(f"    No nodes had more than {self.__rps} slow ranks.")
        print()

        print("----------------------------------------------------------")
        print("Results from Intra-Rank Analysis")
        print()
        print(f"    {len(ranks_with_outlying_iterations)} Rank{s} With Outlying Iterations: {ranks_with_outlying_iterations}")
        print(f"    Slowest Iteration: {slowest_iteration} on Rank {rank_with_slowest_iteration} ({self.__rank_to_proc_map[rank_with_slowest_iteration]}) - {slowest_time}s")
        print()

        print(f"View generated plots in {self.__plots_dir}.\n")

    def create_hostfile(self):
        """
        Outputs a hostfile that contains a list of all nodes, omitting
        any slow nodes.
        """
        good_proc_names = set([
            proc_name for proc_name in self.__rank_to_proc_map.values()
            if proc_name not in self.__slow_proc_names
        ])

        hostfile_path = os.path.join(self.__output_dir, "hostfile.txt")
        with open(hostfile_path, "w") as hostfile:
            for proc_name in good_proc_names:
                hostfile.write(proc_name + "\n")

        s = 's' if len(good_proc_names) != 1 else ''
        print(f"hostfile with {len(good_proc_names)} processor{s} has been written to {hostfile_path}")

def main():
    parser = argparse.ArgumentParser(description='Slow Rank Detector script.')
    parser.add_argument('-f', '--filepath', help='Absolute or relative path to the output file from running slow_node executable', required=True)
    parser.add_argument('-t', '--threshold', help='Percentage above average time that indicates a "slow" rank', default=0.05)
    parser.add_argument('-spn', '--spn', help='Number of sockets per node', default=2)
    parser.add_argument('-rpn', '--rpn', help='Number of ranks per node', default=48)
    args = parser.parse_args()

    filepath = args.filepath if os.path.isabs(args.filepath) else os.path.join(os.getcwd(), args.filepath)
    threshold_pct = args.threshold
    spn = args.spn
    rpn = args.rpn

    slowRankDetector = SlowRankDetector(
        datafile=filepath,
        threshold_percentage=threshold_pct,
        sockets_per_node=spn,
        ranks_per_node=rpn)

    slowRankDetector.detect()
    slowRankDetector.create_hostfile()

if __name__ == "__main__":
    main()
