import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

class SlowNodeDetector:
    """
    The SlowNodeDetector analyzes the output from the `slow_node` executable
    and outputs relevant information related to the processing speed and temperature
    of the ranks used during execution.

    There are two main methods of the SlowNodeDetector:

        detect(): This will print out information regarding slow and/or over-heated
            ranks, along with the sockets and nodes they reside on.

        create_hostfile(): This will generate a `hostfile.txt` with all "good" nodes.
            This file can be used in future jobs to ensure that slow nodes are
            avoided. Importantly, nodes are only omitted from the hostfile if
            the number of slow ranks on that node surpasses the size of a socket.

    The following terminology will be used through the SlowNodeDetector:

        Rank: An MPI process
        Core: Processing unit on a socket
        Socket: Collection of cores on a node
        Node: Computing unit in a cluster
    """

    def __init__(self, datafile, num_nodes, threshold_percentage, sockets_per_node, ranks_per_node, plot_rank_breakdowns):
        # Create empty dicts for storing data
        self.__rank_times = {}
        self.__rank_breakdowns = {}
        self.__rank_to_node_map = {}   # Maps each rank to the name of its corresponding node
        self.__rank_to_core_map = {}   # Maps the rank IDs to core IDs (from sensors output)
        self.__rank_to_socket_map = {} # Maps the rank IDs to the socket IDs
        self.__socket_temps = {}

        # Structure of self.__socket_temps:
        # {
        #     socket_id: {
        #         "temperature": float,
        #         "high": float,
        #         "cores": {
        #             core_id: {
        #                 "temperature": float
        #                 "high": float
        #             }
        #         }
        #     }
        # }

        # Initialize variables
        self.__filepath = datafile
        self.__num_nodes = int(num_nodes) if num_nodes is not None else None
        self.__threshold_pct = float(threshold_percentage)
        self.__spn = int(sockets_per_node)
        self.__rpn = int(ranks_per_node)
        self.__rps = self.__rpn / self.__spn
        self.__avg_socket_temp = 0.0
        self.__temperature_analysis_available = False
        self.__plot_rank_breakdowns = plot_rank_breakdowns
        self.__num_ranks = 0

        # Initialize outliers
        self.__slow_ranks = {}
        self.__slow_rank_slowdowns = {}
        self.__slow_node_names = []
        self.__slow_iterations = {}
        self.__hot_sockets = []
        self.__hot_sockets_diffs = {}
        self.__hot_ranks = []
        self.__hot_ranks_diffs = {}

        # Initialize (and create) directories
        self.__output_dir = os.path.join(
            os.path.dirname(datafile),
            "output")
        self.__plots_dir = os.path.join(
            self.__output_dir,
            "plots")
        os.makedirs(self.__plots_dir, exist_ok=True)


    ###########################################################################
    ## Utilities

    def __get_s(self, lst: list):
        """Helper function for the print statements."""
        return "s" if len(lst) != 1 else ""

    def __match_regex(self, pattern: str, line: str):
        """Helper function for matching regex expressions."""
        match = re.search(pattern, line)
        if match:
            return tuple(match.groups())
        raise RuntimeError(f"regex matching failed on line {line}")

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
        plt.close()

    def __parse_output(self):
        """Parses text output from slow_node.cc"""
        is_parsing_sensors, is_parsing_mapping, is_looking_for_core = False, False, False
        current_socket, current_sensor_core = -1, -1
        current_rank, current_core = -1, -1
        with open(self.__filepath, "r") as output:
            for line in output:

                if line.startswith("gather"):
                    # splits: ['gather', rank_info, total_time, 'breakdown', [times]]
                    splits = line.split(":")

                    # 1. Determine the Rank ID (and node name, if present)
                    raw_rank_info = splits[1].strip()
                    # raw_rank_info = 'rank_id (node)'
                    rank_info = re.findall(
                        r"(\d+)\s+\(([^)]+)\)",
                        raw_rank_info
                    )[0]
                    rank_id = int(rank_info[0])
                    node_name = rank_info[1]
                    self.__rank_to_node_map[rank_id] = node_name

                    # 2. Get the total time for the current rank
                    total_time =  float(splits[2].strip())

                    # 3. Isolate the times for each iteration on the current rank
                    breakdown = splits[4].strip()
                    breakdown_list = [float(t) for t in breakdown.split(" ")]

                    # Populate rank data dicts
                    self.__rank_times[rank_id] = total_time
                    self.__rank_breakdowns[rank_id] = breakdown_list

                elif line.startswith("Sensors output"):
                    self.__temperature_analysis_available = True
                    is_parsing_sensors = True
                    is_parsing_mapping = False

                # Note: The following comments use sample `sensors` output from StackExchange:
                # https://unix.stackexchange.com/questions/740689/why-arent-the-cpu-core-numbers-in-sensors-output-consecutive

                elif is_parsing_sensors and line.startswith("Package id"):
                    # Package id 0:  +73.0°C  (high = +80.0°C, crit = +100.0°C)
                    socket_pattern = r"Package id\s+(\d+):\s+\+([\d.]+) C\s+\(high = \+([\d.]+) C"
                    socket_id, socket_temp, socket_high = self.__match_regex(socket_pattern, line)
                    current_socket = int(socket_id)
                    self.__socket_temps[current_socket] = {
                        "temperature": float(socket_temp),
                        "high": float(socket_high),
                        "cores": {}
                    }

                elif is_parsing_sensors and line.startswith("Core"):
                    # Core 0:        +46.0°C  (high = +80.0°C, crit = +100.0°C)
                    core_pattern = r"Core\s+(\d+):\s+\+([\d.]+) C\s+\(high = \+([\d.]+) C"
                    core_id, core_temp, core_high = self.__match_regex(core_pattern, line)
                    current_sensor_core = int(core_id)
                    self.__socket_temps[current_socket]["cores"][current_sensor_core] = {
                        "temperature": float(core_temp),
                        "high": float(core_high)
                    }

                elif line.startswith("Processor to Core ID Mapping"):
                    is_parsing_sensors = False
                    is_parsing_mapping = True

                elif is_parsing_mapping and line.startswith("processor"):
                    rank_pattern = r"processor\s+:\s+(\d+)"
                    current_rank = int(self.__match_regex(rank_pattern, line)[0])
                    # Don't save the rank-to-core mapping for hyperthreaded cores
                    is_looking_for_core = True if current_rank < self.__rpn else False

                elif is_looking_for_core and line.startswith("core id"):
                    core_pattern = r"core id\s+:\s+(\d+)"
                    current_core = int(self.__match_regex(core_pattern, line)[0])
                    self.__rank_to_core_map[current_rank] = current_core
                    try_socket = int(current_rank // self.__rps)
                    socket = try_socket if (try_socket in self.__socket_temps and current_core in self.__socket_temps[try_socket]["cores"]) else -1
                    if socket == -1:
                        raise RuntimeError(f"Could not determine correct socket ID for rank {current_rank} (core {current_core})")
                    self.__rank_to_socket_map[current_rank] = socket
                    is_looking_for_core = False

            self.__num_ranks = len(self.__rank_times)


    ###########################################################################
    ## Secondary analytical functions

    def __get_n_slow_ranks_on_node(self, node_name):
        """
        Returns the number of ranks in self.__slow_ranks that
        belong to the given node.
        """
        return sum(1 for r_id in self.__slow_ranks if self.__rank_to_node_map[r_id] == node_name)

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

    def __sort_nodes_by_execution_time(self, nodes: list):
        """
        Takes in a list of node names and sorts them based on total execution time.
        The fastest nodes will be first, and the slowest will be last.
        """
        node_times = {}
        for r, n in self.__rank_to_node_map.items():
            if n in nodes:
                if n not in node_times:
                    node_times[n] = 0.0
                node_times[n] += self.__rank_times[r]
        # Alternative:
        # return sorted(nodes, key=lambda n: self.__get_n_slow_ranks_on_node(n))
        return sorted(node_times, key=lambda t: node_times[t])

    def __find_high_outliers(self, data):
        """
        Finds data points that are some percentage (given by self.__threshold_pct)
        higher than the mean of the data.
        """
        avg = np.mean(data)
        threshold = avg * (1.0 + self.__threshold_pct)
        outliers = [elt for elt in data if elt > threshold]
        diffs = [t / avg for t in outliers]
        assert len(outliers) == len(diffs) # sanity check
        return outliers, diffs


    ###########################################################################
    ## Primary analytical functions

    def __analyze_across_ranks(self):
        """
        Compares the total execution time across all ranks to
        find any slow (self.__threshold_pct slower than the mean) ranks.
        """
        rank_ids, total_times = zip(*self.__rank_times.items())
        outliers, slowdowns = self.__find_high_outliers(total_times)

        self.__plot_data(rank_ids, total_times, "Across-Rank Comparison", "Rank ID", outliers)

        for r_id, time in self.__rank_times.items():
            if time in outliers:
                self.__slow_ranks[r_id] = time
                self.__slow_rank_slowdowns[r_id] = slowdowns[outliers.index(time)]

        for r_id in self.__slow_ranks.keys():
            node_name = self.__rank_to_node_map[r_id]
            if self.__is_slow_node(node_name) and node_name not in self.__slow_node_names:
                self.__slow_node_names.append(node_name)

    def __analyze_within_ranks(self):
        """
        Compares the execution of each iteration on a single rank to
        find any slow (self.__threshold_pct slower than the mean) iterations.
        """
        for rank_id, breakdown in self.__rank_breakdowns.items():
            outliers, _ = self.__find_high_outliers(breakdown)
            n_iterations = len(breakdown)
            iters = list(range(n_iterations))

            if self.__plot_rank_breakdowns:
                self.__plot_data(
                    iters, breakdown,
                    f"Rank {rank_id} Breakdown", "Iteration",
                    outliers)

            if len(outliers) > 0:
                self.__slow_iterations[rank_id] = []

            for t in outliers:
                idx = breakdown.index(t)
                self.__slow_iterations[rank_id].append((idx,t))

    def __analyze_temperatures(self):
        """
        Identifies over-heated sockets and ranks.
        """
        all_socket_temps = [s["temperature"] for s in self.__socket_temps.values()]
        socket_outlier_temps, socket_diffs = self.__find_high_outliers(all_socket_temps)
        for socket_id, socket_data in self.__socket_temps.items():
            if (socket_temp := socket_data["temperature"]) in socket_outlier_temps:
                self.__hot_sockets.append(socket_id)
                self.__hot_sockets_diffs[socket_id] = socket_diffs[socket_outlier_temps.index(socket_temp)]
            core_temps = [c["temperature"] for c in socket_data["cores"].values()]
            core_outlier_temps, core_diffs = self.__find_high_outliers(core_temps)
            for core_id, core_data in socket_data["cores"].items():
                # This is slightly tricky because our maps go from ranks (unique) to cores (non-unique)
                # But now we have to go from core to rank, using the socket_id to do so
                if (core_temp := core_data["temperature"]) in core_outlier_temps:
                    ranks = [r for r, c in self.__rank_to_core_map.items() if c == core_id]
                    rank = ranks[socket_id] # This assumes that the ranks are in order, e.g. ranks [1, 25] are cores [1, 1] on sockets [0, 1]
                    self.__hot_ranks.append(rank)
                    self.__hot_ranks_diffs[rank] = core_diffs[core_outlier_temps.index(core_temp)]


    ###########################################################################
    ## Public functions

    def detect(self):
        """
        Main function of the SlowNodeDetector class.
        Parses the output file from the slow_node executable
        and identifies any slow ranks or iterations.

        Plots are generated in the same directory as the output
        file.
        """
        self.__parse_output()
        self.__analyze_across_ranks()
        self.__analyze_within_ranks()
        if self.__temperature_analysis_available:
            self.__analyze_temperatures()

        # Gather results
        rank_ids, total_times = zip(*self.__rank_times.items())
        slow_rank_ids = sorted(list(self.__slow_ranks.keys()), reverse=True, key=lambda r: self.__slow_rank_slowdowns[r])
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
        s = self.__get_s(slow_rank_ids)
        n = len(str(abs(int(self.__num_ranks))))
        print("\n----------------------------------------------------------")
        print("Results from Across-Rank Analysis")
        print()
        print(f"    {len(slow_rank_ids)} Outlier Rank{s} (at least {self.__threshold_pct:.0%} slower than the mean): {slow_rank_ids}")
        if len(slow_rank_ids) > 0:
            print()
            headline = f"    Slowdown % (Relative to Average), Temperature, and Node for Slow Rank{s}:" if \
                       self.__temperature_analysis_available else \
                       f"    Slowdown % (Relative to Average) and Node for Slow Rank{s}:"
            print(headline)
            for rank in slow_rank_ids:
                slowdown = self.__slow_rank_slowdowns[rank]
                node = self.__rank_to_node_map[rank]
                info = f"        {rank:>{n}}: {slowdown:.2%} ({node})"
                if self.__temperature_analysis_available:
                    c_id = self.__rank_to_core_map[rank]
                    temp = f"{self.__socket_temps[s_id]['cores'][c_id]['temperature']} C"
                    s_id = self.__rank_to_socket_map[rank]
                    info = f"        {rank:>{n}}: {slowdown:.2%}, {temp} ({node} - socket {s_id})"
                print(info)
            print()
        print(f"    Slowest Rank: {rank_ids[np.argmax(total_times)]} ({np.max(total_times)}s)")
        print(f"    Fastest Rank: {rank_ids[np.argmin(total_times)]} ({np.min(total_times)}s)")
        print(f"    Avg Time Across All Ranks: {np.mean(total_times)} s")
        print(f"    Std Dev Across All Ranks: {np.std(total_times)} s")
        print()
        if len(self.__slow_node_names) > 0:
            s = self.__get_s(self.__slow_node_names)
            print(f"    {len(self.__slow_node_names)} node{s} will be excluded from the hostfile:")
            for node_name in self.__slow_node_names:
                print(f"        {node_name} ({self.__get_n_slow_ranks_on_node(node_name)} slow ranks)")
        else:
            print(f"    No nodes had more than {int(self.__rps)} slow ranks.")
        print()

        if self.__temperature_analysis_available:
            print("----------------------------------------------------------")
            print("Results from Temperature Analysis")
            print()
            s = self.__get_s(self.__hot_sockets)
            print(f"    {len(self.__hot_sockets)} Socket{s} at least {self.__threshold_pct:.0%} hotter than the mean ({self.__avg_socket_temp} ºC)")
            for s_id in self.__hot_sockets:
                print(f"        {s_id}: {self.__socket_temps[s_id]['temperature']} C (+{self.__hot_sockets_diffs[s_id]:.0%})")
            print()
            s = self.__get_s(self.__hot_ranks)
            print(f"    {len(self.__hot_ranks)} Rank{s} at least {self.__threshold_pct:.0%} hotter than the mean.")
            for r_id in self.__hot_ranks:
                c_id = self.__rank_to_core_map[r_id]
                s_id = self.__rank_to_socket_map[r_id]
                print(f"        {r_id:>{n}} (socket {s_id}): {self.__socket_temps[s_id]['cores'][c_id]['temperature']} C (+{self.__hot_ranks_diffs[r_id]:.0%})")
            print()

        s = self.__get_s(ranks_with_outlying_iterations)
        print("----------------------------------------------------------")
        print("Results from Intra-Rank Analysis")
        print()
        print(f"    {len(ranks_with_outlying_iterations)} Rank{s} With Outlying Iterations: {ranks_with_outlying_iterations}")
        print(f"    Slowest Iteration: {slowest_iteration} on Rank {rank_with_slowest_iteration} ({self.__rank_to_node_map[rank_with_slowest_iteration]}) - {slowest_time}s")
        print()

        print(f"View generated plots in {self.__plots_dir}.\n")

    def create_hostfile(self):
        """
        Outputs a hostfile that contains a list of all nodes, omitting
        any slow nodes.
        """
        good_node_names = set([
            node_name for node_name in self.__rank_to_node_map.values()
            if node_name not in self.__slow_node_names
        ])

        # If num_nodes was provided, only add that many to the hostfile
        if self.__num_nodes is not None:
            num_good_nodes = len(good_node_names)
            if num_good_nodes < self.__num_nodes:
                print(f"WARNING: SlowNodeDetector will only include {num_good_nodes} nodes "
                      f"in the hostfile, but the user requested {self.__num_nodes}.")
            elif num_good_nodes > self.__num_nodes:
                n_nodes_to_drop = num_good_nodes - self.__num_nodes
                assert n_nodes_to_drop > 0, f"Cannot drop {n_nodes_to_drop}"
                sorted_nodes = self.__sort_nodes_by_execution_time(good_node_names)
                print(
                    f"Since the SlowNodeDetector originally found {num_good_nodes} good nodes, "
                    f"but only {self.__num_nodes} are needed, the following nodes will also be "
                    f"omitted from the hostfile:")
                for node in sorted_nodes[-n_nodes_to_drop:]:
                    print(f"    {node} ({self.__get_n_slow_ranks_on_node(node)} slow ranks)")
                good_node_names = sorted_nodes[:-n_nodes_to_drop]

        hostfile_path = os.path.join(self.__output_dir, "hostfile.txt")
        with open(hostfile_path, "w") as hostfile:
            for node_name in good_node_names:
                hostfile.write(node_name + "\n")

        s = 's' if len(good_node_names) != 1 else ''
        print()
        print(f"hostfile with {len(good_node_names)} node{s} has been written to {hostfile_path}")
        print("----------------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description='Slow Rank Detector script.')
    parser.add_argument('-f', '--filepath', help='Absolute or relative path to the output file from running slow_node executable', required=True)
    parser.add_argument('-N', '--num_nodes', help='The number of nodes required by the application', default=None)
    parser.add_argument('-t', '--threshold', help='Percentage above average time that indicates a "slow" rank', default=0.05)
    parser.add_argument('-spn', '--spn', help='Number of sockets per node', default=2)
    parser.add_argument('-rpn', '--rpn', help='Number of ranks per node', default=48)
    parser.add_argument('-p', '--plot_all_ranks', action='store_true', help='Plot the breakdowns for every rank')
    args = parser.parse_args()

    filepath = args.filepath if os.path.isabs(args.filepath) else os.path.join(os.getcwd(), args.filepath)

    slowNodeDetector = SlowNodeDetector(
        datafile=filepath,
        num_nodes=args.num_nodes,
        threshold_percentage=args.threshold,
        sockets_per_node=args.spn,
        ranks_per_node=args.rpn,
        plot_rank_breakdowns=args.plot_all_ranks)

    slowNodeDetector.detect()
    slowNodeDetector.create_hostfile()

    print("Done.")

if __name__ == "__main__":
    main()
