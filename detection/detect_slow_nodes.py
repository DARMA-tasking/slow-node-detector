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

        createHostfile(): This will generate a `hostfile.txt` with all "good" nodes.
            This file can be used in future jobs to ensure that slow nodes are
            avoided. Importantly, nodes are only omitted from the hostfile if
            the number of slow ranks on that node surpasses the size of a socket.

            Optional: Use `-N` argument to specify the number of nodes that should be
            included in the hostfile.

    The following terminology will be used through the SlowNodeDetector:

        Rank: An MPI process
        Core: Processing unit on a socket
        Socket: Collection of cores on a node
        Node: Computing unit in a cluster
    """

    def __init__(
            self, path, sensors, num_nodes, pct, spn, rpn, plot_rank_breakdowns):
        # Create empty dicts for storing data
        self.__rank_times = {}
        self.__rank_breakdowns = {}
        self.__rank_to_node_map = {} # Maps each rank to the name of its corresponding node
        self.__node_id_to_node_name_map = {}
        self.__node_temps = {}
        self.__overheated_nodes = {}

        # Initialize variables
        self.__filepath = path
        self.__sensors_output_file = sensors
        self.__num_nodes = int(num_nodes) if num_nodes is not None else None
        self.__threshold_pct = float(pct)
        self.__spn = int(spn)
        self.__rpn = int(rpn)
        self.__rps = self.__rpn / self.__spn
        self.__temperature_analysis_available = True if self.__sensors_output_file is not None else False
        self.__plot_rank_breakdowns = plot_rank_breakdowns
        self.__num_ranks = 0

        # Initialize outliers
        self.__slow_ranks = {}
        self.__slow_rank_slowdowns = {}
        self.__slow_node_names = []
        self.__slow_iterations = {}

        # Initialize (and create) directories
        self.__output_dir = os.path.join(
            os.path.dirname(path),
            "output")
        self.__plots_dir = os.path.join(
            self.__output_dir,
            "plots")
        os.makedirs(self.__plots_dir, exist_ok=True)


    ###########################################################################
    ## Utilities

    def __s(self, lst: list):
        """Helper function for the print statements."""
        return "s" if len(lst) != 1 else ""

    def __matchRegex(self, pattern: str, line: str):
        """Helper function for matching regex expressions."""
        match = re.search(pattern, line)
        if match:
            return tuple(match.groups())
        raise RuntimeError(f"regex matching failed on line {line}")

    def __plotData(self, x_data, y_data, title, xlabel, highlights=[]):
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


    ###########################################################################
    ## Parsing

    def __parseOutput(self):
        """Parses text output from slow_node.cc"""
        is_parsing_sensors, is_parsing_mapping, is_looking_for_core = False, False, False
        current_socket, current_sensor_core = -1, -1
        current_rank, current_core = -1, -1
        with open(self.__filepath, "r") as output:
            for line in output:

                if line.startswith("Node"):
                    node_pattern = r"Node (\d+): (.+)"
                    node_id_str, node_name = self.__matchRegex(node_pattern, line)
                    self.__node_id_to_node_name_map[int(node_id_str)] = node_name

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

            self.__num_ranks = len(self.__rank_times)

    def __parseSensors(self):
        """
        Iterates through the sensors directory (given with -s on the command line) and identifies the maximum
        temperature of each rank on that node.
        """

        with open(self.__sensors_output_file, 'r') as sensor_data:
            for line in sensor_data:
                if line.startswith("Node"):
                    node_id = int(line.split(":")[-1].strip())
                    assert node_id in self.__node_id_to_node_name_map, f"Unrecognized node ID: {node_id}"
                    node_name = self.__node_id_to_node_name_map[node_id]
                    if node_name not in self.__node_temps:
                        self.__node_temps = {node_name: {}}
                elif line.startswith("Socket"):
                    pattern = r"Socket id (\d+), Core (\d+): (\d+)(?:°C| C)"
                    socket_str, core_str, temp_str = self.__matchRegex(pattern, line)
                    socket_id = int(socket_str)
                    core_id = int(core_str)
                    temp = float(temp_str)
                    if socket_id not in self.__node_temps[node_name]:
                        self.__node_temps[node_name][socket_id] = {}
                    self.__node_temps[node_name][socket_id][core_id] = temp


    ###########################################################################
    ## Secondary analytical functions

    def __getNumberOfSlowRanksOnNode(self, node_name):
        """
        Returns the number of ranks in self.__slow_ranks that
        belong to the given node.
        """
        return sum(1 for r_id in self.__slow_ranks if self.__rank_to_node_map[r_id] == node_name)

    def __isSlowNode(self, node_name):
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
        n_slow_ranks = self.__getNumberOfSlowRanksOnNode(node_name)

        return n_slow_ranks >= self.__rps

    def __sortNodesByExecutionTime(self, nodes: list):
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
        # return sorted(nodes, key=lambda n: self.__getNumberOfSlowRanksOnNode(n))
        return sorted(node_times, key=lambda t: node_times[t])

    def __findHighOutliers(self, data):
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

    def __analyzeAcrossRanks(self):
        """
        Compares the total execution time across all ranks to
        find any slow (self.__threshold_pct slower than the mean) ranks.
        """
        rank_ids, total_times = zip(*self.__rank_times.items())
        outliers, slowdowns = self.__findHighOutliers(total_times)

        self.__plotData(rank_ids, total_times, "Across-Rank Comparison", "Rank ID", outliers)

        for r_id, time in self.__rank_times.items():
            if time in outliers:
                self.__slow_ranks[r_id] = time
                self.__slow_rank_slowdowns[r_id] = slowdowns[outliers.index(time)]

        for r_id in self.__slow_ranks.keys():
            node_name = self.__rank_to_node_map[r_id]
            if self.__isSlowNode(node_name) and node_name not in self.__slow_node_names:
                self.__slow_node_names.append(node_name)

    def __analyzeWithinRanks(self):
        """
        Compares the execution of each iteration on a single rank to
        find any slow (self.__threshold_pct slower than the mean) iterations.
        """
        for rank_id, breakdown in self.__rank_breakdowns.items():
            outliers, _ = self.__findHighOutliers(breakdown)
            n_iterations = len(breakdown)
            iters = list(range(n_iterations))

            if self.__plot_rank_breakdowns:
                self.__plotData(
                    iters, breakdown,
                    f"Rank {rank_id} Breakdown", "Iteration",
                    outliers)

            if len(outliers) > 0:
                self.__slow_iterations[rank_id] = []

            for t in outliers:
                idx = breakdown.index(t)
                self.__slow_iterations[rank_id].append((idx,t))

    def __analyzeTemperatures(self):
        """
        Identifies over-heated sockets and ranks.
        """
        self.__parseSensors()
        for n_id, node_data in self.__node_temps.items():
            for s_id, socket_data in node_data.items():
                outliers, diffs = self.__findHighOutliers(list(socket_data.values()))
                i = 0
                for c_id, core_temp in socket_data.items():
                    if core_temp in outliers:
                        if n_id not in self.__overheated_nodes:
                            self.__overheated_nodes[n_id] = {}
                        if s_id not in self.__overheated_nodes[n_id]:
                            self.__overheated_nodes[n_id][s_id] = {}
                        self.__overheated_nodes[n_id][s_id][c_id] = {
                            "temperature": core_temp,
                            "diff": diffs[i]
                        }
                        i += 1


    ###########################################################################
    ## Public getters

    def getSlowRanks(self) -> dict:
        """Return map of slow rank IDs to their times."""
        return self.__slow_ranks

    def getSlowNodes(self) -> list:
        """Return list of slow node names."""
        return self.__slow_node_names

    def getOverheatedNodes(self) -> dict:
        """Return map of slow node names to the sockets and cores on each node."""
        return self.__overheated_nodes


    ###########################################################################
    ## Public functions

    def detect(self, print_results=True):
        """
        Main function of the SlowNodeDetector class.
        Parses the output file from the slow_node executable
        and identifies any slow ranks or iterations.

        Plots are generated in the same directory as the output
        file.
        """
        self.__parseOutput()
        self.__analyzeAcrossRanks()
        self.__analyzeWithinRanks()
        if self.__temperature_analysis_available:
            self.__analyzeTemperatures()

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
        if print_results:
            s = self.__s(slow_rank_ids)
            n = len(str(abs(int(self.__num_ranks))))
            print("\n----------------------------------------------------------")
            print("Across-Rank Analysis")
            print()
            print(f"    {len(slow_rank_ids)} Outlier Rank{s} (at least {self.__threshold_pct:.0%} slower than the mean): {slow_rank_ids}")
            if len(slow_rank_ids) > 0:
                print()
                print(f"    Slowdown % (Relative to Average) and Node for Slow Rank{s}:")
                for rank in slow_rank_ids:
                    slowdown = self.__slow_rank_slowdowns[rank]
                    node = self.__rank_to_node_map[rank]
                    print(f"        {rank:>{n}}: {slowdown:.2%} ({node})")
                print()
            print(f"    Slowest Rank: {rank_ids[np.argmax(total_times)]} ({np.max(total_times)}s)")
            print(f"    Fastest Rank: {rank_ids[np.argmin(total_times)]} ({np.min(total_times)}s)")
            print(f"    Avg Time Across All Ranks: {np.mean(total_times)} s")
            print(f"    Std Dev Across All Ranks: {np.std(total_times)} s")
            print()
            if len(self.__slow_node_names) > 0:
                s = self.__s(self.__slow_node_names)
                print(f"    {len(self.__slow_node_names)} node{s} will be excluded from the hostfile:")
                for node_name in self.__slow_node_names:
                    print(f"        {node_name} ({self.__getNumberOfSlowRanksOnNode(node_name)} slow ranks)")
            else:
                print(f"    No nodes had more than {int(self.__rps)} slow ranks.")
            print()

            if self.__temperature_analysis_available:
                print("Temperature Analysis")
                print()
                core_temp_outputs = []
                for n_id, n_data in self.__overheated_nodes.items():
                    for s_id, s_data in n_data.items():
                        for c_id, c_data in s_data.items():
                            diff = c_data["diff"]
                            temp = c_data["temperature"]
                            core_temp_outputs.append(f"        Core {c_id}: {temp} C ({diff:.0%} hotter than mean on this socket) - {n_id} (socket {s_id})")
                s = self.__s(core_temp_outputs)
                print(f"    Found {len(core_temp_outputs)} over-heated cores")
                for core_temp_output in core_temp_outputs:
                    print(core_temp_output)
                print()

            s = self.__s(ranks_with_outlying_iterations)
            print("Intra-Rank Analysis")
            print()
            print(f"    {len(ranks_with_outlying_iterations)} Rank{s} With Outlying Iterations: {ranks_with_outlying_iterations}")
            print(f"    Slowest Iteration: {slowest_iteration} on Rank {rank_with_slowest_iteration} ({self.__rank_to_node_map[rank_with_slowest_iteration]}) - {slowest_time}s")
            print()

            print(f"View generated plots in {self.__plots_dir}.")
            print("----------------------------------------------------------")
            print()

    def createHostfile(self):
        """
        Outputs a hostfile that contains a list of all nodes, omitting
        any slow nodes.

        If the -N argument was passed, exactly N nodes will be written
        in the hostfile, assuming that there are least N "good" nodes.

        If there were more than N "good" nodes, then we sort the nodes
        by total execution time and only include the N fastest.

        If there are fewer than N "good" nodes, then we write that all
        "good" nodes to the file and output a warning that not enough
        nodes were found. If this hostfile is used in a run with the
        full N number of nodes specified, mpiexec will throw an error.
        """
        good_node_names = set([
            node_name for node_name in self.__rank_to_node_map.values()
            if node_name not in self.__slow_node_names
        ])

        # If num_nodes was provided, only add that many nodes to the hostfile
        if self.__num_nodes is not None:
            num_good_nodes = len(good_node_names)
            s = self.__s(good_node_names)
            if num_good_nodes < self.__num_nodes:
                print(f"WARNING: SlowNodeDetector will only include {num_good_nodes} node{s} "
                      f"in the hostfile, but the user requested {self.__num_nodes}.")
            elif num_good_nodes > self.__num_nodes:
                n_nodes_to_drop = num_good_nodes - self.__num_nodes
                assert n_nodes_to_drop > 0, f"Cannot drop {n_nodes_to_drop}"
                sorted_nodes = self.__sortNodesByExecutionTime(good_node_names)
                print(
                    f"Since the SlowNodeDetector originally found {num_good_nodes} good node{s}, "
                    f"but only {self.__num_nodes} are needed, the following nodes will also be "
                    f"omitted from the hostfile:")
                for node in sorted_nodes[-n_nodes_to_drop:]:
                    print(f"    {node} ({self.__getNumberOfSlowRanksOnNode(node)} slow ranks)")
                good_node_names = sorted_nodes[:-n_nodes_to_drop]

        hostfile_path = os.path.join(self.__output_dir, "hostfile.txt")
        with open(hostfile_path, "w") as hostfile:
            for node_name in good_node_names:
                hostfile.write(node_name + "\n")

        s = self.__s(good_node_names)
        print(f"hostfile with {len(good_node_names)} node{s} has been written to {hostfile_path}\n")

def getFilepath(path: str):
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)

def main():
    """
    See documentation of SlowNodeDetector class, as well as
    the detect() and createHostfile() methods, for more information.
    """
    parser = argparse.ArgumentParser(description='Slow Rank Detector script.')
    parser.add_argument('-f', '--filepath', help='Absolute or relative path to the output file from running slow_node executable', required=True)
    parser.add_argument('-s', '--sensors', help='Absolute or relative path to the sensors.log file', default=None)
    parser.add_argument('-N', '--num_nodes', help='The number of nodes required by the application', default=None)
    parser.add_argument('-t', '--threshold', help='Percentage above average time that indicates a "slow" rank', default=0.05)
    parser.add_argument('-spn', '--spn', help='Number of sockets per node', default=2)
    parser.add_argument('-rpn', '--rpn', help='Number of ranks per node', default=48)
    parser.add_argument('-p', '--plot_all_ranks', action='store_true', help='Plot the breakdowns for every rank')
    args = parser.parse_args()

    filepath = getFilepath(args.filepath)
    sensors_filepath = getFilepath(args.sensors) if args.sensors is not None else None

    slowNodeDetector = SlowNodeDetector(
        path=filepath,
        sensors=sensors_filepath,
        num_nodes=args.num_nodes,
        pct=args.threshold,
        spn=args.spn,
        rpn=args.rpn,
        plot_rank_breakdowns=args.plot_all_ranks)

    slowNodeDetector.detect()
    slowNodeDetector.createHostfile()

if __name__ == "__main__":
    main()
