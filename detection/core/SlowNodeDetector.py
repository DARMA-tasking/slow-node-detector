import os
import numpy as np
import matplotlib.pyplot as plt

from detection.utils.Parse import parseSensors, parseOutput
from detection.utils.Plot import plotData, plotDroppedNodes
from detection.utils.Time import timeFtn


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
            self, path, sensors, num_nodes, pct, target_mean, benchmark, type, spn, rpn, plot_rank_breakdowns):
        # Create empty dicts for storing data
        self.__rank_times = {}
        self.__rank_breakdowns = {}
        self.__rank_to_node_map = {} # Maps each rank to the name of its corresponding node
        self.__node_temps = {}
        self.__node_freqs = {}
        self.__overheated_nodes = {}

        # Initialize variables
        self.__filepath = path
        self.__sensors_output_file = sensors
        self.__num_nodes = int(num_nodes) if num_nodes is not None else None
        self.__threshold_pct = float(pct)
        self.__target_mean = target_mean
        self.__benchmark = benchmark
        self.__datatype = type
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


    ###########################################################################
    ## Parsing

    def __parseOutput(self):
        """Parses text output from slow_node.cc"""
        self.__rank_times,      \
        self.__rank_breakdowns, \
        self.__rank_to_node_map, \
        self.__rank_info = parseOutput(self.__filepath, self.__benchmark, self.__datatype)

        self.__num_ranks = len(self.__rank_times)

    def __parseSensors(self):
        """
        Iterates through the sensors directory (given with -s on the command line)
        and identifies the temperature of each rank on that node.
        """
        self.__node_temps, self.__node_freqs = parseSensors(self.__sensors_output_file)

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

    def __sortNodesByMaxRankExecutionTime(self, nodes: list):
        """
        Takes in a list of node names and sorts them based on total execution time.
        The fastest nodes will be first, and the slowest will be last.
        """
        node_times = {}
        for r, n in self.__rank_to_node_map.items():
            if n in nodes:
                if n not in node_times:
                    node_times[n] = 0.0
                if self.__rank_times[r] > node_times[n]:
                    node_times[n] = self.__rank_times[r]
        # Alternative:
        # return sorted(nodes, key=lambda n: self.__getNumberOfSlowRanksOnNode(n))
        return sorted(node_times, key=lambda t: node_times[t])

    def __sortNodesByNodeDevFromAvgExecutionTime(self, nodes: list):
        """
        Takes in a list of node names and sorts them based on how much they deviate
        from the average total execution time.
        """
        node_times = {}
        for r, n in self.__rank_to_node_map.items():
            if n in nodes:
                if n not in node_times:
                    node_times[n] = 0.0
                node_times[n] += self.__rank_times[r]
        avg = np.mean(list(node_times.values()))
        return sorted(node_times, key=lambda t: abs(node_times[t]-avg))

    def __sortNodesByRankDevFromAvgExecutionTime(self, nodes: list):
        """
        Takes in a list of node names and sorts them based on how much they deviate
        from the average total execution time.
        """
        avg = np.mean(list(self.__rank_times.values()))
        node_dev_times = {}
        for r, n in self.__rank_to_node_map.items():
            if n in nodes:
                if n not in node_dev_times:
                    node_dev_times[n] = 0.0
                this_dev_time = abs(self.__rank_times[r]-avg)
                if this_dev_time > node_dev_times[n]:
                    node_dev_times[n] = this_dev_time
        return sorted(node_dev_times, key=lambda t: node_dev_times[t])

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

    def __findHighLowOutliers(self, data):
        """
        Finds data points that are some percentage (given by self.__threshold_pct)
        higher than the mean of the data.
        """
        avg = np.mean(data)
        outliers = [elt for elt in data if elt > avg * (1.0 + self.__threshold_pct) or elt < avg * (1.0 - self.__threshold_pct)]
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
        if self.__target_mean:
            outliers, slowdowns = self.__findHighLowOutliers(total_times)
        else:
            outliers, slowdowns = self.__findHighOutliers(total_times)

        plotData(rank_ids, total_times,
                 "Across-Rank Comparison", "Rank ID",
                 self.__plots_dir, self.__threshold_pct,
                 outliers)

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
                plotData(
                    iters, breakdown,
                    f"Rank {rank_id} Breakdown", "Iteration",
                    self.__plots_dir, self.__threshold_pct,
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
        timeFtn(self.__parseOutput)
        timeFtn(self.__analyzeAcrossRanks)
        timeFtn(self.__analyzeWithinRanks)
        if self.__temperature_analysis_available:
            timeFtn(self.__analyzeTemperatures)

        # Gather results
        rank_ids, total_times = zip(*self.__rank_times.items())
        slow_rank_ids = sorted(list(self.__slow_ranks.keys()), reverse=True, key=lambda r: self.__slow_rank_slowdowns[r])
        ranks_with_outlying_iterations = list(self.__slow_iterations.keys())

        rank_with_slowest_iteration = -1
        slowest_iteration = -1
        slowest_time = -np.inf
        all_ranks_slowest_iters = {}
        if len(ranks_with_outlying_iterations) > 0:
            for r_id, slow_iters in self.__slow_iterations.items():
                slowest_iter_on_this_rank = max(slow_iters, key=lambda x: x[1])
                slowest_iter_id = slowest_iter_on_this_rank[0]
                slowest_iter_t = slowest_iter_on_this_rank[1]
                all_ranks_slowest_iters[r_id] = (slowest_iter_id, slowest_iter_t)
                slowest_time = max(slowest_time, slowest_iter_t)
                if slowest_iter_t == slowest_time:
                    slowest_iteration = slowest_iter_id
                    rank_with_slowest_iteration = r_id
        else:
            for r_id, breakdown in self.__rank_breakdowns.items():
                slowest_time = max(np.max(breakdown), slowest_time)
                if slowest_time in breakdown:
                    slowest_iteration = np.argmax(breakdown)
                    rank_with_slowest_iteration = r_id
        if len(all_ranks_slowest_iters) > 0:
            all_ranks_slowest_iters = dict(sorted(all_ranks_slowest_iters.items(), reverse=True, key=lambda item: item[1][1]))

        # Print results
        if print_results:
            s = self.__s(slow_rank_ids)
            n = len(str(abs(int(self.__num_ranks))))
            print(f"\nPrinting analysis from {self.__benchmark}_{self.__datatype} benchmark...")
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
                            freq = self.__node_freqs[n_id][s_id][c_id]
                            core_temp_outputs.append(f"        Core {c_id}: {temp} C ({diff:.0%} hotter than mean on this socket) - {n_id} (socket {s_id}); Frequency {freq} KHz")
                s = self.__s(core_temp_outputs)
                print(f"    Found {len(core_temp_outputs)} over-heated cores")
                for core_temp_output in core_temp_outputs:
                    print(core_temp_output)
                print()

            s = self.__s(ranks_with_outlying_iterations)
            print("Intra-Rank Analysis")
            print()
            print(f"    {len(ranks_with_outlying_iterations)} Rank{s} With Outlying Iterations.")
            if len(ranks_with_outlying_iterations) > 100:
                print(f"\n        100 Slowest Iterations:")
            for i, (r_id, (iter_id, iter_t)) in enumerate(all_ranks_slowest_iters.items()):
                if i == 100:
                    break
                print(f"        {iter_t} s (Iter {iter_id} on Rank {r_id}, Node {self.__rank_to_node_map[r_id]})")
            print()
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
                if self.__target_mean:
                    #sorted_nodes = self.__sortNodesByNodeDevFromAvgExecutionTime(good_node_names)
                    #sorted_nodes = self.__sortNodesByRankDevFromAvgExecutionTime(good_node_names)
                    sorted_nodes = self.__sortNodesByMaxRankExecutionTime(good_node_names)
                else:
                    #sorted_nodes = self.__sortNodesByExecutionTime(good_node_names)
                    sorted_nodes = self.__sortNodesByMaxRankExecutionTime(good_node_names)
                print(
                    f"Since the SlowNodeDetector originally found {num_good_nodes} good node{s}, "
                    f"but only {self.__num_nodes} are needed, the following nodes will also be "
                    f"omitted from the hostfile:")
                for node in sorted_nodes[-n_nodes_to_drop:]:
                    self.__slow_node_names.append(node)
                    print(f"    {node} ({self.__getNumberOfSlowRanksOnNode(node)} slow ranks)")
                good_node_names = sorted_nodes[:-n_nodes_to_drop]

        hostfile_path = os.path.join(self.__output_dir, "hostfile.txt")
        with open(hostfile_path, "w") as hostfile:
            for node_name in good_node_names:
                hostfile.write(node_name + "\n")

        s = self.__s(good_node_names)
        print(f"hostfile with {len(good_node_names)} node{s} has been written to {hostfile_path}")

        plotDroppedNodes(
            self.__rank_times,
            self.__rank_to_node_map,
            self.__slow_node_names,
            self.__plots_dir
        )

        print(f"Plot with all dropped nodes has been written to {self.__output_dir}\n")
