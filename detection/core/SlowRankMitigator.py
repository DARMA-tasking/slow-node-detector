import os
import numpy as np
import matplotlib.pyplot as plt

from detection.utils.Parse import parseSensors, parseOutput
from detection.utils.Plot import plotData, plotDroppedNodes
from detection.utils.Time import timeFtn


class SlowRankMitigator:
    """
    The SlowRankMitigator analyzes the output from the `slow_node` executable
    and outputs relevant information related to the processing speed and temperature
    of the ranks used during execution.

    There are two main methods of the SlowRankMitigator:

        detect(): This will print out information regarding slow and/or over-heated
            ranks, along with the sockets and nodes they reside on.

        createHostfile(): This will generate a `hostfile.txt` with all "good" nodes.
            This file can be used in future jobs to ensure that slow nodes are
            avoided. Importantly, nodes are only omitted from the hostfile if
            the number of slow ranks on that node surpasses the size of a socket.

            Optional: Use `-N` argument to specify the number of nodes that should be
            included in the hostfile.

    The following terminology will be used through the SlowRankMitigator:

        Rank: An MPI process
        Core: Processing unit on a socket
        Socket: Collection of cores on a node
        Node: Computing unit in a cluster
    """

    def __init__(
            self, path, sensors, num_nodes, pct, weight, benchmark, type, spn, rpn, plot_rank_breakdowns):
        # Create empty dicts for storing data
        self.__rank_times = {}
        self.__rank_breakdowns = {}
        self.__rank_to_node_map = {} # Maps each rank to the name of its corresponding node
        self.__rank_info = {}

        # Initialize variables
        self.__filepath = path
        self.__num_nodes = int(num_nodes) if num_nodes is not None else None
        self.__threshold_pct = float(pct)
        self.__weight = float(weight)
        self.__benchmark = benchmark
        self.__datatype = type
        self.__spn = int(spn)
        self.__rpn = int(rpn)
        self.__rps = self.__rpn / self.__spn
        self.__plot_rank_breakdowns = plot_rank_breakdowns
        self.__num_ranks = 0

        # Initialize outliers
        self.__slow_ranks = {}
        self.__slow_rank_slowdowns = {}
        self.__slow_node_names = []

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

    ###########################################################################
    ## Public getters

    def getSlowRanks(self) -> dict:
        """Return map of slow rank IDs to their times."""
        return self.__slow_ranks

    def getSlowNodes(self) -> list:
        """Return list of slow node names."""
        return self.__slow_node_names

    ###########################################################################
    ## Public functions

    def detect(self, print_results=True):
        """
        Main function of the SlowRankMitigator class.
        Parses the output file from the slow_node executable
        and identifies any slow ranks or iterations.

        Plots are generated in the same directory as the output
        file.
        """
        timeFtn(self.__parseOutput)
        timeFtn(self.__analyzeAcrossRanks)

        # Gather results
        rank_ids, total_times = zip(*self.__rank_times.items())
        slow_rank_ids = sorted(list(self.__slow_ranks.keys()), reverse=True, key=lambda r: self.__slow_rank_slowdowns[r])

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

            print(f"View generated plots in {self.__plots_dir}.")
            print("----------------------------------------------------------")
            print()

    def createAlphafile(self):
        alphafile_path = os.path.join(self.__output_dir, "alphafile.dat")
        with open(alphafile_path, "w") as alphafile:
            for rank_id, rank_info in self.__rank_info.items():
                if rank_id in self.__slow_ranks:
                    alpha = self.__slow_rank_slowdowns[rank_id] * self.__weight
                else:
                    alpha = 1.0
                alphafile.write(f"{rank_info[0]} {rank_info[1]} {alpha}\n")
        print("Alpha map has been written to alphafile.dat")
