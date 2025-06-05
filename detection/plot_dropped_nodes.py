import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.utils.Parse import parseOutput, parseAnalysis
from detection.utils.Time import timeFtn
from detection.utils.Plot import plotDroppedNodes


"""
This functionality is included in detect_slow_nodes.py, but is included here
so that we can post-process previous runs of the detector. For new runs,
there is no need to call this script explicitly.
"""

def main():
    parser = argparse.ArgumentParser(description='Slow Rank Detector script.')
    parser.add_argument('-s', '--slownode', help='Absolute or relative path to the output from slow_node executable', required=True)
    parser.add_argument('-a', '--analysis', help='Absolute or relative path to the output from detect_slow_nodes', required=True)
    parser.add_argument('-o', '--output', help='Absolute or relative path to the output directory where the plot will be saved. Defaults to $(pwd)/output', default=None)
    parser.add_argument('-r', '--show_ranges', help='Show range for all nodes (not just the dropped ones)', action="store_true")
    parser.add_argument('-n', '--no_ranges', help='Hide ranges for all nodes (including the dropped ones). Overrides -r', action="store_true")
    args = parser.parse_args()

    slownode_filepath = os.path.abspath(args.slownode)
    analysis_filepath = os.path.abspath(args.analysis)
    output_filepath = os.path.abspath(args.output) if args.output is not None else os.path.join(os.getcwd(), "output")

    rank_times, _, rank_to_node_map = parseOutput(slownode_filepath)
    dropped_nodes = parseAnalysis(analysis_filepath)

    plotDroppedNodes(rank_times, rank_to_node_map, dropped_nodes, output_filepath, args.show_ranges, args.no_ranges)

if __name__ == "__main__":
    timeFtn(main)
