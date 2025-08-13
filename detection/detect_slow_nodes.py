import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.core.SlowNodeDetector import SlowNodeDetector
from detection.utils.Time import timeFtn

def main():
    """
    See documentation of SlowNodeDetector class, as well as
    the detect() and createHostfile() methods, for more information.
    """
    parser = argparse.ArgumentParser(description='Slow Rank Detector script.')
    parser.add_argument('-f', '--filepath', help='Absolute or relative path to the output file from running slow_node executable', required=True)
    parser.add_argument('-s', '--sensors', help='Absolute or relative path to the sensors file that will be analyzed', default=None)
    parser.add_argument('-N', '--num_nodes', help='The number of nodes required by the application', default=None)
    parser.add_argument('-t', '--threshold', help='Percentage above average time that indicates a "slow" rank', default=0.05)
    parser.add_argument('-m', '--mean', action='store_true', help='Eliminate nodes faster than the mean as well')
    parser.add_argument('-b', '--benchmark', help='Benchmark to analyze: [level1, level2, level3, dpotrf]', default='level3')
    parser.add_argument('-d', '--datatype', help='Datatype of benchmark to analyze: [double, complex]', default='double')
    parser.add_argument('-spn', '--spn', help='Number of sockets per node', default=2)
    parser.add_argument('-rpn', '--rpn', help='Number of ranks per node', default=48)
    parser.add_argument('-p', '--plot_all_ranks', action='store_true', help='Plot the breakdowns for every rank')
    args = parser.parse_args()

    filepath = os.path.abspath(args.filepath)
    sensors_filepath = os.path.abspath(args.sensors) if args.sensors is not None else None

    slowNodeDetector = SlowNodeDetector(
        path=filepath,
        sensors=sensors_filepath,
        num_nodes=args.num_nodes,
        pct=args.threshold,
        target_mean=args.mean,
        benchmark=args.benchmark,
        type=args.datatype,
        spn=args.spn,
        rpn=args.rpn,
        plot_rank_breakdowns=args.plot_all_ranks)

    timeFtn(slowNodeDetector.detect)
    timeFtn(slowNodeDetector.createHostfile)

if __name__ == "__main__":
    timeFtn(main)
