import os
import unittest
from unittest.mock import patch

from detection.detect_slow_nodes import SlowNodeDetector

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Determine file paths
        self.logs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "logs")
        self.filepath = os.path.join(self.logs_dir, "slownode.log")
        self.sensors_file = os.path.join(self.logs_dir, "sensors.log")

        # Define initialization variables
        self.rpn = 12
        self.spn = 1

        # Determine expected values
        self.expected_slow_ranks = [1, 2, 4, 6, 7, 9, 10, 11]
        self.expected_slow_nodes = []
        self.expected_overheated_nodes = ["ozark"]

        # Instantiate detector
        self.detector = SlowNodeDetector(
            path=self.filepath,
            sensors=self.sensors_file,
            num_nodes=2,
            pct=0.05,
            spn=self.spn,
            rpn=self.rpn,
            plot_rank_breakdowns=False,
        )

        # Run detection
        self.detector.detect(print_results=False)

    def test_slow_ranks(self):
        slow_ranks = self.detector.getSlowRanks()
        self.assertListEqual(list(slow_ranks.keys()), self.expected_slow_ranks)

    def test_slow_nodes(self):
        slow_nodes = self.detector.getSlowNodes()
        self.assertListEqual(slow_nodes, self.expected_slow_nodes)

    def test_overheated_nodes(self):
        overheated_nodes = self.detector.getOverheatedNodes()
        self.assertListEqual(list(overheated_nodes.keys()), self.expected_overheated_nodes)
