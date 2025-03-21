import os
import unittest
from unittest.mock import patch

from detection.detect_slow_nodes import SlowNodeDetector

class TestNoClustering(unittest.TestCase):
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
        self.expected_slow_ranks = set([1, 2, 4, 6, 7, 9, 10, 11])
        self.expected_slow_nodes = set()
        self.expected_overheated_nodes = set(["ozark"])

        # Instantiate detector
        self.detector = SlowNodeDetector(
            path=self.filepath,
            sensors=self.sensors_file,
            num_nodes=2,
            pct=0.05,
            spn=self.spn,
            rpn=self.rpn,
            plot_rank_breakdowns=False,
            use_clstr=False
        )

        # Run detection
        self.detector.detect(print_results=False)

    def test_slow_ranks(self):
        slow_ranks = self.detector.getSlowRanks()
        self.assertSetEqual(slow_ranks, self.expected_slow_ranks)

    def test_slow_nodes(self):
        slow_nodes = self.detector.getSlowNodes()
        self.assertSetEqual(slow_nodes, self.expected_slow_nodes)

    def test_overheated_nodes(self):
        overheated_nodes = self.detector.getOverheatedNodes()
        self.assertSetEqual(overheated_nodes, self.expected_overheated_nodes)


class TestClusteringOutliers(unittest.TestCase):
    def setUp(self):
        # Determine file paths
        self.logs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "logs_1920_ranks")
        self.filepath = os.path.join(self.logs_dir, "slownode.log")

        # Define initialization variables
        self.rpn = 48
        self.spn = 2

        # Determine expected values
        self.expected_slow_ranks = set([
            1702,1462,1902,1222,1182,1262,1862,1342,1382,1422,1742,1502,1102,1582,
            1142,1782,1662,1022,1062,1542,1622,982,1302,1822,1381,1501,1021,1341,
            1541,1301,1701,1821,1261,1621,1901,1012,1461,1861,1181,1221,981,1661,
            1061,1781,1421,1741,1101,1581,1141,902,1812,1252,1492,1532,1772,1292,
            1332,1852,302,382,182,702,1652,1212,582,1172,1452,1692,972,1572,1732,
            62,1412
        ])
        self.expected_slow_nodes = set([
            'node446', 'node442' #, 'node378', 'node257'
        ])
        self.expected_overheated_nodes = set()

        # Instantiate detector
        self.detector = SlowNodeDetector(
            path=self.filepath,
            sensors=None,
            num_nodes=36,
            pct=0.05,
            spn=self.spn,
            rpn=self.rpn,
            plot_rank_breakdowns=False,
            use_clstr=True
        )

        # Run detection
        self.detector.detect(print_results=False)

    def test_slow_ranks(self):
        slow_ranks = self.detector.getSlowRanks()
        print(slow_ranks)
        self.assertSetEqual(slow_ranks, self.expected_slow_ranks)

    def test_slow_nodes(self):
        slow_nodes = self.detector.getSlowNodes()
        print(slow_nodes)
        print(self.expected_slow_nodes)
        self.assertSetEqual(slow_nodes, self.expected_slow_nodes)

    def test_overheated_nodes(self):
        overheated_nodes = self.detector.getOverheatedNodes()
        self.assertSetEqual(overheated_nodes, self.expected_overheated_nodes)

class TestClusteringNoOutliers(unittest.TestCase):
    def setUp(self):
        # Determine file paths
        self.logs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "logs_1008_ranks")
        self.filepath = os.path.join(self.logs_dir, "slownode.log")

        # Define initialization variables
        self.rpn = 48
        self.spn = 2

        # Determine expected values
        self.expected_slow_ranks = set()
        self.expected_slow_nodes = set()
        self.expected_overheated_nodes = set()

        # Instantiate detector
        self.detector = SlowNodeDetector(
            path=self.filepath,
            sensors=None,
            num_nodes=19,
            pct=0.05,
            spn=self.spn,
            rpn=self.rpn,
            plot_rank_breakdowns=False,
            use_clstr=True
        )

        # Run detection
        self.detector.detect(print_results=False)

    def test_slow_ranks(self):
        slow_ranks = self.detector.getSlowRanks()
        print(slow_ranks)
        self.assertSetEqual(slow_ranks, self.expected_slow_ranks)

    def test_slow_nodes(self):
        slow_nodes = self.detector.getSlowNodes()
        print(slow_nodes)
        print(self.expected_slow_nodes)
        self.assertSetEqual(slow_nodes, self.expected_slow_nodes)

    def test_overheated_nodes(self):
        overheated_nodes = self.detector.getOverheatedNodes()
        self.assertSetEqual(overheated_nodes, self.expected_overheated_nodes)
