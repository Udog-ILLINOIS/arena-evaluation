#!/usr/bin/env python3
"""Issues 33 and 34: collisions between rows (scan_min.csv) and pedestrian velocity."""

import csv
import importlib.machinery
import importlib.util
import math
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import arena_evaluation.metrics as metrics  # noqa: E402
import arena_evaluation.metrics_static as metrics_static  # noqa: E402
from arena_evaluation.utils import Pedestrian, Utils  # noqa: E402
from test_metrics_episode_results import Folder, MetricsTestCase, _line  # noqa: E402
from test_recorder_duplicate_folders import (  # noqa: E402
    FakeSubscriber, NAMESPACE, RECORDER_PATH, _stub_modules,
)

v = types.SimpleNamespace


def _agent(vx, vy, fx, fy):
    return v(
        id="1", type="adult", social_state="walking",
        pose=v(position=v(x=1.0, y=2.0)),
        twist=v(linear=v(x=vx, y=vy)),
        forces=v(force=v(x=fx, y=fy)),
        destination=v(x=5.0, y=5.0),
    )


class RecorderTest(unittest.TestCase):

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()
        with open(os.path.join(self.base_dir, "data_recorder_config.yaml"), "w") as f:
            f.write("record_frequency: 400\n")

        FakeSubscriber.instances = []
        stubs = _stub_modules(self.base_dir)
        stubs["arena_evaluation.utils"].Pedestrian = Pedestrian

        self.modules = mock.patch.dict(sys.modules, stubs)
        self.modules.start()

        loader = importlib.machinery.SourceFileLoader("data_recorder_node_under_test", RECORDER_PATH)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        self.node = importlib.util.module_from_spec(spec)
        loader.exec_module(self.node)

        self.now = 0

    def tearDown(self):
        self.modules.stop()
        shutil.rmtree(self.base_dir)

    def _publish(self, topic, msg):
        for sub in list(FakeSubscriber.instances):
            if sub.registered and sub.topic == topic:
                sub.callback(msg)

    def _tick(self):
        self.now += 400 * 10**6
        self._publish("/clock", v(clock=v(secs=self.now // 10**9, nsecs=self.now % 10**9)))

    def _scan(self, ranges):
        self._publish(NAMESPACE + "scan", v(ranges=ranges, range_max=10.0))

    def _rows(self, name):
        with open(os.path.join(self.base_dir, "data", "run", name + ".csv"), newline="") as f:
            return list(csv.reader(f))

    def test_scan_min_between_rows(self):
        self.node.Recorder("run")
        self._tick()  # the first clock tick only sets the reference time
        self._scan([1.0, 2.0])
        self._tick()  # first row
        self._scan([1.0, 0.15])  # contact between rows
        self._scan([1.0, float("nan")])  # nan counts as range_max
        self._scan([1.0, 2.0])
        self._tick()
        self._tick()  # no scan since the previous row

        rows = self._rows("scan_min")
        self.assertEqual(rows[0], ["time", "data"])
        self.assertEqual([r[1] for r in rows[1:]], ["1.0", "0.15", ""])
        # scan.csv still holds the last scan only
        self.assertEqual(self._rows("scan")[2][1], "[1.0, 2.0]")
        # same times as the other files
        self.assertEqual([r[0] for r in rows[1:]], [r[0] for r in self._rows("scan")[1:]])

    def test_pedestrian_heading_and_velocity(self):
        self.node.Recorder("run")
        self._publish("/pedsim_simulator/pedsim_agents_data", v(agent_states=[
            _agent(0.0, 1.0, 1.0, 0.0),  # walking north, pushed east
            _agent(0.0, 0.0, 1.0, 0.0),  # standing, pushed east
        ]))
        self._tick()
        self._tick()

        peds = Utils.parse_pedsim(self._rows("pedsim_agents_data")[1][1])
        self.assertAlmostEqual(peds[0].theta, math.pi / 2)
        self.assertEqual(peds[0].velocity, [0.0, 1.0])
        self.assertAlmostEqual(peds[1].theta, 0.0)
        self.assertEqual(peds[1].velocity, [0.0, 0.0])


class PedestrianParseTest(unittest.TestCase):

    def test_old_entries_without_velocity(self):
        peds = Utils.parse_pedsim(
            "[{'id': '1', 'type': 'adult', 'social_state': 'walking', "
            "'position': [1.0, 2.0], 'theta': 0.5, 'destination': [5.0, 5.0]}]"
        )
        self.assertEqual(peds[0].theta, 0.5)
        self.assertIsNone(peds[0].velocity)


class ScanMinMetricsTest(MetricsTestCase):

    def _write_scan_min(self, folder, values):
        with open(os.path.join(self.dir, "scan_min.csv"), "w") as f:
            f.write("time,data\n")
            for (t, *_), value in zip(folder.rows, values):
                f.write(f"{t},{value}\n")

    def test_contact_between_rows_is_counted(self):
        folder = Folder(self.dir, {})
        # every recorded scan is clear; the contact happened between rows 2 and 3
        folder.episode(0, _line(0, 4.5, 6), goal=(5, 0, 0))
        folder.write()
        self._write_scan_min(folder, ["1.0", "1.0", "0.15", "", "1.0", "1.0"])

        for name, episodes in self.analyze(folder).items():
            self.assertEqual(episodes[0]["collision_amount"], 1, name)
            # the row without a scan keeps the contact state
            self.assertEqual(episodes[0]["collisions"], [2, 3], name)
            self.assertEqual(episodes[0]["result"], "COLLISION", name)

    def test_without_scan_min_the_scan_is_used(self):
        folder = Folder(self.dir, {})
        folder.episode(0, _line(0, 4.5, 6), goal=(5, 0, 0),
                       scans=[(1.0, 2.0)] * 2 + [(0.1, 2.0)] + [(1.0, 2.0)] * 3)

        for name, episodes in self.analyze(folder).items():
            self.assertEqual(episodes[0]["collision_amount"], 1, name)
            self.assertEqual(episodes[0]["collisions"], [2], name)

    def analyze(self, folder, **kwargs):
        # the first test writes the folder itself, then adds scan_min.csv
        if not os.path.exists(os.path.join(self.dir, "params.yaml")):
            folder.write(**kwargs)
        results = {}
        for module in (metrics, metrics_static):
            with mock.patch.object(module.Metrics, "_get_robot_params", return_value={"robot_radius": 0.2}):
                results[module.__name__] = module.Metrics(self.dir)._episode_data
        return results


if __name__ == "__main__":
    unittest.main()
