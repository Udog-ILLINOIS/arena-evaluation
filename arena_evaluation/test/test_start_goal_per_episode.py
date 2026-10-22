#!/usr/bin/env python3
"""Issue 09: start_goal.csv holds one row per episode and metrics merge it by episode id."""

import csv
import importlib.machinery
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import arena_evaluation.metrics as metrics  # noqa: E402
import arena_evaluation.metrics_static as metrics_static  # noqa: E402
from test_recorder_duplicate_folders import (  # noqa: E402
    FakeSubscriber, NAMESPACE, RECORDER_PATH, _stub_modules,
)

STARTS = {0: [1.0, 1.0, 0.0], 1: [2.0, 2.0, 0.0], 2: [3.0, 3.0, 0.0]}
GOALS = {0: [10.0, 1.0, 0.0], 1: [20.0, 2.0, 0.0], 2: [30.0, 3.0, 0.0]}


def _ns(secs, nsecs=0):
    return secs * 10**9 + nsecs


def _odom(x):
    v = types.SimpleNamespace
    return v(
        pose=v(pose=v(position=v(x=x, y=0.0, z=0.0), orientation=v(x=0.0, y=0.0, z=0.0, w=1.0))),
        twist=v(twist=v(linear=v(x=0.5, y=0.0, z=0.0), angular=v(x=0.0, y=0.0, z=0.0))),
    )


class RecorderStartGoalTest(unittest.TestCase):

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()
        with open(os.path.join(self.base_dir, "data_recorder_config.yaml"), "w") as f:
            f.write("record_frequency: 400\n")

        FakeSubscriber.instances = []
        stubs = _stub_modules(self.base_dir)
        self.params = {}
        stubs["rospy"].get_param = lambda name, default=None: self.params.get(name, default)

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

    def _ticks(self, count):
        for _ in range(count):
            self.now += 400 * 10**6
            self._publish(NAMESPACE + "scan", types.SimpleNamespace(ranges=[1.0, 2.0], range_max=10.0))
            self._publish(NAMESPACE + "odom", _odom(self.now / 10**9))
            self._publish("/clock", types.SimpleNamespace(
                clock=types.SimpleNamespace(secs=self.now // 10**9, nsecs=self.now % 10**9)))

    def _reset(self, episode):
        # order of the task generator: set the params, then publish scenario_reset
        self.params[NAMESPACE + "start"] = STARTS[episode]
        self.params[NAMESPACE + "goal"] = GOALS[episode]
        self._publish("/scenario_reset", types.SimpleNamespace(data=episode))

    def _start_goal_rows(self):
        with open(os.path.join(self.base_dir, "data", "run", "start_goal.csv"), newline="") as f:
            return list(csv.reader(f))

    def _record(self, reset_first):
        self.node.Recorder("run")
        if reset_first:
            self._reset(0)
            self._ticks(5)
        else:
            # ticks of episode 0 arrive before its scenario_reset
            self._ticks(2)
            self._reset(0)
            self._ticks(3)
        self._reset(1)
        self._ticks(4)
        self._reset(2)
        self._ticks(6)

    def test_one_row_per_episode(self):
        self._record(reset_first=True)

        rows = self._start_goal_rows()
        self.assertEqual(rows[0], ["episode", "start", "goal"])
        self.assertEqual(rows[1:], [
            [str(e), str(STARTS[e]), str(GOALS[e])] for e in (0, 1, 2)
        ])

    def test_metrics_merge_start_goal_by_episode(self):
        self._record(reset_first=False)

        # first-tick row with the recorder default, then the scenario_reset row
        self.assertEqual(len(self._start_goal_rows()), 1 + 4)

        folder = os.path.join(self.base_dir, "data", "run")
        warnings.simplefilter("ignore")
        for module in (metrics, metrics_static):
            with mock.patch.object(module.Metrics, "_get_robot_params", return_value={"robot_radius": 0.2}):
                episodes = module.Metrics(folder)._episode_data

            self.assertEqual(sorted(episodes), [0, 1, 2], module.__name__)
            for e in (0, 1, 2):
                self.assertEqual(episodes[e]["start"], STARTS[e], (module.__name__, e))
                self.assertEqual(episodes[e]["goal"], GOALS[e], (module.__name__, e))


class MetricsLegacyStartGoalTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        warnings.simplefilter("ignore")

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_per_tick_folder_reads_as_before(self):
        # old recorder: one start_goal row per tick, row aligned with episode.csv
        episodes = [0] * 4 + [1] * 3
        goals = [[0, 0, 0]] + [GOALS[0]] * 3 + [GOALS[1]] * 3
        times = [_ns(1) + i * 400 * 10**6 for i in range(len(episodes))]

        with open(os.path.join(self.folder, "params.yaml"), "w") as f:
            yaml.dump({"model": "rto", "time_unit": "ns"}, f)

        def write(name, header, rows):
            with open(os.path.join(self.folder, name + ".csv"), "w") as f:
                f.write(header + "\n" + "\n".join(rows) + "\n")

        write("episode", "time,episode", [f"{t},{e}" for t, e in zip(times, episodes)])
        write("scan", "time,data", [f"{t},\"[1.0, 2.0]\"" for t in times])
        write("odom", "time,data", [
            f"{t},\"{{'position': [{0.1 * i}, 0.0, 0.0], 'velocity': [0.1, 0.0, 0.0]}}\""
            for i, t in enumerate(times)
        ])
        write("start_goal", "episode,start,goal", [
            f"{e},\"[0, 0, 0]\",\"{g}\"" for e, g in zip(episodes, goals)
        ])

        for module in (metrics, metrics_static):
            with mock.patch.object(module.Metrics, "_get_robot_params", return_value={"robot_radius": 0.2}):
                data = module.Metrics(self.folder)._episode_data

            self.assertEqual(len(data[0]["path"]), 4)
            self.assertEqual(len(data[1]["path"]), 3)
            # the most frequent value of an episode, not the recorder default
            # written before the task generator set the param
            self.assertEqual(data[0]["goal"], GOALS[0])
            self.assertEqual(data[1]["goal"], GOALS[1])


if __name__ == "__main__":
    unittest.main()
