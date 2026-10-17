#!/usr/bin/env python3
"""Issue 02: the recorder stores true ns and metrics decode new and old folders."""

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


def _clock(secs, nsecs):
    return types.SimpleNamespace(clock=types.SimpleNamespace(secs=secs, nsecs=nsecs))


class RecorderTimestampTest(unittest.TestCase):

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()
        with open(os.path.join(self.base_dir, "data_recorder_config.yaml"), "w") as f:
            f.write("record_frequency: 400\n")

        FakeSubscriber.instances = []
        self.modules = mock.patch.dict(sys.modules, _stub_modules(self.base_dir))
        self.modules.start()

        loader = importlib.machinery.SourceFileLoader("data_recorder_node_under_test", RECORDER_PATH)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        self.node = importlib.util.module_from_spec(spec)
        loader.exec_module(self.node)

    def tearDown(self):
        self.modules.stop()
        shutil.rmtree(self.base_dir)

    def _publish_clock(self, secs, nsecs):
        for sub in list(FakeSubscriber.instances):
            if sub.registered and sub.topic == "/clock":
                sub.callback(_clock(secs, nsecs))

    def test_rows_hold_true_ns_and_gating_uses_true_time(self):
        self.node.Recorder("run")
        # 5.995 s, 6.0 s (5 ms later), then 200 ms steps
        for secs, nsecs in [(5, 995000000), (6, 0), (6, 200000000), (6, 400000000),
                            (6, 600000000), (6, 800000000), (7, 0), (7, 200000000)]:
            self._publish_clock(secs, nsecs)

        with open(os.path.join(self.base_dir, "data", "run", "episode.csv")) as f:
            rows = [line.split(",")[0] for line in f.read().splitlines()[1:] if line]

        self.assertEqual(rows, ["6400000000", "6800000000", "7200000000"])

    def test_params_declare_ns(self):
        self.node.Recorder("run")
        with open(os.path.join(self.base_dir, "data", "run", "params.yaml")) as f:
            self.assertEqual(yaml.safe_load(f)["time_unit"], "ns")


def _write_folder(folder, times, params):
    os.makedirs(folder)
    with open(os.path.join(folder, "params.yaml"), "w") as f:
        yaml.dump(params, f)

    def write(name, header, rows):
        with open(os.path.join(folder, name + ".csv"), "w") as f:
            f.write(header + "\n")
            for row in rows:
                f.write(row + "\n")

    write("episode", "time,episode", [f"{t},0" for t in times])
    write("scan", "time,data", [f"{t},\"[1.0, 2.0]\"" for t in times])
    write("odom", "time,data", [
        f"{t},\"{{'position': [{0.1 * i}, 0.0, 0.0], 'velocity': [0.1, 0.0, 0.0]}}\""
        for i, t in enumerate(times)
    ])
    write("start_goal", "episode,start,goal", ["0,\"[0, 0, 0]\",\"[1, 0, 0]\"" for _ in times])


class MetricsTimestampTest(unittest.TestCase):

    # true sim time of the samples: 5.995 s, 6.0 s, 6.4 s, 6.8 s, 7.0 s
    SECONDS = [(5, 995000000), (6, 0), (6, 400000000), (6, 800000000), (7, 0)]
    DURATION = 7.0 - 5.995

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()
        warnings.simplefilter("ignore")

    def tearDown(self):
        shutil.rmtree(self.base_dir)

    def _duration(self, module, folder):
        with mock.patch.object(module.Metrics, "_get_robot_params", return_value={"robot_radius": 0.2}):
            return module.Metrics(folder)._episode_data[0]["time_diff"]

    def test_new_folder_in_ns(self):
        folder = os.path.join(self.base_dir, "new")
        _write_folder(folder, [s * 10**9 + ns for s, ns in self.SECONDS], {"model": "rto", "time_unit": "ns"})

        for module in (metrics, metrics_static):
            self.assertAlmostEqual(self._duration(module, folder), self.DURATION, places=6)

    def test_legacy_folder_without_time_unit(self):
        folder = os.path.join(self.base_dir, "old")
        # encoding written by the old recorder: secs * 10e9 + nsecs, as float
        _write_folder(folder, [repr(s * 10e9 + ns) for s, ns in self.SECONDS], {"model": "rto"})

        for module in (metrics, metrics_static):
            self.assertAlmostEqual(self._duration(module, folder), self.DURATION, places=6)


if __name__ == "__main__":
    unittest.main()
