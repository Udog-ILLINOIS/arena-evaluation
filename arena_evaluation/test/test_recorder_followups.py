#!/usr/bin/env python3
"""Follow-ups of issues 01 and 02: sim time 0 is a time; a failed directory change keeps
recording; metrics row times are seconds and plots use them as such."""

import importlib.machinery
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from test_metrics_episode_results import Folder, MetricsTestCase, _line  # noqa: E402
from test_recorder_duplicate_folders import FakeSubscriber, NAMESPACE, RECORDER_PATH, _stub_modules  # noqa: E402

v = types.SimpleNamespace


class RecorderTest(unittest.TestCase):

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

    def clock(self, ns):
        for sub in list(FakeSubscriber.instances):
            if sub.registered and sub.topic == "/clock":
                sub.callback(v(clock=v(secs=ns // 10**9, nsecs=ns % 10**9)))

    def rows(self, folder):
        with open(os.path.join(self.base_dir, "data", folder, "episode.csv")) as f:
            return f.read().splitlines()[1:]

    def test_time_zero_is_the_reference(self):
        self.node.Recorder("run")
        self.clock(0)
        self.clock(400 * 10**6)
        # the 400 ms row is written because 0 was taken as the reference
        self.assertEqual([r.split(",")[0] for r in self.rows("run")], [str(400 * 10**6)])

    def test_failed_directory_change_keeps_recording(self):
        recorder = self.node.Recorder("run")
        with mock.patch.object(self.node, "Recorder", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.node.replace_recorder(recorder, "other")
        self.clock(10**9)
        self.clock(2 * 10**9)
        self.assertEqual(len(self.rows("run")), 1)


class MetricsTimeTest(MetricsTestCase):

    def test_row_times_are_seconds(self):
        folder = Folder(self.dir, {})
        folder.episode(0, _line(0, 2, 4), goal=(5, 0, 0))
        for name, episodes in self.analyze(folder).items():
            times = episodes[0]["time"]
            self.assertEqual([round(t - times[0], 3) for t in times], [0.0, 0.4, 0.8, 1.2], name)


if __name__ == "__main__":
    unittest.main()
