#!/usr/bin/env python3
"""Issue 08: topics are subscribed by name and every row carries msg_time."""

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import arena_evaluation.metrics as metrics  # noqa: E402
import arena_evaluation.metrics_static as metrics_static  # noqa: E402
from arena_evaluation.utils import Pedestrian  # noqa: E402
from test_recorder_duplicate_folders import (  # noqa: E402
    FakeSubscriber, NAMESPACE, RECORDER_PATH, _stub_modules,
)

PEDSIM_TOPIC = "/pedsim_simulator/pedsim_agents_data"
TOPIC_FILES = {
    NAMESPACE + "scan": "scan",
    NAMESPACE + "odom": "odom",
    NAMESPACE + "cmd_vel": "cmd_vel",
    PEDSIM_TOPIC: "pedsim_agents_data",
}


def _ns(secs, nsecs=0):
    return secs * 10**9 + nsecs


def _scan():
    return types.SimpleNamespace(ranges=[1.0, 2.0], range_max=10.0)


def _odom(x):
    v = types.SimpleNamespace
    return v(
        pose=v(pose=v(position=v(x=x, y=0.0, z=0.0), orientation=v(x=0.0, y=0.0, z=0.0, w=1.0))),
        twist=v(twist=v(linear=v(x=0.5, y=0.0, z=0.0), angular=v(x=0.0, y=0.0, z=0.0))),
    )


def _twist():
    v = types.SimpleNamespace
    return v(linear=v(x=0.5, y=0.0, z=0.0), angular=v(x=0.0, y=0.0, z=0.1))


def _pedsim():
    v = types.SimpleNamespace
    agent = v(
        id="1", type="adult", social_state="WALKING",
        pose=v(position=v(x=3.0, y=1.0, z=0.0)),
        twist=v(linear=v(x=0.0, y=1.0, z=0.0), angular=v(x=0.0, y=0.0, z=0.0)),
        forces=v(force=v(x=0.0, y=1.0, z=0.0)),
        destination=v(x=3.0, y=5.0, z=0.0),
    )
    return v(agent_states=[agent])


class RecorderTopicDiscoveryTest(unittest.TestCase):

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()
        with open(os.path.join(self.base_dir, "data_recorder_config.yaml"), "w") as f:
            f.write("record_frequency: 400\n")

        FakeSubscriber.instances = []
        stubs = _stub_modules(self.base_dir)
        # at startup only scan and odom are published, cmd_vel and pedsim come later
        stubs["rostopic"].get_topic_list = lambda: ([
            [NAMESPACE + "scan", "sensor_msgs/LaserScan", []],
            [NAMESPACE + "odom", "nav_msgs/Odometry", []],
        ], [])
        self.now = 0
        stubs["rospy"].get_rostime = lambda: types.SimpleNamespace(
            secs=self.now // 10**9, nsecs=self.now % 10**9)
        stubs["arena_evaluation.utils"].Pedestrian = Pedestrian

        self.modules = mock.patch.dict(sys.modules, stubs)
        self.modules.start()

        loader = importlib.machinery.SourceFileLoader("data_recorder_node_under_test", RECORDER_PATH)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        self.node = importlib.util.module_from_spec(spec)
        loader.exec_module(self.node)

    def tearDown(self):
        self.modules.stop()
        shutil.rmtree(self.base_dir)

    def _publish(self, topic, msg, at=None):
        if at is not None:
            self.now = at
        for sub in list(FakeSubscriber.instances):
            if sub.registered and sub.topic == topic:
                sub.callback(msg)

    def _clock(self, t):
        self.now = t
        self._publish("/clock", types.SimpleNamespace(
            clock=types.SimpleNamespace(secs=t // 10**9, nsecs=t % 10**9)))

    def _start(self):
        recorder = self.node.Recorder("run")
        # the first clock message only sets the reference time, rows start 400 ms later
        self._clock(_ns(0, 600000000))
        return recorder

    def _rows(self, folder, name):
        with open(os.path.join(self.base_dir, "data", folder, name + ".csv"), newline="") as f:
            return list(csv.reader(f))

    def test_subscribes_to_topics_not_yet_published(self):
        self.node.Recorder("run")
        topics = {sub.topic for sub in FakeSubscriber.instances}

        for topic, name in TOPIC_FILES.items():
            self.assertIn(topic, topics)
            self.assertTrue(os.path.isfile(os.path.join(self.base_dir, "data", "run", name + ".csv")), name)

    def test_cmd_vel_published_late_is_recorded(self):
        self._start()
        self._clock(_ns(1))
        self._publish(NAMESPACE + "cmd_vel", _twist(), at=_ns(1, 500000000))
        self._clock(_ns(2))

        rows = self._rows("run", "cmd_vel")
        self.assertEqual(rows[0], ["time", "data", "msg_time"])
        self.assertEqual(rows[1], [str(_ns(1)), "", ""])
        self.assertEqual(rows[2], [str(_ns(2)), "[0.5, 0.0, 0.1]", str(_ns(1, 500000000))])

    def test_held_rows_keep_receive_time_of_held_message(self):
        self._start()
        self._clock(_ns(1))
        for topic, msg in [(NAMESPACE + "scan", _scan()), (NAMESPACE + "odom", _odom(1.0)),
                           (NAMESPACE + "cmd_vel", _twist()), (PEDSIM_TOPIC, _pedsim())]:
            self._publish(topic, msg, at=_ns(1, 100000000))
        # nothing new arrives for two more ticks: rows repeat the data with the old msg_time
        self._clock(_ns(2))
        self._clock(_ns(3))
        self._publish(NAMESPACE + "scan", _scan(), at=_ns(3, 200000000))
        self._clock(_ns(4))

        for name in TOPIC_FILES.values():
            rows = self._rows("run", name)
            self.assertEqual(rows[0], ["time", "data", "msg_time"], name)
            self.assertEqual([r[0] for r in rows[1:]], [str(_ns(s)) for s in (1, 2, 3, 4)], name)
            self.assertEqual(rows[1][1:], ["", ""], name)
            self.assertEqual(rows[2][1], rows[3][1], name)

            expected = [_ns(1, 100000000)] * 2
            expected.append(_ns(3, 200000000) if name == "scan" else _ns(1, 100000000))
            self.assertEqual([r[2] for r in rows[2:]], [str(t) for t in expected], name)

    def test_shutdown_unregisters_topic_subscribers(self):
        recorder = self.node.Recorder("run")
        recorder.shutdown()

        for sub in FakeSubscriber.instances:
            self.assertFalse(sub.registered, sub.topic)
        self.assertEqual({c.full_topic_name for c in recorder.data_collectors}, set(TOPIC_FILES.values()))

    def test_metrics_read_folder_with_msg_time(self):
        self._start()
        self._clock(_ns(1))
        for i in range(6):
            t = _ns(1) + (i + 1) * 400 * 10**6
            for topic, msg in [(NAMESPACE + "scan", _scan()), (NAMESPACE + "odom", _odom(0.1 * i)),
                               (PEDSIM_TOPIC, _pedsim())]:
                self._publish(topic, msg, at=t - 10**8)
            if i == 0:
                self._publish(NAMESPACE + "cmd_vel", _twist(), at=t - 10**8)
            self._clock(t)

        folder = os.path.join(self.base_dir, "data", "run")
        # drop the first tick (before any message) like a recording whose topics were already up
        for name in list(TOPIC_FILES.values()) + ["episode", "start_goal"]:
            path = os.path.join(folder, name + ".csv")
            with open(path) as f:
                lines = f.read().splitlines()
            with open(path, "w") as f:
                f.write("\n".join([lines[0]] + lines[2:]) + "\n")

        warnings.simplefilter("ignore")
        for module in (metrics, metrics_static):
            for cls in (module.Metrics, module.PedsimMetrics):
                with mock.patch.object(cls, "_get_robot_params", return_value={"robot_radius": 0.2}):
                    episode = cls(folder)._episode_data[0]
                self.assertAlmostEqual(episode["time_diff"], 2.0, places=6)
                self.assertEqual(len(episode["path"]), 6)


if __name__ == "__main__":
    unittest.main()
