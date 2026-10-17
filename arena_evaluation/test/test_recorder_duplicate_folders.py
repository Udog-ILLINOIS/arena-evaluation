#!/usr/bin/env python3
"""Issue 01: after a directory change the old recorder must stop writing."""

import importlib.machinery
import numpy  # noqa: F401  imported once here so the node import does not reload it
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

RECORDER_PATH = os.environ.get(
    "DATA_RECORDER_NODE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "data_recorder_node"),
)

NAMESPACE = "/rto/"


class FakeSubscriber:
    instances = []

    def __init__(self, topic, msg_class, callback):
        self.topic = topic
        self.callback = callback
        self.registered = True
        FakeSubscriber.instances.append(self)

    def unregister(self):
        self.registered = False


def _stub_modules(base_dir):
    rospy = types.ModuleType("rospy")
    rospy.Subscriber = FakeSubscriber
    rospy.get_namespace = lambda: NAMESPACE
    rospy.get_param = lambda name, default=None: default
    rospy.get_rostime = lambda: types.SimpleNamespace(secs=0, nsecs=0)
    rospy_core = types.ModuleType("rospy.core")
    rospy_core.traceback = __import__("traceback")
    rospy.core = rospy_core

    rostopic = types.ModuleType("rostopic")
    rostopic.get_topic_list = lambda: ([
        [NAMESPACE + "scan", "sensor_msgs/LaserScan", []],
        [NAMESPACE + "odom", "nav_msgs/Odometry", []],
        [NAMESPACE + "cmd_vel", "geometry_msgs/Twist", []],
    ], [])

    rospkg = types.ModuleType("rospkg")
    rospkg.RosPack = lambda: types.SimpleNamespace(get_path=lambda pkg: base_dir)

    def msg_module(name, *classes):
        mod = types.ModuleType(name)
        for cls in classes:
            setattr(mod, cls, type(cls, (), {}))
        return mod

    tf = types.ModuleType("tf")
    tf_transformations = types.ModuleType("tf.transformations")
    tf_transformations.euler_from_quaternion = lambda q: (0.0, 0.0, 0.0)
    tf.transformations = tf_transformations

    arena_evaluation = types.ModuleType("arena_evaluation")
    arena_evaluation_utils = types.ModuleType("arena_evaluation.utils")
    arena_evaluation_utils.Pedestrian = object
    arena_evaluation_msgs = types.ModuleType("arena_evaluation_msgs")

    return {
        "rospy": rospy,
        "rospy.core": rospy_core,
        "rostopic": rostopic,
        "rospkg": rospkg,
        "rosgraph_msgs": types.ModuleType("rosgraph_msgs"),
        "rosgraph_msgs.msg": msg_module("rosgraph_msgs.msg", "Clock"),
        "std_msgs": types.ModuleType("std_msgs"),
        "std_msgs.msg": msg_module("std_msgs.msg", "Int16"),
        "geometry_msgs": types.ModuleType("geometry_msgs"),
        "geometry_msgs.msg": msg_module("geometry_msgs.msg", "Twist"),
        "sensor_msgs": types.ModuleType("sensor_msgs"),
        "sensor_msgs.msg": msg_module("sensor_msgs.msg", "LaserScan"),
        "nav_msgs": types.ModuleType("nav_msgs"),
        "nav_msgs.msg": msg_module("nav_msgs.msg", "Odometry"),
        "pedsim_msgs": types.ModuleType("pedsim_msgs"),
        "pedsim_msgs.msg": msg_module("pedsim_msgs.msg", "PedsimAgentsDataframe"),
        "arena_evaluation_msgs": arena_evaluation_msgs,
        "arena_evaluation_msgs.srv": msg_module("arena_evaluation_msgs.srv", "ChangeDirectory"),
        "arena_evaluation": arena_evaluation,
        "arena_evaluation.utils": arena_evaluation_utils,
        "tf": tf,
        "tf.transformations": tf_transformations,
    }


def _clock(secs):
    return types.SimpleNamespace(clock=types.SimpleNamespace(secs=secs, nsecs=0))


def _scan(ranges):
    return types.SimpleNamespace(ranges=ranges, range_max=10.0)


class RecorderDuplicateFoldersTest(unittest.TestCase):

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

    def _replace(self, recorder, result_dir):
        if hasattr(self.node, "replace_recorder"):
            return self.node.replace_recorder(recorder, result_dir)
        # behaviour of the reconfigure() service handler before the fix
        del recorder
        return self.node.Recorder(result_dir)

    def _publish(self, topic, msg):
        for sub in list(FakeSubscriber.instances):
            if sub.registered and sub.topic == topic:
                sub.callback(msg)

    def _rows(self, folder, name):
        path = os.path.join(self.base_dir, "data", folder, name + ".csv")
        with open(path) as f:
            return [line for line in f.read().splitlines() if line]

    def _tick(self, secs, episode):
        self._publish("/scenario_reset", types.SimpleNamespace(data=episode))
        self._publish(NAMESPACE + "scan", _scan([1.0, 2.0]))
        self._publish("/clock", _clock(secs))

    def test_old_folder_gets_no_rows_after_directory_change(self):
        recorder = self.node.Recorder("run/scenario_1")
        for secs in range(1, 4):
            self._tick(secs, 1)

        old_episode_rows = len(self._rows("run/scenario_1", "episode"))
        old_scan_rows = len(self._rows("run/scenario_1", "scan"))
        self.assertGreater(old_episode_rows, 1)

        recorder = self._replace(recorder, "run/scenario_2")
        for secs in range(10, 14):
            self._tick(secs, 2)

        self.assertEqual(len(self._rows("run/scenario_1", "episode")), old_episode_rows)
        self.assertEqual(len(self._rows("run/scenario_1", "scan")), old_scan_rows)
        self.assertEqual(len(self._rows("run/scenario_1", "start_goal")), old_episode_rows)
        self.assertGreater(len(self._rows("run/scenario_2", "episode")), 1)
        self.assertGreater(len(self._rows("run/scenario_2", "scan")), 1)

    def test_shutdown_unregisters_every_subscriber(self):
        recorder = self.node.Recorder("run/scenario_1")
        old_subs = list(FakeSubscriber.instances)
        topics = {sub.topic for sub in old_subs}
        self.assertIn("/clock", topics)
        self.assertIn("/scenario_reset", topics)
        self.assertIn(NAMESPACE + "scan", topics)

        self._replace(recorder, "run/scenario_2")

        self.assertTrue(all(not sub.registered for sub in old_subs))
        self.assertTrue(all(sub.registered for sub in FakeSubscriber.instances if sub not in old_subs))

    def test_callback_racing_shutdown_writes_nothing(self):
        recorder = self.node.Recorder("run/scenario_1")
        self._tick(1, 1)
        clock_cb = [s.callback for s in FakeSubscriber.instances if s.topic == "/clock"][0]
        reset_cb = [s.callback for s in FakeSubscriber.instances if s.topic == "/scenario_reset"][0]
        rows = len(self._rows("run/scenario_1", "episode"))

        self._replace(recorder, "run/scenario_2")

        # messages already dequeued before unregister() still reach the callback
        reset_cb(types.SimpleNamespace(data=7))
        clock_cb(_clock(50))

        self.assertEqual(len(self._rows("run/scenario_1", "episode")), rows)
        self.assertEqual(recorder.current_episode, 1)


if __name__ == "__main__":
    unittest.main()
