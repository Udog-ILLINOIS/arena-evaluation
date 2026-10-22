#!/usr/bin/env python3
"""Issues 03, 04, 05, 32: episode loop, success definition, run metadata, velocity metric."""

import importlib.machinery
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import arena_evaluation.metrics as metrics  # noqa: E402
import arena_evaluation.metrics_static as metrics_static  # noqa: E402
from test_recorder_duplicate_folders import (  # noqa: E402
    FakeSubscriber, NAMESPACE, RECORDER_PATH, _stub_modules,
)

MODULES = (metrics, metrics_static)
TICK_NS = 400 * 10**6


def _ns(secs):
    return int(secs * 10**9)


class Folder:
    """Writes a recorder folder (new layout: time in ns, start_goal per episode)."""

    def __init__(self, path, params):
        self.path = path
        self.params = dict({"model": "rto", "time_unit": "ns"}, **params)
        self.rows = []  # (time, episode, position, velocity, scan)
        self.start_goal = []  # (episode, start, goal)

    def episode(self, episode, positions, goal, velocity=(0.5, 0.0, 0.0), start=None, scans=None):
        t0 = self.rows[-1][0] + TICK_NS if self.rows else _ns(1)
        scans = scans or [(1.0, 2.0)] * len(positions)
        for i, (p, scan) in enumerate(zip(positions, scans)):
            self.rows.append((t0 + i * TICK_NS, episode, list(p), list(velocity), list(scan)))
        self.start_goal.append((episode, list(start or positions[0]), list(goal)))

    def write(self, empty_first_row=False):
        with open(os.path.join(self.path, "params.yaml"), "w") as f:
            yaml.dump(self.params, f)

        def write(name, header, rows):
            with open(os.path.join(self.path, name + ".csv"), "w") as f:
                f.write(header + "\n" + "\n".join(rows) + "\n")

        def cell(i, text):
            return "" if empty_first_row and i == 0 else f"\"{text}\""

        write("episode", "time,episode", [f"{t},{e}" for t, e, _, _, _ in self.rows])
        write("scan", "time,data,msg_time", [
            f"{t},{cell(i, s)},{t}" for i, (t, _, _, _, s) in enumerate(self.rows)
        ])
        write("odom", "time,data,msg_time", [
            f"{t},{cell(i, {'position': p, 'velocity': v})},{t}"
            for i, (t, _, p, v, _) in enumerate(self.rows)
        ])
        write("start_goal", "episode,start,goal", [
            f"{e},\"{s}\",\"{g}\"" for e, s, g in self.start_goal
        ])


def _line(x0, x1, n):
    return [(x, 0.0, 0.0) for x in np.linspace(x0, x1, n)]


class MetricsTestCase(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        warnings.simplefilter("ignore")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def analyze(self, folder, **kwargs):
        folder.write(**kwargs)
        results = {}
        for module in MODULES:
            with mock.patch.object(module.Metrics, "_get_robot_params", return_value={"robot_radius": 0.2}):
                results[module.__name__] = module.Metrics(self.dir)._episode_data
        return results


class EpisodeLoopTest(MetricsTestCase):
    """Issue 04: every episode id present is analysed, not just 0..first gap."""

    def test_ids_not_starting_at_zero_and_with_gaps(self):
        folder = Folder(self.dir, {})
        for episode in (2, 3, 5, 9):
            folder.episode(episode, _line(0, 5, 6), goal=(5, 0, 0))

        for name, episodes in self.analyze(folder).items():
            self.assertEqual(sorted(episodes), [2, 3, 5, 9], name)
            self.assertEqual(episodes[5]["episode"], 5, name)

    def test_rows_with_empty_cells_are_skipped(self):
        # the recorder writes an empty data cell before the first message
        folder = Folder(self.dir, {})
        folder.episode(0, _line(0, 5, 6), goal=(5, 0, 0))

        for name, episodes in self.analyze(folder, empty_first_row=True).items():
            self.assertEqual(len(episodes[0]["path"]), 5, name)


class SuccessTest(MetricsTestCase):
    """Issue 03: GOAL_REACHED only at the goal; timeout from params.yaml."""

    def test_results(self):
        folder = Folder(self.dir, {"task_timeout": 10, "record_frequency": 400, "goal_radius": 1.0})
        # ends 0.5 m from the goal
        folder.episode(0, _line(0, 4.5, 6), goal=(5, 0, 0))
        # ends 3 m from the goal, before the timeout (e.g. the run was stopped)
        folder.episode(1, _line(0, 2, 6), goal=(5, 0, 0))
        # 25 rows = 9.6 s: within one tick of the 10 s timeout
        folder.episode(2, _line(0, 2, 25), goal=(5, 0, 0))
        # at the goal but touched an obstacle on the way
        folder.episode(3, _line(0, 4.5, 6), goal=(5, 0, 0), scans=[(1.0, 2.0), (1.0, 2.0), (0.1, 2.0)] + [(1.0, 2.0)] * 3)

        for name, episodes in self.analyze(folder).items():
            results = {e: str(episodes[e]["result"].value) for e in episodes}
            self.assertEqual(results, {
                0: "GOAL_REACHED", 1: "ABORTED", 2: "TIMEOUT", 3: "COLLISION",
            }, name)

    def test_last_row_before_goal_check(self):
        # 1 m/s towards the goal; the last row is 1.4 m out, but the robot
        # covers up to (0.4 + 0.5) m before the task generator sees it
        folder = Folder(self.dir, {"task_timeout": 60, "goal_radius": 1.0})
        folder.episode(0, _line(0, 3.6, 10), goal=(5, 0, 0), velocity=(1.0, 0.0, 0.0))
        # same position, standing still: not at the goal
        folder.episode(1, _line(0, 3.6, 10), goal=(5, 0, 0), velocity=(0.0, 0.0, 0.0))
        # 1 m/s but 2 m out: not at the goal
        folder.episode(2, _line(0, 3.0, 10), goal=(5, 0, 0), velocity=(1.0, 0.0, 0.0))

        for name, episodes in self.analyze(folder).items():
            results = {e: str(episodes[e]["result"].value) for e in episodes}
            self.assertEqual(results, {0: "GOAL_REACHED", 1: "ABORTED", 2: "ABORTED"}, name)

    def test_goal_radius_from_params(self):
        folder = Folder(self.dir, {"goal_radius": 0.25})
        folder.episode(0, _line(0, 4.5, 6), goal=(5, 0, 0), velocity=(0.0, 0.0, 0.0))

        for name, episodes in self.analyze(folder).items():
            self.assertEqual(episodes[0]["result"], "ABORTED", name)

    def test_goal_angle_from_params(self):
        folder = Folder(self.dir, {"goal_tolerance_angle": 0.5})
        folder.episode(0, _line(0, 4.9, 6), goal=(5, 0, 3.0), velocity=(0.0, 0.0, 0.0))

        for name, episodes in self.analyze(folder).items():
            self.assertEqual(episodes[0]["result"], "ABORTED", name)

    def test_legacy_params_use_defaults(self):
        # no task_timeout etc.: 180 s timeout, 1 m radius, any heading
        folder = Folder(self.dir, {})
        folder.episode(0, _line(0, 4.2, 6), goal=(5, 0, 2.0))

        for name, episodes in self.analyze(folder).items():
            self.assertEqual(episodes[0]["result"], "GOAL_REACHED", name)

    def test_negative_timeout_means_none(self):
        folder = Folder(self.dir, {"task_timeout": -1})
        folder.episode(0, _line(0, 1, 30), goal=(5, 0, 0), velocity=(0.0, 0.0, 0.0))

        for name, episodes in self.analyze(folder).items():
            self.assertEqual(episodes[0]["result"], "ABORTED", name)


class VelocityTest(MetricsTestCase):
    """Issue 32: speed comes from the odom velocity, not the position."""

    def test_speed_is_linear_velocity_norm(self):
        folder = Folder(self.dir, {})
        folder.episode(0, _line(10, 12, 5), goal=(12, 0, 0), velocity=(0.3, 0.4, 2.0))

        for name, episodes in self.analyze(folder).items():
            self.assertEqual(episodes[0]["velocity"], [0.5] * 5, name)
            self.assertEqual(episodes[0]["acceleration"], [0.0] * 4, name)


class RecorderParamsTest(unittest.TestCase):
    """Issue 05: params.yaml records the run settings."""

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()
        with open(os.path.join(self.base_dir, "data_recorder_config.yaml"), "w") as f:
            f.write("record_frequency: 250\n")

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

    def tearDown(self):
        self.modules.stop()
        shutil.rmtree(self.base_dir)

    def _params(self):
        self.node.Recorder("run")
        with open(os.path.join(self.base_dir, "data", "run", "params.yaml")) as f:
            return yaml.safe_load(f)

    def test_run_settings_written(self):
        self.params.update({
            "/man_gap": True,
            "/use_sim_time": True,
            "/task_generator_server/timeout": 60.0,
            "/task_generator_server/goal_radius": 0.5,
            "/task_generator_server/goal_tolerance_angle": 1.0,
        })

        params = self._params()

        self.assertIs(params["man_gap"], True)
        self.assertIs(params["use_sim_time"], True)
        self.assertEqual(params["record_frequency"], 250)
        self.assertEqual(params["task_timeout"], 60.0)
        self.assertEqual(params["goal_radius"], 0.5)
        self.assertEqual(params["goal_tolerance_angle"], 1.0)
        self.assertEqual(params["time_unit"], "ns")
        self.assertEqual(params["namespace"], NAMESPACE.replace("/", ""))

    def test_missing_params_are_null(self):
        params = self._params()

        for key in ("man_gap", "use_sim_time", "task_timeout", "goal_radius", "goal_tolerance_angle"):
            self.assertIn(key, params)
            self.assertIsNone(params[key], key)


if __name__ == "__main__":
    unittest.main()
