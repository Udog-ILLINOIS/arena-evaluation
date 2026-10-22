#!/usr/bin/env python3
"""Issue 36: repair_recorded_data removes duplicated episodes and converts legacy time."""

import csv
import importlib.machinery
import importlib.util
import os
import shutil
import tempfile
import unittest

import yaml

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "repair_recorded_data")


def load():
    loader = importlib.machinery.SourceFileLoader("repair_recorded_data", SCRIPT)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def legacy(secs, nsecs=0):
    return float(secs * 10**10 + nsecs)


def write_folder(path, episodes_from, last_episode, first_secs, legacy_time=True, nested=False):
    """A scenario folder as the old recorder left it: its first episode and every later one."""
    folder = os.path.join(path, "rto") if nested else path
    os.makedirs(folder)
    params = {"model": "rto", "namespace": "rto"}
    if not legacy_time:
        params["time_unit"] = "ns"
    with open(os.path.join(folder, "params.yaml"), "w") as f:
        yaml.dump(params, f)
    rows = []
    t = first_secs
    for episode in range(episodes_from, last_episode + 1):
        for _ in range(3):
            rows.append((legacy(t, 500000000) if legacy_time else float(t * 10**9 + 500000000), episode))
            t += 1
    with open(os.path.join(folder, "episode.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "episode"])
        w.writerows(rows)
    with open(os.path.join(folder, "scan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "data"])
        w.writerows([(t, "[1.0, 2.0]") for t, _ in rows])
    with open(os.path.join(folder, "start_goal.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "start", "goal"])
        w.writerows([(e, "[0, 0, 0]", "[{}, 0, 0]".format(e)) for _, e in rows])  # per tick


def read(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))[1:]


class RepairTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = load()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.run = os.path.join(self.tmp, "dgap_run")
        # created in this order; each holds its episode and all later ones
        write_folder(os.path.join(self.run, "dgap", "scenario_2"), 0, 2, 5, nested=True)
        write_folder(os.path.join(self.run, "dgap", "scenario_3"), 1, 2, 100)
        write_folder(os.path.join(self.run, "dgap", "scenario_10"), 2, 2, 200)
        self.before = {root: sorted(files) for root, _, files in os.walk(self.run)}
        with open(os.path.join(self.run, "dgap", "scenario_3", "episode.csv")) as f:
            self.original_episodes = f.read()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_plan_keeps_own_episodes(self):
        plan = self.m.plan_repair(self.run)
        keep = {os.path.relpath(f, self.run): info["keep"] for f, info in plan.items()}
        self.assertEqual(keep, {
            os.path.join("dgap", "scenario_2", "rto"): [0],
            os.path.join("dgap", "scenario_3"): [1],
            os.path.join("dgap", "scenario_10"): [2],
        })

    def test_dry_run_writes_nothing(self):
        self.m.repair(self.run, dry_run=True)
        self.assertFalse(os.path.exists(self.run + "_repaired"))

    def test_repaired_copy(self):
        self.m.repair(self.run)
        out = self.run + "_repaired"

        # the original is untouched
        self.assertEqual({root: sorted(files) for root, _, files in os.walk(self.run)}, self.before)
        with open(os.path.join(self.run, "dgap", "scenario_3", "episode.csv")) as f:
            self.assertEqual(f.read(), self.original_episodes)

        s3 = os.path.join(out, "dgap", "scenario_3")
        episodes = read(os.path.join(s3, "episode.csv"))
        self.assertEqual({r[1] for r in episodes}, {"1"})
        # secs * 1e10 + nsecs -> ns
        self.assertEqual(episodes[0][0], str(100 * 10**9 + 500000000))
        self.assertEqual([r[0] for r in read(os.path.join(s3, "scan.csv"))], [r[0] for r in episodes])
        self.assertEqual({r[0] for r in read(os.path.join(s3, "start_goal.csv"))}, {"1"})
        with open(os.path.join(s3, "params.yaml")) as f:
            self.assertEqual(yaml.safe_load(f)["time_unit"], "ns")

        s2 = os.path.join(out, "dgap", "scenario_2", "rto")
        self.assertEqual({r[1] for r in read(os.path.join(s2, "episode.csv"))}, {"0"})
        self.assertEqual(len(read(os.path.join(s2, "scan.csv"))), 3)

    def test_new_time_unit_is_not_converted(self):
        new_run = os.path.join(self.tmp, "new_run")
        write_folder(os.path.join(new_run, "a"), 0, 0, 7, legacy_time=False)
        self.m.repair(new_run)
        episodes = read(os.path.join(new_run + "_repaired", "a", "episode.csv"))
        self.assertEqual(float(episodes[0][0]), float(7 * 10**9 + 500000000))

    def test_existing_output_is_refused(self):
        os.makedirs(self.run + "_repaired")
        with self.assertRaises(SystemExit):
            self.m.repair(self.run)

    def test_metrics_read_the_repaired_folder(self):
        import sys
        from unittest import mock
        import warnings
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        import arena_evaluation.metrics as metrics
        warnings.simplefilter("ignore")
        # odom is needed by metrics; add it to the original before repairing
        for folder in self.m.recording_folders(self.run):
            rows = read(os.path.join(folder, "episode.csv"))
            with open(os.path.join(folder, "odom.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "data"])
                w.writerows([(r[0], "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}") for r in rows])
        self.m.repair(self.run)
        folder = os.path.join(self.run + "_repaired", "dgap", "scenario_3")
        with mock.patch.object(metrics.Metrics, "_get_robot_params", return_value={"robot_radius": 0.2}):
            data = metrics.Metrics(folder)._episode_data
        self.assertEqual(sorted(data), [1])
        self.assertAlmostEqual(data[1]["time_diff"], 2.0)


if __name__ == "__main__":
    unittest.main()
