"""
This file is used to calculate from the simulation data, various metrics, such as
- did a collision occur
- how long did the robot take form start to goal
the metrics / evaluation data will be saved to be preproccesed in the next step
"""
import enum
import typing
from typing import List
import numpy as np
import pandas as pd
import os
from pandas.core.api import DataFrame as DataFrame
import yaml
import rospkg
import json

from arena_evaluation.utils import Utils

class Action(str, enum.Enum):
    STOP = "STOP"
    ROTATE = "ROTATE"
    MOVE = "MOVE"


class DoneReason(str, enum.Enum):
    TIMEOUT = "TIMEOUT"
    GOAL_REACHED = "GOAL_REACHED"
    COLLISION = "COLLISION"
    ABORTED = "ABORTED" # ended before the timeout, away from the goal


class Metric(typing.TypedDict):

    time: typing.List[int]
    time_diff: int
    episode: int
    goal: typing.List
    start: typing.List

    path: typing.List
    path_length_values: typing.List
    path_length: float
    angle_over_length: float
    curvature: typing.List
    normalized_curvature: typing.List 
    roughness: typing.List

#    cmd_vel: typing.List
    velocity: typing.List
    acceleration: typing.List
    jerk: typing.List
    
    collision_amount: int
    collisions: typing.List
    
#    action_type: typing.List[Action]
    result: DoneReason

class PedsimMetric(Metric, typing.TypedDict):

    num_pedestrians: int

    avg_velocity_in_personal_space: float
    total_time_in_personal_space: int
    time_in_personal_space: typing.List[int]

    total_time_looking_at_pedestrians: int
    time_looking_at_pedestrians: typing.List[int]

    total_time_looked_at_by_pedestrians: int
    time_looked_at_by_pedestrians: typing.List[int]


class Config:
    TIMEOUT_TRESHOLD = 180 # used when params.yaml has no task_timeout
    MAX_COLLISIONS = 1
    MIN_EPISODE_LENGTH = 1
    RECORD_FREQUENCY = 400 # ms, used when params.yaml has no record_frequency

    # task generator defaults (TaskGenerator.cfg), used when params.yaml has none
    GOAL_RADIUS = 1.0
    GOAL_TOLERANCE_ANGLE = 2 * np.pi
    GOAL_CHECK_PERIOD = 0.5 # s, task generator checks for the goal at this period
    
    PERSONAL_SPACE_RADIUS = 1 # personal space is estimated at around 1'-4'
    ROBOT_GAZE_ANGLE = np.radians(5) # size of (one half of) direct robot gaze cone
    PEDESTRIAN_GAZE_ANGLE = np.radians(5) # size of (one half of) direct ped gaze cone

class Math:

    @classmethod
    def round_values(cls, values, digits=3):
        return [round(v, digits) for v in values]

    @classmethod
    def grouping(cls, base: np.ndarray, size: int) -> np.ndarray:
        return np.moveaxis( 
            np.array([
                np.roll(base, i, 0)
                for i
                in range(size)
            ]),
            [1],
            [0]
        )[:-size]

    @classmethod
    def triangles(cls, position: np.ndarray) -> np.ndarray:
        return cls.grouping(position, 3)
    
    @classmethod
    def triangle_area(cls, vertices: np.ndarray) -> np.ndarray:
        return np.linalg.norm(
            np.cross(
                vertices[:,1] - vertices[:,0],
                vertices[:,2] - vertices[:,0],
                axis=1
            ),
            axis=1
        ) / 2
    
    @classmethod
    def path_length(cls, position: np.ndarray) -> np.ndarray:
        pairs = cls.grouping(position, 2)
        return np.linalg.norm(pairs[:,0,:] - pairs[:,1,:], axis=1)

    @classmethod
    def curvature(cls, position: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

        triangles = cls.triangles(position)

        d01 = np.linalg.norm(triangles[:,0,:] - triangles[:,1,:], axis=1)
        d12 = np.linalg.norm(triangles[:,1,:] - triangles[:,2,:], axis=1)
        d20 = np.linalg.norm(triangles[:,2,:] - triangles[:,0,:], axis=1)

        triangle_area = cls.triangle_area(triangles)
        divisor = np.prod([d01, d12, d20], axis=0)
        divisor[divisor==0] = np.nan

        curvature = 4 * triangle_area / divisor
        curvature[np.isnan(divisor)] = 0

        normalized = np.multiply(
            curvature,
            d01 + d12
        )

        return curvature, normalized

    @classmethod
    def roughness(cls, position: np.ndarray) -> np.ndarray:
        
        triangles = cls.triangles(position)

        triangle_area = cls.triangle_area(triangles)
        length = np.linalg.norm(triangles[:,:,0] - triangles[:,:,2], axis=1)
        length[length==0] = np.nan

        roughness = 2 * triangle_area / np.square(length)
        roughness[np.isnan(length)] = 0

        return roughness

    @classmethod
    def acceleration(cls, speed: np.ndarray) -> np.ndarray:
        return np.diff(speed)

    @classmethod
    def jerk(cls, speed: np.ndarray) -> np.ndarray:
        return np.diff(np.diff(speed))

    @classmethod
    def turn(cls, yaw: np.ndarray) -> np.ndarray:
        pairs = cls.grouping(yaw, 2)
        return cls.angle_difference(pairs[:,0], pairs[:,1])
    
    @classmethod
    def angle_difference(cls, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            return np.pi - np.abs(np.abs(x1 - x2) - np.pi)

class Metrics:

    dir: str
    _episode_data: typing.Dict[int, Metric]

    def _load_data(self) -> typing.List[pd.DataFrame]:
        episode = pd.read_csv(os.path.join(self.dir, "episode.csv"), converters={
            "data": lambda val: 0 if len(val) <= 0 else int(val) 
        })

        laserscan = pd.read_csv(os.path.join(self.dir, "scan.csv"), converters={
            "data": Utils.string_to_float_list
        }).rename(columns={"data": "laserscan"})

        odom = pd.read_csv(os.path.join(self.dir, "odom.csv"), converters={
            "data": lambda col: json.loads(col.replace("'", "\"")) if col else None
        }).rename(columns={"data": "odom"})

#        cmd_vel = pd.read_csv(os.path.join(self.dir, "cmd_vel.csv"), converters={
#            "data": Utils.string_to_float_list
#        }).rename(columns={"data": "cmd_vel"})

        frames = [
            episode,
            laserscan,
            odom,
#            cmd_vel,
        ]

        # smallest scan range between rows; written by newer recorders only
        scan_min_file = os.path.join(self.dir, "scan_min.csv")
        if os.path.exists(scan_min_file):
            frames.append(pd.read_csv(scan_min_file, converters={
                "data": lambda val: float(val) if val else np.nan
            }).rename(columns={"data": "scan_min"}))

        return frames

    def _load_start_goal(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dir, "start_goal.csv"), converters={
            "start": Utils.string_to_float_list,
            "goal": Utils.string_to_float_list
        })

    @staticmethod
    def _is_start_goal_per_tick(start_goal: pd.DataFrame) -> bool:
        # old recorders wrote one row per tick (row aligned with the other
        # files), new ones one row per episode (at most two: the first tick
        # and the scenario_reset of that episode)
        return len(start_goal) > 2 * start_goal["episode"].nunique()

    @staticmethod
    def _merge_start_goal(data: pd.DataFrame, start_goal: pd.DataFrame) -> pd.DataFrame:
        # the last row of an episode wins; episodes without a row get the
        # recorder's default
        rows = start_goal.drop_duplicates(subset="episode", keep="last").set_index("episode")

        for key in ["start", "goal"]:
            values = rows[key].to_dict()
            data[key] = [values.get(episode, [0, 0, 0]) for episode in data["episode"]]

        return data

    def __init__(self, dir: str):

        self.dir = dir
        self.robot_params = self._get_robot_params()
        self.time_unit = self._get_time_unit()
        self.run_params = self._get_run_params()

        frames = self._load_data()
        start_goal = self._load_start_goal()
        start_goal_per_tick = self._is_start_goal_per_tick(start_goal)

        if start_goal_per_tick:
            frames.append(start_goal)

        data = pd.concat(frames, axis=1, join="inner")
        data = data.loc[:,~data.columns.duplicated()].copy()

        if not start_goal_per_tick:
            data = self._merge_start_goal(data, start_goal)

        data = self._drop_empty_rows(data)

        episode_data = self._episode_data = {}

        # every episode id in the data, whether or not the ids are contiguous
        for i in sorted(data["episode"].unique()):
            current_episode = data[data["episode"] == i]

            # print("current_episode: ", current_episode)

            if len(current_episode) < Config.MIN_EPISODE_LENGTH:
                continue
            
            episode_data[i] = self._analyze_episode(current_episode, i)

    def _drop_empty_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        # rows written before the first message of a topic have an empty cell
        complete = np.array([odom is not None for odom in data["odom"]], dtype=bool)
        complete &= np.array([len(scan) > 0 for scan in data["laserscan"]], dtype=bool)

        return data[complete]

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._episode_data).transpose().set_index("episode")

    
    def _analyze_episode(self, episode: pd.DataFrame, index) -> Metric:

        episode["time"] = Utils.time_to_seconds(episode["time"], self.time_unit)
        
        positions = np.array([frame["position"] for frame in episode["odom"]])
        # linear velocity (vx, vy); the angular rate is not part of the speed
        velocities = np.array([frame["velocity"][:2] for frame in episode["odom"]])

        curvature, normalized_curvature = Math.curvature(positions)
        roughness = Math.roughness(positions)

        vel_absolute = np.linalg.norm(velocities, axis=1)
        acceleration = Math.acceleration(vel_absolute)
        jerk = Math.jerk(vel_absolute)

        if "scan_min" in episode:
            collisions, collision_amount = self._get_collisions_from_min(
                episode["scan_min"],
                self.robot_params["robot_radius"]
            )
        else:
            collisions, collision_amount = self._get_collisions(
                episode["laserscan"],
                self.robot_params["robot_radius"]
            )

        path_length = Math.path_length(positions)
        turn = Math.turn(positions[:,2])

        time = list(episode["time"])[-1] - list(episode["time"])[0]

        start_position = self._get_mean_position(episode, "start")
        goal_position = self._get_mean_position(episode, "goal")

        # print("PATH LENGTH", path_length, path_length_per_step)

        return Metric(
            curvature = Math.round_values(curvature),
            normalized_curvature = Math.round_values(normalized_curvature),
            roughness = Math.round_values(roughness),
            path_length_values = Math.round_values(path_length),
            path_length = path_length.sum(),
            acceleration = Math.round_values(acceleration),
            jerk = Math.round_values(jerk),
            velocity = Math.round_values(vel_absolute),
            collision_amount = collision_amount,
            collisions = list(collisions),
            path = [list(p) for p in positions],
            angle_over_length = np.abs(turn.sum() / path_length.sum()),
#            action_type = list(self._get_action_type(episode["cmd_vel"])),
            time_diff = time, ## Ros time in ns
            time = list(map(int, episode["time"].tolist())),
            episode = index,
            result = self._get_success(time, collision_amount, positions[-1], velocities[-1], goal_position),
#            cmd_vel = list(map(list, episode["cmd_vel"].to_list())),
            goal = goal_position,
            start = start_position
        )
    
    def _get_robot_params(self):
        with open(os.path.join(self.dir, "params.yaml")) as file:
            content = yaml.safe_load(file)

            model = content["model"]

        robot_model_params_file = os.path.join(
            rospkg.RosPack().get_path("arena_simulation_setup"),
            "entities",
            "robots", 
            model, 
            "model_params.yaml"
        )

        with open(robot_model_params_file, "r") as file:
            return yaml.safe_load(file)

    def _get_time_unit(self):
        with open(os.path.join(self.dir, "params.yaml")) as file:
            return yaml.safe_load(file).get("time_unit")

    def _get_run_params(self):
        with open(os.path.join(self.dir, "params.yaml")) as file:
            content = yaml.safe_load(file)

        def get(key, default):
            value = content.get(key)
            return default if value is None else value

        timeout = get("task_timeout", Config.TIMEOUT_TRESHOLD)

        return {
            # the task generator treats a negative timeout as none
            "timeout": np.inf if timeout < 0 else timeout,
            "record_interval": get("record_frequency", Config.RECORD_FREQUENCY) / 1000,
            "goal_radius": get("goal_radius", Config.GOAL_RADIUS),
            "goal_tolerance_angle": get("goal_tolerance_angle", Config.GOAL_TOLERANCE_ANGLE)
        }

    def _get_mean_position(self, episode, key):
        positions = episode[key].to_list()
        counter = {}

        for p in positions:
            hash = ":".join([str(pos) for pos in p])

            counter[hash] = counter.get(hash, 0) + 1

        # most frequent value; old per-tick files start an episode with a few
        # rows of the recorder default before the task generator sets the param
        most_frequent = max(counter.items(), key=lambda x: x[1])[0]

        return [float(r) for r in most_frequent.split(":")]

    def _get_position_for_collision(self, collisions, positions):
        for i, collision in enumerate(collisions):
            collisions[i][2] = positions[collision[0]]

        return collisions

    def _get_success(self, time, collisions, final_position, final_velocity, goal):
        # rows are written every record interval, so the last row of an
        # episode that timed out can be up to one interval (plus one tick)
        # before the timeout
        if time >= self.run_params["timeout"] - 2 * self.run_params["record_interval"]:
            return DoneReason.TIMEOUT

        if collisions >= Config.MAX_COLLISIONS:
            return DoneReason.COLLISION

        if self._is_at_goal(final_position, final_velocity, goal):
            return DoneReason.GOAL_REACHED

        return DoneReason.ABORTED

    def _is_at_goal(self, position, velocity, goal):
        # same test as the task generator (RobotManager._is_goal_reached), on
        # the last recorded row. The task generator checks every
        # GOAL_CHECK_PERIOD and rows are written every record interval, so
        # the last row can be from before the robot entered the goal radius;
        # allow for the distance it covers in that time.
        lag = self.run_params["record_interval"] + Config.GOAL_CHECK_PERIOD
        radius = self.run_params["goal_radius"] + np.linalg.norm(velocity) * lag

        distance = np.linalg.norm(np.array(goal[:2]) - np.array(position[:2]))
        angle = Math.angle_difference(goal[2], position[2])

        return (
            distance < radius
            and angle < self.run_params["goal_tolerance_angle"]
        )
    
    def _get_collisions(self, laser_scans, lower_bound):
        """
        Calculates the collisions. Therefore, 
        the laser scans is examinated and all values below a 
        specific range are marked as collision.

        Argument:
            - Array laser scans representing the scans over
            time
            - the lower bound for which a collisions are counted

        Returns tupel of:
            - Array of tuples with indexs and time in which
            a collision happened
        """
        collisions = []
        collisions_marker = []

        eps = 0.01 # 0.0075 # empirical, scan never actually gets to robot radius

        print("Lower bound: ", lower_bound)
        print("Lower bound + eps: ", lower_bound + eps)

        for i, scan in enumerate(laser_scans):

            # print(scan)

            is_collision = len(scan[scan <= (lower_bound + eps)]) > 0

            collisions_marker.append(is_collision)
            
            if is_collision:
                collisions.append(i)

        collision_amount = 0

        for i, coll in enumerate(collisions_marker[1:]):
            prev_coll = collisions_marker[i]

            if coll - prev_coll > 0:
                collision_amount += 1

        return collisions, collision_amount

    def _get_collisions_from_min(self, scan_min, lower_bound):
        """
        Like _get_collisions, from the smallest scan range seen between two
        rows (scan_min.csv), so contacts shorter than a row are counted.
        A row without a scan since the previous one keeps the previous state.
        """
        eps = 0.01 # same as _get_collisions

        collisions_marker = []

        for value in scan_min:
            if np.isnan(value):
                collisions_marker.append(collisions_marker[-1] if collisions_marker else False)
            else:
                collisions_marker.append(value <= (lower_bound + eps))

        collisions = [i for i, is_collision in enumerate(collisions_marker) if is_collision]

        collision_amount = 0

        for i, coll in enumerate(collisions_marker[1:]):
            prev_coll = collisions_marker[i]

            if coll - prev_coll > 0:
                collision_amount += 1

        return collisions, collision_amount

    def _get_action_type(self, actions):
        action_type = []

        for action in actions:
            if sum(action) == 0:
                action_type.append(Action.STOP.value)
            elif action[0] == 0 and action[1] == 0:
                action_type.append(Action.ROTATE.value)
            else:
                action_type.append(Action.MOVE.value)

        return action_type

    
        
class PedsimMetrics(Metrics):

    def _load_data(self) -> List[DataFrame]:
        pedsim_data = pd.read_csv(
            os.path.join(self.dir, "pedsim_agents_data.csv"),
            converters = {"data": lambda entry: Utils.parse_pedsim(entry) if entry else None}
        ).rename(columns={"data": "peds"})
        
        return super()._load_data() + [pedsim_data]

    def _drop_empty_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        data = super()._drop_empty_rows(data)

        return data[np.array([peds is not None for peds in data["peds"]], dtype=bool)]
    
    def __init__(self, dir: str, **kwargs):
        super().__init__(dir=dir, **kwargs)

    def _analyze_episode(self, episode: pd.DataFrame, index):

        max_peds = max(*(len(peds) for peds in episode['peds']), 0)
        episode = episode.iloc[[len(peds) == max_peds for peds in episode['peds']]].copy()

        super_analysis = super()._analyze_episode(episode, index)

        robot_position = np.array([odom["position"][:2] for odom in episode["odom"]])
        peds_position = np.array([[ped.position for ped in peds] for peds in episode["peds"]])

        # list of (timestamp, ped) indices, duplicate timestamps allowed
        personal_space_frames = np.linalg.norm(peds_position - robot_position[:,None], axis=-1) <= Config.PERSONAL_SPACE_RADIUS
        # list of timestamp indices, no duplicates
        is_personal_space = personal_space_frames.max(axis=1)

        # time in personal space
        time = np.diff(np.array(episode["time"]), prepend=0)
        total_time_in_personal_space = time[is_personal_space].sum()
        time_in_personal_space = [time[frames].sum(axis=0).astype(np.integer) for frames in personal_space_frames.T]

        # v_avg in personal space
        velocity = np.array(super_analysis["velocity"])
        velocity = velocity[is_personal_space]
        avg_velocity_in_personal_space = velocity.mean() if velocity.size else 0


        # gazes
        robot_direction = np.array([odom["position"][2] for odom in episode["odom"]])
        peds_direction = np.array([[ped.theta for ped in peds] for peds in episode["peds"]])
        angle_robot_peds = np.squeeze(np.angle(np.array(peds_position - robot_position[:,np.newaxis]).view(np.complex128)))

        # time looking at pedestrians
        robot_gaze = Math.angle_difference(robot_direction[:,np.newaxis], angle_robot_peds)
        looking_at_frames = np.abs(robot_gaze) <= Config.ROBOT_GAZE_ANGLE
        total_time_looking_at_pedestrians = time[looking_at_frames.max(axis=1)].sum()
        time_looking_at_pedestrians = [time[frames].sum(axis=0).astype(np.integer) for frames in looking_at_frames.T]
        
        # time being looked at by pedestrians
        ped_gaze = Math.angle_difference(peds_direction, np.pi - angle_robot_peds)
        looked_at_frames = np.abs(ped_gaze) <= Config.PEDESTRIAN_GAZE_ANGLE
        total_time_looked_at_by_pedestrians = time[looked_at_frames.max(axis=1)].sum()
        time_looked_at_by_pedestrians = [time[frames].sum(axis=0).astype(np.integer) for frames in looked_at_frames.T]

        return PedsimMetric(
            **super_analysis,
            avg_velocity_in_personal_space = avg_velocity_in_personal_space,
            total_time_in_personal_space = total_time_in_personal_space,
            time_in_personal_space = time_in_personal_space,
            total_time_looking_at_pedestrians = total_time_looking_at_pedestrians,
            time_looking_at_pedestrians = time_looking_at_pedestrians,
            total_time_looked_at_by_pedestrians = total_time_looked_at_by_pedestrians,
            time_looked_at_by_pedestrians = time_looked_at_by_pedestrians,
            num_pedestrians = peds_position.shape[1]
        )