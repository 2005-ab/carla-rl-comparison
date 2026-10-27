"""Standalone 2-D driving environment. No CARLA server or display needed to train."""
from copy import deepcopy
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from common.config import EnvConfig
from common.reward import CarlaRewardCalculator

SCENARIOS = ("straight", "curve", "traffic", "two_lane", "mixed")


class LocalDrivingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, scenario="mixed", config=None):
        super().__init__()
        if scenario not in SCENARIOS:
            raise ValueError("Unknown scenario: " + scenario)
        self.config = deepcopy(config) if config else EnvConfig()
        self.config.simulator.fixed_delta_seconds = 0.1
        self.config.max_episode_steps = 500
        self.render_mode, self.scenario = render_mode, scenario
        self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        # speed, lateral, heading, remaining, forward clearance,
        # road heading at 8/20m relative to vehicle, traffic lateral, traffic speed, previous steer
        self.observation_space = spaces.Box(-1, 1, (10,), dtype=np.float32)
        self.reward_calculator = CarlaRewardCalculator(self.config.reward, self.config.route)
        self.renderer = None
        self.reset(seed=self.config.seed)

    def road_y(self, x):
        return self.amplitude * np.sin(np.asarray(x) / 32 + self.phase)

    def road_heading(self, x):
        return np.arctan(self.amplitude / 32 * np.cos(np.asarray(x) / 32 + self.phase))

    @property
    def is_two_lane(self):
        return self.active_scenario == "two_lane"

    @property
    def road_half_width(self):
        return 3.6 if self.is_two_lane else 4.0

    def road_center_y(self, x):
        """road_y remains the ego lane reference, preserving saved policy inputs.

        The challenge uses left-hand traffic: ego left lane, oncoming right lane.
        """
        return self.road_y(x) + (1.8 if self.is_two_lane else 0)

    def traffic_yaw(self, car):
        return float(self.road_heading(car[0])) + (math.pi if car[2] < 0 else 0)

    def geometry(self):
        tangent = float(self.road_heading(self.x))
        lateral = (self.y - float(self.road_y(self.x))) * math.cos(tangent)
        heading = math.atan2(math.sin(self.yaw - tangent), math.cos(self.yaw - tangent))
        return lateral, heading

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.active_scenario = options.get("scenario", self.scenario)
        if self.active_scenario == "mixed":
            self.active_scenario = str(self.np_random.choice(["straight", "curve", "traffic"]))
        if self.active_scenario not in SCENARIOS[:-1]:
            raise ValueError("Unknown scenario")
        self.amplitude = 0.0 if self.active_scenario == "straight" else float(self.np_random.uniform(7, 10) if self.is_two_lane else self.np_random.uniform(3, 7))
        self.phase = float(self.np_random.uniform(-math.pi, math.pi))
        self.x = 0.0
        self.y = float(self.road_y(0)) + float(self.np_random.uniform(-0.35, 0.35))
        self.yaw = float(self.road_heading(0)) + float(self.np_random.uniform(-0.04, 0.04))
        self.speed = 0.0
        self.step_count = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.total_reward = 0.0
        self.traffic = []
        if self.active_scenario == "traffic":
            self.traffic = [[float(s), 0.0, float(self.np_random.uniform(3.5, 5.0))] for s in (32, 85, 135)]
        elif self.is_two_lane:
            self.traffic = [[float(s), 0.0, float(self.np_random.uniform(3.0, 4.5))] for s in (28, 95)]
            self.traffic += [[float(s), 3.6, -float(self.np_random.uniform(7, 11))] for s in range(45, 431, 55)]
        self.trail, self.reward_history, self.speed_history = [], [], []
        self.done = False
        self.last_info = self._info(False, False, False, {})
        return self._observation(), self.last_info.copy()

    def nearest_traffic(self):
        ahead = [t for t in self.traffic if t[0] >= self.x - 4]
        if self.is_two_lane:
            # Detect vehicles threatening the current lane; lawful oncoming traffic
            # in the adjacent lane must not trigger emergency following/braking.
            offset = self.y-float(self.road_y(self.x))
            ahead = [t for t in ahead if abs(t[1]-offset) < 2.2]
        return min(ahead, key=lambda t: t[0]) if ahead else None

    def nearest_oncoming(self):
        ahead = [t for t in self.traffic if t[2]<0 and t[0]>=self.x]
        return min(ahead,key=lambda t:t[0]) if ahead else None

    def _overlaps_vehicle(self, car):
        """Separating axis test for two oriented 4.4 x 1.8 metre footprints."""
        dx,dy = car[0]-self.x, float(self.road_y(car[0]))+car[1]-self.y
        if abs(dx)>5.0 or abs(dy)>5.0:
            return False
        yaw = self.traffic_yaw(car)
        ego = ((math.cos(self.yaw),math.sin(self.yaw)),(-math.sin(self.yaw),math.cos(self.yaw)))
        other = ((math.cos(yaw),math.sin(yaw)),(-math.sin(yaw),math.cos(yaw)))
        def dot(a,b):
            return a[0]*b[0]+a[1]*b[1]
        for axis in (*ego,*other):
            radius = 2.2*(abs(dot(ego[0],axis))+abs(dot(other[0],axis))) + .9*(abs(dot(ego[1],axis))+abs(dot(other[1],axis)))
            if abs(dx*axis[0]+dy*axis[1]) >= radius:
                return False
        return True

    def _observation(self):
        lateral, heading = self.geometry()
        car = self.nearest_traffic()
        clearance = max(0, car[0] - self.x - 4.4) if car else 50
        obs = [self.speed * 3.6 / 50, lateral / 2.5, heading / math.pi,
               (self.config.route.route_length_m - self.x) / 200, clearance / 50,
               math.atan2(math.sin(float(self.road_heading(self.x + 8)) - self.yaw), math.cos(float(self.road_heading(self.x + 8)) - self.yaw)) / math.pi,
               math.atan2(math.sin(float(self.road_heading(self.x + 20)) - self.yaw), math.cos(float(self.road_heading(self.x + 20)) - self.yaw)) / math.pi,
               ((car[1] - lateral) / 5 if car else 0), (car[2] / 14 if car else 1), self.previous_action[0]]
        return np.clip(obs, -1, 1).astype(np.float32)

    def _info(self, collision, goal, offroad, components):
        lateral, heading = self.geometry()
        offset = self.y-float(self.road_y(self.x))
        oncoming = self.nearest_oncoming()
        return dict(step_count=self.step_count, speed_kmh=self.speed * 3.6,
                    lateral_deviation_m=lateral, heading_error_rad=heading,
                    distance_to_goal_m=max(0.0, self.config.route.route_length_m - self.x),
                    route_completion_pct=float(np.clip(100 * self.x / self.config.route.route_length_m, 0, 100)),
                    collision=bool(collision), goal_reached=bool(goal), out_of_lane=bool(offroad),
                    reward_components=components, scenario=self.active_scenario,
                    wrong_lane=bool(self.is_two_lane and offset>1.8),
                    oncoming_distance_m=float(max(0,oncoming[0]-self.x-4.4)) if oncoming else None,
                    oncoming_speed_kmh=float(-oncoming[2]*3.6) if oncoming else None,
                    termination_reason="collision" if collision else "lane departure" if offroad else "goal reached" if goal else "time limit" if self.step_count >= self.config.max_episode_steps else "driving")

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended. Call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Action must contain three finite values")
        action = np.clip(action, -1, 1)
        steer, throttle, brake = float(action[0]), max(0, float(action[1])), max(0, float(action[2]))
        if brake > self.config.action.brake_deadzone:
            throttle = 0.0
        applied = np.array([steer, throttle, brake], dtype=np.float32)
        dt = self.config.simulator.fixed_delta_seconds
        old_x = self.x
        # Kinematic bicycle model: wheelbase 2.7m, max wheel angle 0.45rad.
        # Substeps limit tunnelling and keep collision checks independent of display FPS.
        collision = False
        for _ in range(5):
            self.speed = float(np.clip(self.speed + (4 * throttle - 8 * brake - 0.12 * self.speed) * dt / 5, 0, 14))
            self.yaw += self.speed / 2.7 * math.tan(0.45 * steer) * dt / 5
            self.x += self.speed * math.cos(self.yaw) * dt / 5
            self.y += self.speed * math.sin(self.yaw) * dt / 5
            for car in self.traffic:
                car[0] += car[2] * dt / 5
                contact = self._overlaps_vehicle(car) if self.is_two_lane else (abs(car[0] - self.x) < 4.4 and abs(float(self.road_y(car[0])) + car[1] - self.y) < 1.8)
                if contact:
                    collision = True
        self.step_count += 1
        lateral, heading = self.geometry()
        offset = self.y-float(self.road_y(self.x))
        wrong_lane = self.is_two_lane and offset>1.8
        offroad = (offset < -1.8 or offset > 5.4) if self.is_two_lane else abs(lateral) > 2.5
        goal = self.x >= self.config.route.route_length_m and not (collision or offroad or wrong_lane)
        reward, components = self.reward_calculator.compute_reward(
            self.x - old_x, self.speed * 3.6, lateral, heading, applied,
            self.previous_action, collision, goal, offroad)
        if self.is_two_lane:
            # Crossing the divider is penalized but does not immediately end the
            # episode: real oncoming collisions remain possible and visible.
            penalty = -2.0 if wrong_lane else 0.0
            components["r_wrong_lane"] = penalty
            reward += penalty
            components["total_reward"] = reward
        self.previous_action = applied
        self.total_reward += reward
        self.trail.append((self.x, self.y))
        self.reward_history.append(reward)
        self.speed_history.append(self.speed * 3.6)
        terminated = bool(collision or offroad or goal)
        truncated = bool(self.step_count >= self.config.max_episode_steps and not terminated)
        self.done = terminated or truncated
        self.last_info = self._info(collision, goal, offroad, components)
        return self._observation(), float(reward), terminated, truncated, self.last_info.copy()

    def render(self):
        if self.renderer is None:
            from common.local_renderer import DrivingRenderer
            self.renderer = DrivingRenderer(self.render_mode or "rgb_array")
        return self.renderer.draw(self)

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None


def baseline_action(env):
    """Explicit hand-coded lane/speed controller, used only as a labelled reference."""
    lateral, heading = env.geometry()
    desired_heading = float(env.road_heading(env.x + 6))
    steer = np.clip(1.8 * (desired_heading - env.yaw) - 0.18 * lateral, -1, 1)
    target = env.config.route.target_speed_kmh / 3.6
    car = env.nearest_traffic()
    if car:
        target = min(target, max(0, car[2] + (car[0] - env.x - 12) * 0.35))
    error = target - env.speed
    return np.array([steer, np.clip(error * .5 + .12 * target / 4, 0, 1), np.clip(-error * .4, 0, 1)], dtype=np.float32)
