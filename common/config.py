"""
common/config.py

Centralized configuration for CARLA Reinforcement Learning experiments (SAC, TD3, PPO).
Defines dataclasses for simulator, environment, rewards, routing, and evaluation settings.
All three algorithms import and share these identical parameters for fair benchmarking.
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class SimulatorConfig:
    """CARLA Simulator connection and world settings."""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    town: str = "Town01"
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.05  # 20 Hz simulation rate
    vehicle_blueprint: str = "vehicle.tesla.model3"
    hero_role_name: str = "hero"
    render_display: bool = False
    spectator_follow: bool = True


@dataclass
class ObservationConfig:
    """
    Fixed-length float32 numerical observation vector specification.

    Observation vector layout (shape: (5,), dtype: np.float32):
      Index 0: Normalized vehicle speed (speed_kmh / max_speed), range [0.0, 1.0]
      Index 1: Normalized lateral deviation from lane center (lateral_dev / max_lateral_dev), range [-1.0, 1.0]
               Negative = left of lane center, Positive = right of lane center.
      Index 2: Normalized heading error relative to road tangent (heading_err_rad / pi), range [-1.0, 1.0]
               Negative = facing left of road direction, Positive = facing right.
      Index 3: Normalized distance to route goal (dist_to_goal / max_goal_dist), range [0.0, 1.0]
      Index 4: Normalized distance to nearest detected vehicle obstacle, range [0.0, 1.0]
               1.0 = no obstacle within sensor range; 0.0 = imminent contact.
    """
    obs_dim: int = 5
    max_speed_kmh: float = 50.0
    max_lateral_deviation_m: float = 2.5   # Standard lane half-width plus margin
    max_goal_distance_m: float = 200.0     # Route horizon for normalization
    max_obstacle_distance_m: float = 50.0  # Obstacle sensor horizon

    low: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32))
    high: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))


@dataclass
class ActionConfig:
    """
    Continuous action space specification: Box(low=-1.0, high=1.0, shape=(3,), dtype=float32)

    Action components:
      Index 0: Steering command in [-1.0, 1.0] (mapped directly to CARLA VehicleControl.steer)
      Index 1: Throttle command in [-1.0, 1.0] (mapped to [0.0, 1.0] where action > 0 provides positive throttle)
      Index 2: Brake command in [-1.0, 1.0] (mapped to [0.0, 1.0] where action > 0 provides positive braking)

    Safety constraint:
      Simultaneous positive throttle and braking is strictly prohibited.
      If brake > brake_deadzone (0.05), throttle is clamped to 0.0.
    """
    action_dim: int = 3
    low: float = -1.0
    high: float = 1.0
    brake_deadzone: float = 0.05
    steer_smoothing_alpha: float = 0.0  # 0.0 for direct control


@dataclass
class RouteConfig:
    """Route generation and tracking settings."""
    route_id: int = 0
    spawn_point_index: Optional[int] = None  # None selects default deterministic spawn
    waypoint_resolution_m: float = 2.0       # Distance between consecutive path waypoints
    route_length_m: float = 150.0            # Total target route distance
    goal_tolerance_m: float = 3.0            # Distance to final waypoint to register success
    target_speed_kmh: float = 30.0           # Desired nominal cruise speed


@dataclass
class RewardConfig:
    """
    Shared reward function weights and penalties.
    All weights are named parameters to ensure full reproducibility and prevent magic numbers.
    """
    # Positive incentives
    w_progress: float = 1.0        # Reward for forward progress along the planned route (m/step)
    w_speed: float = 0.3           # Reward for tracking the desired target speed (30 km/h)
    goal_bonus: float = 100.0      # Terminal bonus for reaching the route goal safely

    # Negative penalties
    w_lateral: float = -0.4        # Penalty for lateral distance from the lane centerline
    w_heading: float = -0.4        # Penalty for heading misalignment with the road direction
    w_action_smoothness: float = -0.05  # Penalty for abrupt steering and jerk
    stationary_penalty: float = -0.1    # Small step penalty to discourage idling/stopping
    collision_penalty: float = -100.0   # Terminal penalty on collision with vehicles or static geometry
    out_of_lane_penalty: float = -50.0  # Terminal penalty for leaving the drivable lane


@dataclass
class EnvConfig:
    """Comprehensive environment configuration combining all sub-configs."""
    max_episode_steps: int = 500  # 500 steps * 0.05s = 25.0 seconds per episode
    seed: int = 42
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Logging and output paths
    models_dir: str = "models"
    results_dir: str = "results"
    tensorboard_dir: str = "tensorboard_logs"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        d = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if "observation" in d:
            if "low" in d["observation"] and isinstance(d["observation"]["low"], np.ndarray):
                d["observation"]["low"] = d["observation"]["low"].tolist()
            if "high" in d["observation"] and isinstance(d["observation"]["high"], np.ndarray):
                d["observation"]["high"] = d["observation"]["high"].tolist()
        return d

    def save_json(self, filepath: str) -> None:
        """Save configuration dictionary to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvConfig":
        """Reconstruct EnvConfig from a dictionary."""
        sim_data = data.get("simulator", {})
        obs_data = data.get("observation", {})
        act_data = data.get("action", {})
        route_data = data.get("route", {})
        reward_data = data.get("reward", {})

        # Rebuild numpy arrays if present as lists
        if "low" in obs_data and isinstance(obs_data["low"], list):
            obs_data["low"] = np.array(obs_data["low"], dtype=np.float32)
        if "high" in obs_data and isinstance(obs_data["high"], list):
            obs_data["high"] = np.array(obs_data["high"], dtype=np.float32)

        return cls(
            max_episode_steps=data.get("max_episode_steps", 500),
            seed=data.get("seed", 42),
            simulator=SimulatorConfig(**sim_data),
            observation=ObservationConfig(**obs_data),
            action=ActionConfig(**act_data),
            route=RouteConfig(**route_data),
            reward=RewardConfig(**reward_data),
            models_dir=data.get("models_dir", "models"),
            results_dir=data.get("results_dir", "results"),
            tensorboard_dir=data.get("tensorboard_dir", "tensorboard_logs"),
        )


@dataclass
class EvaluationConfig:
    """Standardized evaluation protocol parameters across all algorithms."""
    n_eval_episodes: int = 10
    deterministic: bool = True
    eval_seeds: Tuple[int, ...] = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010)
    route_ids: Tuple[int, ...] = (0, 1, 2)
    output_csv_name: str = "evaluation_summary.csv"
    results_dir: str = "results"


# Global default configuration instance
CarlaConfig = EnvConfig()
