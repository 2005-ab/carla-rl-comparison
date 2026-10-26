"""
common/reward.py

Shared, deterministic reward calculation for CARLA autonomous driving RL experiments.
Used identically across SAC, TD3, and PPO to ensure fair benchmark comparisons.

Reward formulation:
  R_step = R_progress + R_speed + R_lateral + R_heading + R_smoothness + R_stationary
  R_terminal = R_collision + R_goal + R_out_of_lane
  R_total = R_step + R_terminal

Design principles:
  1. Purely deterministic and finite (strictly checked for NaN/Inf).
  2. Exploitation mitigation:
     - Stationary vehicles are penalized each step (R_stationary) and receive zero progress reward.
     - Progress is measured as net positive forward displacement along the planned waypoint route.
     - Moving backwards yields negative progress reward (clamped to prevent runaway penalties).
  3. All weights are configurable through `RewardConfig` and logged individually.
  4. Baseline weights are experimental starting points and should be tuned systematically.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import numpy as np

from common.config import RewardConfig, RouteConfig


class CarlaRewardCalculator:
    """
    Computes step-level and terminal rewards for autonomous driving.
    Maintains internal step diagnostics to support interpretability and ablation studies.
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        route_config: Optional[RouteConfig] = None,
    ):
        self.reward_config = reward_config or RewardConfig()
        self.route_config = route_config or RouteConfig()

    def compute_reward(
        self,
        progress_m: float,
        speed_kmh: float,
        lateral_deviation_m: float,
        heading_error_rad: float,
        action: np.ndarray,
        previous_action: Optional[np.ndarray],
        collision: bool,
        goal_reached: bool,
        out_of_lane: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates total scalar reward and decomposes it into named diagnostic components.

        Parameters
        ----------
        progress_m : float
            Net forward displacement along the route during the current time step (in meters).
            Positive indicates moving toward the goal; negative indicates moving backward.
        speed_kmh : float
            Current vehicle forward speed in km/h.
        lateral_deviation_m : float
            Signed lateral distance from the vehicle center to the lane centerline (in meters).
        heading_error_rad : float
            Angular difference between vehicle yaw and road tangent in radians (in [-pi, pi]).
        action : np.ndarray
            Current normalized action array [steer, throttle, brake] in [-1.0, 1.0].
        previous_action : Optional[np.ndarray]
            Action executed at the previous time step, used to penalize control jerk.
        collision : bool
            True if collision sensor registered an impact.
        goal_reached : bool
            True if vehicle is within the goal tolerance of the final waypoint.
        out_of_lane : bool
            True if vehicle lateral deviation exceeds the drivable lane boundary.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            - total_reward: float (strictly finite)
            - components: Dict mapping component names to their scalar contributions
        """
        cfg = self.reward_config
        target_speed = self.route_config.target_speed_kmh

        # -------------------------------------------------------------
        # 1. Forward Route Progress Reward
        # -------------------------------------------------------------
        # Positive reward proportional to forward distance along the route.
        # Moving backwards is penalized up to -1.0 per meter.
        clipped_progress = float(np.clip(progress_m, -2.0, 5.0))
        r_progress = cfg.w_progress * clipped_progress

        # -------------------------------------------------------------
        # 2. Desired Speed Tracking Reward
        # -------------------------------------------------------------
        # Peak reward at target speed (e.g. 30 km/h).
        # Smoothly decays as vehicle speeds up too much or crawls too slowly.
        speed_ratio = float(abs(speed_kmh - target_speed) / max(target_speed, 1e-3))
        # speed_factor in [-1.0, 1.0]: 1.0 when speed == target_speed, 0.0 when speed == 0 or 2*target
        speed_factor = float(np.clip(1.0 - speed_ratio, -1.0, 1.0))
        r_speed = cfg.w_speed * speed_factor

        # -------------------------------------------------------------
        # 3. Lateral Deviation Penalty
        # -------------------------------------------------------------
        # Quadratic penalty for drifting away from lane center to encourage centered driving.
        abs_lat = float(abs(lateral_deviation_m))
        r_lateral = cfg.w_lateral * (abs_lat ** 1.5)

        # -------------------------------------------------------------
        # 4. Heading Error Penalty
        # -------------------------------------------------------------
        # Penalizes orientation mismatch with the road direction.
        abs_heading = float(abs(heading_error_rad))
        r_heading = cfg.w_heading * abs_heading

        # -------------------------------------------------------------
        # 5. Control Smoothness / Jerk Penalty
        # -------------------------------------------------------------
        # Penalizes rapid changes in steering and brake to promote smooth driving policies.
        if previous_action is not None and len(previous_action) >= 3 and len(action) >= 3:
            steer_diff = float(action[0] - previous_action[0])
            brake_diff = float(action[2] - previous_action[2])
            smoothness_cost = (steer_diff ** 2) + (0.5 * (brake_diff ** 2))
        else:
            smoothness_cost = 0.0
        r_smoothness = cfg.w_action_smoothness * smoothness_cost

        # -------------------------------------------------------------
        # 6. Stationary Penalty
        # -------------------------------------------------------------
        # Constant negative cost if the vehicle is stopped or moving slower than 1.0 km/h,
        # preventing the policy from freezing to avoid deviation penalties.
        if speed_kmh < 1.0:
            r_stationary = cfg.stationary_penalty
        else:
            r_stationary = 0.0

        # -------------------------------------------------------------
        # 7. Terminal Event Penalties and Bonuses
        # -------------------------------------------------------------
        r_collision = cfg.collision_penalty if collision else 0.0
        r_out_of_lane = cfg.out_of_lane_penalty if (out_of_lane and not collision) else 0.0
        r_goal = cfg.goal_bonus if goal_reached else 0.0

        # -------------------------------------------------------------
        # Aggregate and Validate
        # -------------------------------------------------------------
        total_reward = (
            r_progress
            + r_speed
            + r_lateral
            + r_heading
            + r_smoothness
            + r_stationary
            + r_collision
            + r_out_of_lane
            + r_goal
        )

        # Ensure finite value (protect against numerical anomalies)
        if not math.isfinite(total_reward):
            total_reward = -100.0

        components = {
            "r_progress": float(r_progress),
            "r_speed": float(r_speed),
            "r_lateral": float(r_lateral),
            "r_heading": float(r_heading),
            "r_smoothness": float(r_smoothness),
            "r_stationary": float(r_stationary),
            "r_collision": float(r_collision),
            "r_out_of_lane": float(r_out_of_lane),
            "r_goal": float(r_goal),
            "total_reward": float(total_reward),
        }

        return float(total_reward), components
