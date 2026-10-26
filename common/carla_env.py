"""
common/carla_env.py

Shared Gymnasium Reinforcement Learning environment for CARLA autonomous driving.
Provides standard Gymnasium API compliance (reset, step, close) with a 5D numerical
observation space and 3D continuous action space.

Used identically for SAC, TD3, and PPO algorithms.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from common.config import CarlaConfig, EnvConfig
from common.reward import CarlaRewardCalculator
from common.route_manager import RouteManager, Waypoint2D


class CarlaConnectionError(RuntimeError):
    """Raised when CARLA client fails to connect or CARLA package is missing."""
    pass


class CarlaDrivingEnv(gym.Env):
    """
    Gymnasium environment for autonomous driving in CARLA.

    Observation Space
    -----------------
    Box(shape=(5,), dtype=np.float32)
      Index 0: Normalized speed in [0.0, 1.0] (speed_kmh / max_speed_kmh, clamped)
      Index 1: Normalized lateral lane deviation in [-1.0, 1.0] (lateral_dev / max_lateral_dev)
               Negative = vehicle is left of centerline, Positive = right of centerline
      Index 2: Normalized heading error in [-1.0, 1.0] (heading_err_rad / pi)
               Negative = vehicle points left of tangent, Positive = points right
      Index 3: Normalized distance to route goal in [0.0, 1.0] (dist_to_goal / max_goal_dist)
      Index 4: Normalized distance to nearest detected vehicle in [0.0, 1.0]
               1.0 = clear road (no vehicle within max_obstacle_dist); 0.0 = contact

    Action Space
    ------------
    Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
      Index 0: Steering command in [-1.0, 1.0]
      Index 1: Raw throttle command in [-1.0, 1.0] (positive values provide throttle [0.0, 1.0])
      Index 2: Raw brake command in [-1.0, 1.0] (positive values provide braking [0.0, 1.0])

    Safety Interlock:
      If brake > 0.05, throttle is strictly zeroed to prevent simultaneous acceleration and braking.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        connect_on_init: bool = True,
    ):
        super().__init__()
        self.config = config or CarlaConfig

        # Define Observation Space
        obs_cfg = self.config.observation
        self.observation_space = spaces.Box(
            low=obs_cfg.low,
            high=obs_cfg.high,
            shape=(obs_cfg.obs_dim,),
            dtype=np.float32,
        )

        # Define Action Space
        act_cfg = self.config.action
        self.action_space = spaces.Box(
            low=act_cfg.low,
            high=act_cfg.high,
            shape=(act_cfg.action_dim,),
            dtype=np.float32,
        )

        # Helpers
        self.reward_calculator = CarlaRewardCalculator(
            reward_config=self.config.reward,
            route_config=self.config.route,
        )
        self.route_manager = RouteManager(route_config=self.config.route)

        # CARLA Handles
        self.client: Any = None
        self.world: Any = None
        self.carla_map: Any = None
        self.hero_vehicle: Any = None
        self.collision_sensor: Any = None
        self.original_settings: Any = None
        self._actor_cleanup_list: List[Any] = []

        # Episode Tracking State
        self.step_count: int = 0
        self.has_collided: bool = False
        self.collision_actor_id: Optional[int] = None
        self.previous_action: Optional[np.ndarray] = None
        self.is_connected: bool = False

        if connect_on_init:
            self._connect_to_carla()

    def _connect_to_carla(self) -> None:
        """Establishes connection to the CARLA simulator server."""
        sim_cfg = self.config.simulator
        try:
            import carla
        except ImportError:
            raise CarlaConnectionError(
                "The 'carla' Python module is not installed.\n"
                "To install CARLA Python API matching your simulator version:\n"
                "  - pip install carla==0.9.15\n"
                "Or append the CARLA .whl/.egg path to your PYTHONPATH."
            )

        try:
            self.client = carla.Client(sim_cfg.host, sim_cfg.port)
            self.client.set_timeout(sim_cfg.timeout)
            self.world = self.client.get_world()
            self.carla_map = self.world.get_map()

            # Save initial world settings to restore on close
            self.original_settings = self.world.get_settings()

            if sim_cfg.synchronous_mode:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = sim_cfg.fixed_delta_seconds
                self.world.apply_settings(settings)

            self.is_connected = True
        except Exception as e:
            raise CarlaConnectionError(
                f"Failed to connect to CARLA simulator at {sim_cfg.host}:{sim_cfg.port}.\n"
                f"Error details: {e}\n"
                "Troubleshooting:\n"
                "  1. Verify CARLA simulator is actively running.\n"
                "  2. Check host and port (default is localhost:2000).\n"
                "  3. On remote/Colab servers, configure SSH tunneling or port forwarding."
            ) from e

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the driving environment for a new episode.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Initial observation vector and diagnostic info dictionary.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.has_collided = False
        self.collision_actor_id = None
        self.previous_action = None

        if self.is_connected:
            self._cleanup_actors()
            self._spawn_hero_and_sensors()
            self._setup_episode_route()
            self.world.tick()
            x, y, yaw_rad, speed_kmh = self._get_vehicle_telemetry()
            obstacle_dist = self._find_nearest_obstacle_distance()
        else:
            # Offline / mock initialization: generate a deterministic reference route
            # and treat the vehicle as stationary at the route origin.
            self.route_manager.generate_mock_route()
            x, y, yaw_rad, speed_kmh = 0.0, 0.0, 0.0, 0.0
            obstacle_dist = self.config.observation.max_obstacle_distance_m

        # Compute initial route metrics once (consistent obs + info)
        (
            _,
            lateral_dev,
            heading_err_rad,
            dist_to_goal,
            route_completion_pct,
            goal_reached,
        ) = self.route_manager.update_vehicle_position(x, y, yaw_rad)

        obs = self._build_observation(
            speed_kmh=speed_kmh,
            lateral_dev=lateral_dev,
            heading_err_rad=heading_err_rad,
            dist_to_goal=dist_to_goal,
            obstacle_dist=obstacle_dist,
        )

        info = self._get_info(
            progress_m=0.0,
            lateral_dev=lateral_dev,
            heading_err_rad=heading_err_rad,
            dist_to_goal=dist_to_goal,
            route_completion_pct=route_completion_pct,
            goal_reached=goal_reached,
            out_of_lane=False,
            reward_components={},
            speed_kmh=speed_kmh,
        )
        return obs, info

    def _cleanup_actors(self) -> None:
        """Destroys all spawned CARLA actors safely."""
        for actor in self._actor_cleanup_list:
            if actor is not None and getattr(actor, "is_alive", False):
                try:
                    actor.destroy()
                except Exception:
                    pass
        self._actor_cleanup_list.clear()
        self.hero_vehicle = None
        self.collision_sensor = None

    def _spawn_hero_and_sensors(self) -> None:
        """Spawns the ego-vehicle and attaches collision sensor."""
        blueprint_lib = self.world.get_blueprint_library()
        bp = blueprint_lib.find(self.config.simulator.vehicle_blueprint)
        bp.set_attribute("role_name", self.config.simulator.hero_role_name)

        spawn_points = self.carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found on CARLA map.")

        spawn_idx = self.config.route.spawn_point_index
        if spawn_idx is not None and spawn_idx < len(spawn_points):
            spawn_transform = spawn_points[spawn_idx]
        else:
            spawn_transform = spawn_points[0]

        # Spawn ego vehicle
        self.hero_vehicle = self.world.try_spawn_actor(bp, spawn_transform)
        if self.hero_vehicle is None:
            # Try alternate spawn points if primary was blocked
            for pt in spawn_points[:10]:
                self.hero_vehicle = self.world.try_spawn_actor(bp, pt)
                if self.hero_vehicle is not None:
                    spawn_transform = pt
                    break

        if self.hero_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle in CARLA world.")

        self._actor_cleanup_list.append(self.hero_vehicle)

        # Attach collision sensor
        col_bp = blueprint_lib.find("sensor.other.collision")
        import carla
        col_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
        self.collision_sensor = self.world.spawn_actor(
            col_bp, col_transform, attach_to=self.hero_vehicle
        )
        self._actor_cleanup_list.append(self.collision_sensor)
        self.collision_sensor.listen(self._on_collision)

    def _on_collision(self, event: Any) -> None:
        """Collision sensor callback."""
        self.has_collided = True
        other_actor = getattr(event, "other_actor", None)
        self.collision_actor_id = getattr(other_actor, "id", None)

    def _setup_episode_route(self) -> None:
        """Builds route waypoints for the current episode."""
        transform = self.hero_vehicle.get_transform()
        self.route_manager.build_from_carla_map(
            carla_map=self.carla_map,
            spawn_transform=transform,
            target_length=self.config.route.route_length_m,
        )

    def _map_action_to_control(self, action: np.ndarray) -> Tuple[float, float, float]:
        """
        Maps continuous RL action vector [-1.0, 1.0]^3 to CARLA VehicleControl parameters.

        Steering:
          steer in [-1.0, 1.0]
        Throttle & Brake:
          throttle in [0.0, 1.0] if action[1] > 0 else 0.0
          brake in [0.0, 1.0] if action[2] > 0 else 0.0

        Safety Constraint:
          If brake > brake_deadzone (0.05), throttle is zeroed.
        """
        raw_steer = float(np.clip(action[0], -1.0, 1.0))
        raw_throttle = float(np.clip(action[1], -1.0, 1.0))
        raw_brake = float(np.clip(action[2], -1.0, 1.0))

        steer = raw_steer
        throttle = max(0.0, raw_throttle)
        brake = max(0.0, raw_brake)

        # Safety interlock: mutual exclusivity
        if brake > self.config.action.brake_deadzone:
            throttle = 0.0

        return steer, throttle, brake

    def _apply_control(self, steer: float, throttle: float, brake: float) -> None:
        """Sends control packet to CARLA vehicle."""
        if self.hero_vehicle is not None and getattr(self.hero_vehicle, "is_alive", False):
            import carla
            control = carla.VehicleControl(
                steer=steer,
                throttle=throttle,
                brake=brake,
                hand_brake=False,
                reverse=False,
            )
            self.hero_vehicle.apply_control(control)

    def _get_vehicle_telemetry(self) -> Tuple[float, float, float, float]:
        """Returns vehicle (x, y, yaw_rad, speed_kmh)."""
        if self.hero_vehicle is not None and getattr(self.hero_vehicle, "is_alive", False):
            t = self.hero_vehicle.get_transform()
            vel = self.hero_vehicle.get_velocity()
            speed_ms = math.hypot(vel.x, vel.y)
            speed_kmh = float(speed_ms * 3.6)
            yaw_rad = float(math.radians(t.rotation.yaw))
            return float(t.location.x), float(t.location.y), yaw_rad, speed_kmh
        return 0.0, 0.0, 0.0, 0.0

    def _find_nearest_obstacle_distance(self) -> float:
        """Computes distance to nearest vehicle obstacle in meters."""
        if not self.is_connected or self.hero_vehicle is None:
            return self.config.observation.max_obstacle_distance_m

        ego_loc = self.hero_vehicle.get_transform().location
        ego_id = self.hero_vehicle.id
        actors = self.world.get_actors().filter("vehicle.*")

        min_dist = self.config.observation.max_obstacle_distance_m
        for actor in actors:
            if actor.id != ego_id:
                loc = actor.get_transform().location
                d = math.hypot(loc.x - ego_loc.x, loc.y - ego_loc.y)
                if d < min_dist:
                    min_dist = d

        return float(min_dist)

    def _build_observation(
        self,
        speed_kmh: float,
        lateral_dev: float,
        heading_err_rad: float,
        dist_to_goal: float,
        obstacle_dist: float,
    ) -> np.ndarray:
        """Normalizes raw telemetry into the fixed 5D float32 observation vector.

        Pure function: no CARLA calls and no route-manager side effects, so it is
        safe to call in both live and offline (mock) modes.
        """
        obs_cfg = self.config.observation

        # Feature 0: Normalized speed in [0.0, 1.0]
        norm_speed = float(np.clip(speed_kmh / obs_cfg.max_speed_kmh, 0.0, 1.0))

        # Feature 1: Normalized lateral deviation in [-1.0, 1.0]
        norm_lateral = float(
            np.clip(lateral_dev / obs_cfg.max_lateral_deviation_m, -1.0, 1.0)
        )

        # Feature 2: Normalized heading error in [-1.0, 1.0]
        norm_heading = float(np.clip(heading_err_rad / math.pi, -1.0, 1.0))

        # Feature 3: Normalized distance to goal in [0.0, 1.0]
        norm_dist_goal = float(
            np.clip(dist_to_goal / obs_cfg.max_goal_distance_m, 0.0, 1.0)
        )

        # Feature 4: Normalized distance to obstacle in [0.0, 1.0]
        norm_dist_obs = float(
            np.clip(obstacle_dist / obs_cfg.max_obstacle_distance_m, 0.0, 1.0)
        )

        obs = np.array(
            [norm_speed, norm_lateral, norm_heading, norm_dist_goal, norm_dist_obs],
            dtype=np.float32,
        )

        # Protect against NaN / Inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        return obs

    def _get_observation(self) -> np.ndarray:
        """Constructs the normalized 5D observation vector from live or mock telemetry.

        Calls the route manager exactly once per invocation. In offline mode the
        vehicle is treated as stationary at the route origin, so the observation
        reflects the planned route geometry rather than a simulated trajectory.
        """
        x, y, yaw_rad, speed_kmh = self._get_vehicle_telemetry()
        (
            _,
            lateral_dev,
            heading_err_rad,
            dist_to_goal,
            _,
            _,
        ) = self.route_manager.update_vehicle_position(x, y, yaw_rad)

        nearest_obstacle_dist = self._find_nearest_obstacle_distance()

        return self._build_observation(
            speed_kmh=speed_kmh,
            lateral_dev=lateral_dev,
            heading_err_rad=heading_err_rad,
            dist_to_goal=dist_to_goal,
            obstacle_dist=nearest_obstacle_dist,
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment time step.

        Parameters
        ----------
        action : np.ndarray
            3D action [steer, throttle, brake] in [-1.0, 1.0].

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Clean, clip, and map control
        clean_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        steer, throttle, brake = self._map_action_to_control(clean_action)

        if self.is_connected:
            self._apply_control(steer, throttle, brake)
            self.world.tick()
            x, y, yaw_rad, speed_kmh = self._get_vehicle_telemetry()
            obstacle_dist = self._find_nearest_obstacle_distance()
        else:
            # Standalone mock kinematic step for offline testing:
            # the vehicle is treated as stationary at the route origin so the
            # observation reflects route geometry, not a simulated trajectory.
            x, y, yaw_rad, speed_kmh = 0.0, 0.0, 0.0, 0.0
            obstacle_dist = self.config.observation.max_obstacle_distance_m

        # Update route metrics exactly once per step
        (
            progress_m,
            lateral_dev,
            heading_err_rad,
            dist_to_goal,
            route_completion_pct,
            goal_reached,
        ) = self.route_manager.update_vehicle_position(x, y, yaw_rad)

        # Termination checks
        out_of_lane = abs(lateral_dev) >= self.config.observation.max_lateral_deviation_m
        terminated = bool(self.has_collided or goal_reached or out_of_lane)
        truncated = bool(self.step_count >= self.config.max_episode_steps)

        # Reward computation
        reward, reward_components = self.reward_calculator.compute_reward(
            progress_m=progress_m,
            speed_kmh=speed_kmh,
            lateral_deviation_m=lateral_dev,
            heading_error_rad=heading_err_rad,
            action=clean_action,
            previous_action=self.previous_action,
            collision=self.has_collided,
            goal_reached=goal_reached,
            out_of_lane=out_of_lane,
        )

        self.previous_action = clean_action
        obs = self._build_observation(
            speed_kmh=speed_kmh,
            lateral_dev=lateral_dev,
            heading_err_rad=heading_err_rad,
            dist_to_goal=dist_to_goal,
            obstacle_dist=obstacle_dist,
        )

        info = self._get_info(
            progress_m=progress_m,
            lateral_dev=lateral_dev,
            heading_err_rad=heading_err_rad,
            dist_to_goal=dist_to_goal,
            route_completion_pct=route_completion_pct,
            goal_reached=goal_reached,
            out_of_lane=out_of_lane,
            reward_components=reward_components,
            speed_kmh=speed_kmh,
        )

        return obs, float(reward), terminated, truncated, info

    def _get_info(
        self,
        progress_m: float,
        lateral_dev: float,
        heading_err_rad: float,
        dist_to_goal: float,
        route_completion_pct: float,
        goal_reached: bool,
        out_of_lane: bool,
        reward_components: Dict[str, float],
        speed_kmh: float = 0.0,
    ) -> Dict[str, Any]:
        """Assembles diagnostic info dictionary."""
        return {
            "step_count": int(self.step_count),
            "speed_kmh": float(speed_kmh),
            "lateral_deviation_m": float(lateral_dev),
            "heading_error_deg": float(math.degrees(heading_err_rad)),
            "distance_to_goal_m": float(dist_to_goal),
            "route_completion_pct": float(route_completion_pct),
            "progress_m": float(progress_m),
            "collision": bool(self.has_collided),
            "goal_reached": bool(goal_reached),
            "out_of_lane": bool(out_of_lane),
            "reward_components": reward_components,
        }

    def close(self) -> None:
        """Cleans up actors and restores original CARLA simulator settings."""
        self._cleanup_actors()
        if self.is_connected and self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception:
                pass
        self.is_connected = False
