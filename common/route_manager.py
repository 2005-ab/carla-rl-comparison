"""
common/route_manager.py

Route generation, progress tracking, and spatial error computation for CARLA RL driving.
Maintains a continuous sequence of road waypoints, calculates signed lateral lane deviation,
heading misalignment, net forward progress, and goal arrival metrics.

Supports two operational modes:
  1. Live CARLA Map mode: Generates topological routes using CARLA's `map.get_waypoint()`
     and `waypoint.next(step_distance)` graph.
  2. Standalone / Mock mode: Generates deterministic reference trajectories for unit testing,
     validation, and simulation smoke tests without requiring a running CARLA server.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import math
import numpy as np

from common.config import RouteConfig


@dataclass
class Waypoint2D:
    """Lightweight 2D waypoint representation with position, heading, and lane metadata."""
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0  # Heading in radians (-pi to pi)
    lane_width: float = 3.5

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance in the XY ground plane."""
        return math.hypot(self.x - x, self.y - y)


class RouteManager:
    """
    Manages planned route waypoints, monitors vehicle tracking, and computes navigation errors.
    """

    def __init__(self, route_config: Optional[RouteConfig] = None):
        self.config = route_config or RouteConfig()
        self.waypoints: List[Waypoint2D] = []
        self.total_route_length: float = 0.0
        self.current_waypoint_idx: int = 0
        self.previous_distance_to_goal: Optional[float] = None
        self.cumulative_distance_traveled: float = 0.0
        self.last_vehicle_pos: Optional[Tuple[float, float]] = None

    def set_waypoints(self, waypoints: List[Waypoint2D]) -> None:
        """Explicitly set a pre-defined list of waypoints and compute route length."""
        if len(waypoints) < 2:
            raise ValueError("Route must contain at least 2 waypoints.")
        self.waypoints = waypoints
        self._compute_total_length()
        self.reset()

    def generate_mock_route(
        self,
        start_x: float = 0.0,
        start_y: float = 0.0,
        start_yaw: float = 0.0,
        length_m: Optional[float] = None,
        resolution_m: Optional[float] = None,
        curvature: float = 0.0,
    ) -> List[Waypoint2D]:
        """
        Generates a synthetic reference path for unit testing and offline verification.

        Parameters
        ----------
        start_x, start_y : float
            Starting Cartesian position in meters.
        start_yaw : float
            Initial heading angle in radians.
        length_m : float, optional
            Total route length in meters (defaults to config.route_length_m).
        resolution_m : float, optional
            Spacing between consecutive waypoints (defaults to config.waypoint_resolution_m).
        curvature : float
            Yaw change rate in radians per meter (0.0 generates a straight road).
        """
        length = length_m or self.config.route_length_m
        res = resolution_m or self.config.waypoint_resolution_m
        num_points = max(2, int(math.ceil(length / res)) + 1)

        waypoints: List[Waypoint2D] = []
        x, y, yaw = start_x, start_y, start_yaw

        for _ in range(num_points):
            waypoints.append(Waypoint2D(x=x, y=y, z=0.0, yaw=yaw, lane_width=3.5))
            x += res * math.cos(yaw)
            y += res * math.sin(yaw)
            yaw = (yaw + (curvature * res) + math.pi) % (2 * math.pi) - math.pi

        self.set_waypoints(waypoints)
        return self.waypoints

    def build_from_carla_map(
        self,
        carla_map: Any,
        spawn_transform: Any,
        target_length: Optional[float] = None,
    ) -> List[Waypoint2D]:
        """
        Builds a valid forward driving route using CARLA's waypoint topology.

        Parameters
        ----------
        carla_map : carla.Map
            The CARLA world map object.
        spawn_transform : carla.Transform
            Vehicle starting transform.
        target_length : float, optional
            Desired length of the route in meters.
        """
        res = self.config.waypoint_resolution_m
        target_len = target_length or self.config.route_length_m

        # Import carla locally so this module stays importable without a server.
        import carla  # noqa: WPS433 (deferred import for offline compatibility)

        current_wp = carla_map.get_waypoint(
            spawn_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        waypoints: List[Waypoint2D] = []
        accumulated_length = 0.0

        # Add initial starting point
        loc = current_wp.transform.location
        rot = current_wp.transform.rotation
        yaw_rad = math.radians(rot.yaw)
        waypoints.append(Waypoint2D(x=loc.x, y=loc.y, z=loc.z, yaw=yaw_rad, lane_width=current_wp.lane_width))

        while accumulated_length < target_len:
            next_wps = current_wp.next(res)
            if not next_wps:
                # Reached road dead-end; stop extending
                break

            # If there's a junction / fork, pick the one with closest heading to maintain straight path
            if len(next_wps) == 1:
                next_wp = next_wps[0]
            else:
                curr_yaw = current_wp.transform.rotation.yaw
                next_wp = min(
                    next_wps,
                    key=lambda wp: abs((wp.transform.rotation.yaw - curr_yaw + 180.0) % 360.0 - 180.0),
                )

            loc = next_wp.transform.location
            rot = next_wp.transform.rotation
            yaw_rad = math.radians(rot.yaw)

            step_dist = waypoints[-1].distance_to(loc.x, loc.y)
            accumulated_length += step_dist

            waypoints.append(Waypoint2D(x=loc.x, y=loc.y, z=loc.z, yaw=yaw_rad, lane_width=next_wp.lane_width))
            current_wp = next_wp

        if len(waypoints) < 2:
            # Fallback if map lookup produced insufficient waypoints
            return self.generate_mock_route(
                start_x=spawn_transform.location.x,
                start_y=spawn_transform.location.y,
                start_yaw=math.radians(spawn_transform.rotation.yaw),
                length_m=target_len,
            )

        self.set_waypoints(waypoints)
        return self.waypoints

    def _compute_total_length(self) -> None:
        """Calculates cumulative route length across all waypoint segments."""
        self.total_route_length = 0.0
        for i in range(len(self.waypoints) - 1):
            self.total_route_length += self.waypoints[i].distance_to(
                self.waypoints[i + 1].x, self.waypoints[i + 1].y
            )

    def reset(self) -> None:
        """Reset route tracking state at the beginning of an episode."""
        self.current_waypoint_idx = 0
        self.previous_distance_to_goal = None
        self.cumulative_distance_traveled = 0.0
        self.last_vehicle_pos = None

    def update_vehicle_position(
        self,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw_rad: float,
    ) -> Tuple[float, float, float, float, float, bool]:
        """
        Updates tracking against vehicle position and computes navigation metrics.

        Parameters
        ----------
        vehicle_x, vehicle_y : float
            Current vehicle coordinates in meters.
        vehicle_yaw_rad : float
            Current vehicle heading in radians [-pi, pi].

        Returns
        -------
        Tuple containing:
          - progress_m: Net forward displacement along route in meters since last step
          - lateral_deviation_m: Signed distance from lane centerline (negative = left, positive = right)
          - heading_error_rad: Difference between vehicle yaw and road tangent in radians [-pi, pi]
          - distance_to_goal: Euclidean distance to final destination waypoint
          - route_completion_pct: Percentage of total route successfully traversed [0.0, 100.0]
          - goal_reached: True if within goal_tolerance_m of the destination
        """
        if len(self.waypoints) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, True

        # Advance current waypoint index monotonically within a forward search window
        # (prevent snapping backwards while tolerating small tracking deviations)
        search_start = self.current_waypoint_idx
        search_end = min(len(self.waypoints) - 1, search_start + 10)

        best_idx = search_start
        min_dist = float("inf")
        for i in range(search_start, search_end + 1):
            d = self.waypoints[i].distance_to(vehicle_x, vehicle_y)
            if d < min_dist:
                min_dist = d
                best_idx = i

        self.current_waypoint_idx = best_idx

        # Select reference segment [wp_ref, wp_next]
        if self.current_waypoint_idx < len(self.waypoints) - 1:
            wp_a = self.waypoints[self.current_waypoint_idx]
            wp_b = self.waypoints[self.current_waypoint_idx + 1]
        else:
            wp_a = self.waypoints[max(0, len(self.waypoints) - 2)]
            wp_b = self.waypoints[-1]

        # 1. Road tangent vector and signed lateral deviation
        dx = wp_b.x - wp_a.x
        dy = wp_b.y - wp_a.y
        seg_len = math.hypot(dx, dy)

        if seg_len > 1e-4:
            # 2D cross product: (P - A) x (B - A)
            rx = vehicle_x - wp_a.x
            ry = vehicle_y - wp_a.y
            # In CARLA coordinates (X forward, Y right):
            # cross_product > 0 when vehicle is to the right of the path segment
            cross = rx * dy - ry * dx
            lateral_deviation_m = float(cross / seg_len)
            road_yaw = math.atan2(dy, dx)
        else:
            lateral_deviation_m = 0.0
            road_yaw = wp_a.yaw

        # 2. Heading error normalized to [-pi, pi]
        heading_diff = vehicle_yaw_rad - road_yaw
        heading_error_rad = float((heading_diff + math.pi) % (2.0 * math.pi) - math.pi)

        # 3. Distance to final destination waypoint
        goal_wp = self.waypoints[-1]
        distance_to_goal = float(goal_wp.distance_to(vehicle_x, vehicle_y))

        # 4. Step forward progress calculation
        if self.previous_distance_to_goal is None:
            progress_m = 0.0
        else:
            # Reduction in distance to goal constitutes positive progress
            progress_m = float(self.previous_distance_to_goal - distance_to_goal)

        self.previous_distance_to_goal = distance_to_goal

        # 5. Route completion percentage
        # Traversed distance along waypoints up to current index + projection
        if self.total_route_length > 1e-3:
            completion = max(0.0, 1.0 - (distance_to_goal / self.total_route_length))
            route_completion_pct = float(np.clip(completion * 100.0, 0.0, 100.0))
        else:
            route_completion_pct = 100.0

        # 6. Goal reached condition
        goal_reached = bool(distance_to_goal <= self.config.goal_tolerance_m)

        self.last_vehicle_pos = (vehicle_x, vehicle_y)

        return (
            progress_m,
            lateral_deviation_m,
            heading_error_rad,
            distance_to_goal,
            route_completion_pct,
            goal_reached,
        )
