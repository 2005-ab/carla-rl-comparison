"""
tests/test_environment_api.py

Tests for the Gymnasium environment API and integration correctness.

Two tiers:
  1. OFFLINE unit tests (default, no CARLA required): construct the env with
     connect_on_init=False and assert spaces, reset/step contracts, action
     mapping, termination logic, determinism, and gymnasium check_env.
  2. A live CARLA integration test marked @pytest.mark.carla_server that is
     skipped automatically when no simulator / carla package is available.

We never pretend an offline run validates CARLA integration; the live test is
the only thing that does, and it is explicitly separate and skipped locally.
"""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from common.carla_env import CarlaConnectionError, CarlaDrivingEnv
from common.config import EnvConfig
from common.route_manager import Waypoint2D


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def offline_env():
    """An env constructed WITHOUT connecting to a CARLA server."""
    env = CarlaDrivingEnv(config=EnvConfig(), connect_on_init=False)
    try:
        yield env
    finally:
        env.close()


# --------------------------------------------------------------------------
# Spaces
# --------------------------------------------------------------------------

def test_observation_space_contract(offline_env):
    obs_space = offline_env.observation_space
    assert obs_space.shape == (5,)
    assert obs_space.dtype == np.float32
    np.testing.assert_allclose(obs_space.low, np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(obs_space.high, np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_action_space_contract(offline_env):
    act_space = offline_env.action_space
    assert act_space.shape == (3,)
    assert act_space.dtype == np.float32
    assert float(act_space.low.min()) == -1.0
    assert float(act_space.high.max()) == 1.0


# --------------------------------------------------------------------------
# reset / step contracts
# --------------------------------------------------------------------------

def test_reset_returns_obs_and_info(offline_env):
    obs, info = offline_env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert offline_env.observation_space.contains(obs)
    assert isinstance(info, dict)
    for key in ("step_count", "speed_kmh", "lateral_deviation_m", "distance_to_goal_m",
                "route_completion_pct", "collision", "goal_reached", "out_of_lane",
                "reward_components"):
        assert key in info


def test_step_returns_five_tuple_with_correct_types(offline_env):
    offline_env.reset(seed=0)
    action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    result = offline_env.step(action)
    assert len(result) == 5, "step must return (obs, reward, terminated, truncated, info)"
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray) and obs.dtype == np.float32 and obs.shape == (5,)
    assert offline_env.observation_space.contains(obs)
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert np.isfinite(info["distance_to_goal_m"])


def test_observation_within_bounds_after_many_steps(offline_env):
    offline_env.reset(seed=1)
    for _ in range(20):
        action = offline_env.action_space.sample()
        obs, reward, terminated, truncated, info = offline_env.step(action)
        assert offline_env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_reset_is_deterministic_with_seed(offline_env):
    """Two resets with the same seed produce identical observations."""
    obs_a, _ = offline_env.reset(seed=123)
    obs_b, _ = offline_env.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)


def test_close_does_not_raise_offline(offline_env):
    offline_env.close()
    assert offline_env.is_connected is False


# --------------------------------------------------------------------------
# Action mapping (safety interlock)
# --------------------------------------------------------------------------

def test_action_mapping_clamps_to_range(offline_env):
    steer, throttle, brake = offline_env._map_action_to_control(
        np.array([2.0, 2.0, 2.0], dtype=np.float32))
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0


def test_throttle_zeroed_when_braking(offline_env):
    # brake above deadzone -> throttle forced to zero (mutual exclusivity).
    steer, throttle, brake = offline_env._map_action_to_control(
        np.array([0.0, 1.0, 0.5], dtype=np.float32))
    assert brake == pytest.approx(0.5)
    assert throttle == 0.0


def test_throttle_allowed_when_not_braking(offline_env):
    steer, throttle, brake = offline_env._map_action_to_control(
        np.array([0.0, 0.8, 0.0], dtype=np.float32))
    assert throttle == pytest.approx(0.8)
    assert brake == 0.0


def test_negative_actions_map_to_zero_throttle_brake(offline_env):
    # Negative throttle/brake inputs provide no positive command.
    _, throttle, brake = offline_env._map_action_to_control(
        np.array([0.0, -1.0, -1.0], dtype=np.float32))
    assert throttle == 0.0
    assert brake == 0.0


# --------------------------------------------------------------------------
# Termination logic
# --------------------------------------------------------------------------

def test_collision_terminates_episode(offline_env):
    offline_env.reset(seed=0)
    offline_env.has_collided = True  # simulate collision sensor firing
    _, _, terminated, _, info = offline_env.step(np.zeros(3, dtype=np.float32))
    assert terminated is True
    assert info["collision"] is True


def test_truncation_at_max_steps(offline_env):
    offline_env.config.max_episode_steps = 3
    offline_env.reset(seed=0)
    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = offline_env.step(np.zeros(3, dtype=np.float32))
    assert truncated is True


def test_goal_reached_terminates(offline_env):
    """Place the route goal right at the stationary vehicle -> goal_reached."""
    offline_env.reset(seed=0)
    # Short route whose endpoint is within goal tolerance of the origin.
    offline_env.route_manager.set_waypoints([
        Waypoint2D(x=0.0, y=0.0, yaw=0.0),
        Waypoint2D(x=0.5, y=0.0, yaw=0.0),
    ])
    _, _, terminated, _, info = offline_env.step(np.zeros(3, dtype=np.float32))
    assert info["goal_reached"] is True
    assert terminated is True


def test_out_of_lane_terminates(offline_env):
    """Route offset laterally so the stationary vehicle is off-lane."""
    offline_env.reset(seed=0)
    # Line at y=3.0; vehicle at origin sits 3.0 m off -> out of lane (>2.5).
    offline_env.route_manager.set_waypoints([
        Waypoint2D(x=0.0, y=3.0, yaw=0.0),
        Waypoint2D(x=10.0, y=3.0, yaw=0.0),
    ])
    _, _, terminated, _, info = offline_env.step(np.zeros(3, dtype=np.float32))
    assert info["out_of_lane"] is True
    assert terminated is True


# --------------------------------------------------------------------------
# Gymnasium compliance (offline)
# --------------------------------------------------------------------------

def test_check_env_passes_offline():
    """gymnasium.check_env on the offline (connect_on_init=False) env.

    The env supports connect_on_init=False precisely so the API can be
    validated without a running CARLA server. This does NOT validate CARLA
    integration; the live test below does that.
    """
    env = CarlaDrivingEnv(config=EnvConfig(), connect_on_init=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_constructing_without_server_raises_clear_error():
    """When connect_on_init=True and no server/package is available, raise clearly."""
    with pytest.raises(CarlaConnectionError):
        CarlaDrivingEnv(config=EnvConfig(), connect_on_init=True)


def test_env_is_subclass_of_gym_env():
    import gymnasium as gym
    assert issubclass(CarlaDrivingEnv, gym.Env)


# --------------------------------------------------------------------------
# Live CARLA integration (skipped unless a server is available)
# --------------------------------------------------------------------------

@pytest.mark.carla_server
def test_live_carla_reset_step(carla_env_live):
    """Real-server smoke test: connect, reset, step, verify finite telemetry."""
    assert carla_env_live.is_connected is True
    obs, info = carla_env_live.reset(seed=0)
    assert carla_env_live.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    assert isinstance(info, dict)

    for _ in range(10):
        action = carla_env_live.action_space.sample()
        obs, reward, terminated, truncated, info = carla_env_live.step(action)
        assert carla_env_live.observation_space.contains(obs)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        if terminated or truncated:
            break
