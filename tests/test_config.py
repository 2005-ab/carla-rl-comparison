"""
tests/test_config.py

Unit tests for the centralized configuration module. These run fully offline
and do not require CARLA, Gymnasium environments, or a running simulator.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from common.config import (
    ActionConfig,
    EnvConfig,
    EvaluationConfig,
    ObservationConfig,
    RewardConfig,
    RouteConfig,
    SimulatorConfig,
    CarlaConfig,
)


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

def test_default_env_config_values():
    cfg = EnvConfig()
    assert cfg.max_episode_steps == 500
    assert cfg.seed == 42
    assert isinstance(cfg.simulator, SimulatorConfig)
    assert isinstance(cfg.observation, ObservationConfig)
    assert isinstance(cfg.action, ActionConfig)
    assert isinstance(cfg.route, RouteConfig)
    assert isinstance(cfg.reward, RewardConfig)


def test_simulator_defaults():
    sim = SimulatorConfig()
    assert sim.host == "localhost"
    assert sim.port == 2000
    assert sim.town == "Town01"
    assert sim.synchronous_mode is True
    assert sim.fixed_delta_seconds == 0.05
    assert sim.vehicle_blueprint == "vehicle.tesla.model3"


def test_observation_spec_matches_contract():
    """The shared environment contract: 5-element float32 observation vector."""
    obs = ObservationConfig()
    assert obs.obs_dim == 5
    assert obs.low.shape == (5,)
    assert obs.high.shape == (5,)
    assert obs.low.dtype == np.float32
    assert obs.high.dtype == np.float32
    # Expected bounds per the documented contract.
    np.testing.assert_allclose(obs.low, np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(obs.high, np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    # low <= high elementwise (Box validity)
    assert np.all(obs.low <= obs.high)


def test_action_spec_matches_contract():
    """The shared environment contract: 3-element continuous action in [-1, 1]."""
    act = ActionConfig()
    assert act.action_dim == 3
    assert act.low == -1.0
    assert act.high == 1.0
    assert act.brake_deadzone == 0.05


def test_route_defaults():
    route = RouteConfig()
    assert route.route_id == 0
    assert route.waypoint_resolution_m == 2.0
    assert route.goal_tolerance_m == 3.0
    assert route.target_speed_kmh == 30.0
    assert route.route_length_m == 150.0


def test_reward_weights_match_documentation():
    """Guard the baseline weights so a silent edit doesn't shift the benchmark."""
    r = RewardConfig()
    assert r.w_progress == 1.0
    assert r.w_speed == 0.3
    assert r.goal_bonus == 100.0
    assert r.w_lateral == -0.4
    assert r.w_heading == -0.4
    assert r.w_action_smoothness == -0.05
    assert r.stationary_penalty == -0.1
    assert r.collision_penalty == -100.0
    assert r.out_of_lane_penalty == -50.0


def test_evaluation_config_defaults():
    ev = EvaluationConfig()
    assert ev.n_eval_episodes == 10
    assert ev.deterministic is True
    assert len(ev.eval_seeds) == 10
    assert ev.route_ids == (0, 1, 2)


def test_global_carla_config_is_env_config():
    """CarlaConfig is the module-level default instance, not a class."""
    assert isinstance(CarlaConfig, EnvConfig)
    # It must expose the same nested sub-configs.
    assert isinstance(CarlaConfig.observation, ObservationConfig)
    assert isinstance(CarlaConfig.action, ActionConfig)


# --------------------------------------------------------------------------
# Serialization round-trips
# --------------------------------------------------------------------------

def test_to_dict_from_dict_roundtrip():
    cfg = EnvConfig()
    cfg.max_episode_steps = 1234
    cfg.route.target_speed_kmh = 25.0
    cfg.reward.collision_penalty = -250.0

    d = cfg.to_dict()
    # numpy arrays must be serialized to lists for JSON.
    assert isinstance(d["observation"]["low"], list)
    assert isinstance(d["observation"]["high"], list)

    cfg2 = EnvConfig.from_dict(d)
    assert cfg2.max_episode_steps == 1234
    assert cfg2.route.target_speed_kmh == 25.0
    assert cfg2.reward.collision_penalty == -250.0
    # numpy arrays restored with correct dtype/shape.
    assert isinstance(cfg2.observation.low, np.ndarray)
    assert cfg2.observation.low.dtype == np.float32
    np.testing.assert_allclose(cfg2.observation.low, cfg.observation.low)


def test_save_json_and_reload(tmp_path):
    cfg = EnvConfig()
    out = tmp_path / "config.json"
    cfg.save_json(str(out))

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["max_episode_steps"] == 500
    assert data["observation"]["obs_dim"] == 5

    cfg2 = EnvConfig.from_dict(data)
    assert cfg2.max_episode_steps == 500
    assert cfg2.observation.obs_dim == 5


def test_from_dict_handles_missing_keys():
    """from_dict must be tolerant of partial dictionaries (use defaults)."""
    cfg = EnvConfig.from_dict({})
    assert cfg.max_episode_steps == 500
    assert cfg.simulator.port == 2000
    assert cfg.observation.obs_dim == 5


# --------------------------------------------------------------------------
# Gymnasium space construction from config
# --------------------------------------------------------------------------

def test_observation_space_bounds_from_config():
    from gymnasium import spaces
    cfg = EnvConfig()
    space = spaces.Box(low=cfg.observation.low, high=cfg.observation.high,
                       shape=(cfg.observation.obs_dim,), dtype=np.float32)
    assert space.shape == (5,)
    assert space.dtype == np.float32
    # A zero vector and the documented extremes must be contained.
    assert space.contains(np.zeros(5, dtype=np.float32))
    assert space.contains(np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32))
    assert space.contains(np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    # Out-of-range values must be rejected.
    assert not space.contains(np.array([1.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_action_space_bounds_from_config():
    from gymnasium import spaces
    cfg = EnvConfig()
    space = spaces.Box(low=cfg.action.low, high=cfg.action.high,
                       shape=(cfg.action.action_dim,), dtype=np.float32)
    assert space.shape == (3,)
    assert space.contains(np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    assert space.contains(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert not space.contains(np.array([1.5, 0.0, 0.0], dtype=np.float32))
