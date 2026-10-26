"""
tests/test_reward.py

Unit tests for the shared reward function. Verifies finiteness, the sign and
magnitude of each component, and the decomposition returned by
CarlaRewardCalculator.compute_reward. Runs fully offline (no CARLA).
"""

import math

import numpy as np
import pytest

from common.config import RewardConfig, RouteConfig
from common.reward import CarlaRewardCalculator


@pytest.fixture
def calc():
    return CarlaRewardCalculator()


def _all_components():
    return [
        "r_progress", "r_speed", "r_lateral", "r_heading", "r_smoothness",
        "r_stationary", "r_collision", "r_out_of_lane", "r_goal", "total_reward",
    ]


# --------------------------------------------------------------------------
# Finiteness guarantees
# --------------------------------------------------------------------------

def test_reward_is_finite_for_normal_inputs(calc):
    reward, comps = calc.compute_reward(
        progress_m=1.0, speed_kmh=30.0, lateral_deviation_m=0.1,
        heading_error_rad=0.05, action=np.array([0.0, 0.5, 0.0], dtype=np.float32),
        previous_action=np.array([0.0, 0.4, 0.0], dtype=np.float32),
        collision=False, goal_reached=False, out_of_lane=False,
    )
    assert math.isfinite(reward)
    for k, v in comps.items():
        assert math.isfinite(v), f"{k} not finite: {v}"


@pytest.mark.parametrize("seed", range(200))
def test_reward_finite_under_random_extreme_inputs(calc, seed):
    """The reward must remain finite for a wide range of inputs."""
    rng = np.random.default_rng(seed)
    reward, comps = calc.compute_reward(
        progress_m=float(rng.uniform(-50, 50)),
        speed_kmh=float(rng.uniform(0, 200)),
        lateral_deviation_m=float(rng.uniform(-20, 20)),
        heading_error_rad=float(rng.uniform(-math.pi, math.pi)),
        action=rng.uniform(-1, 1, size=3).astype(np.float32),
        previous_action=rng.uniform(-1, 1, size=3).astype(np.float32),
        collision=bool(rng.random() < 0.5),
        goal_reached=bool(rng.random() < 0.2),
        out_of_lane=bool(rng.random() < 0.2),
    )
    assert math.isfinite(reward), f"non-finite reward for seed {seed}: {reward}"
    assert math.isfinite(comps["total_reward"])


def test_reward_components_keys_present(calc):
    _, comps = calc.compute_reward(
        0.0, 0.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
        False, False, False,
    )
    for key in _all_components():
        assert key in comps, f"missing component {key}"


def test_total_equals_sum_of_components(calc):
    reward, comps = calc.compute_reward(
        progress_m=2.0, speed_kmh=30.0, lateral_deviation_m=0.5,
        heading_error_rad=0.2, action=np.array([0.1, 0.6, 0.0], dtype=np.float32),
        previous_action=np.array([0.0, 0.5, 0.0], dtype=np.float32),
        collision=False, goal_reached=False, out_of_lane=False,
    )
    recomputed = sum(comps[k] for k in _all_components() if k != "total_reward")
    assert math.isclose(reward, comps["total_reward"], rel_tol=1e-9)
    assert math.isclose(reward, recomputed, rel_tol=1e-6, abs_tol=1e-6)


# --------------------------------------------------------------------------
# Individual component behavior
# --------------------------------------------------------------------------

def test_collision_penalty_applied(calc):
    reward, comps = calc.compute_reward(
        0.0, 0.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
        collision=True, goal_reached=False, out_of_lane=False,
    )
    assert comps["r_collision"] == RewardConfig().collision_penalty
    assert reward < 0  # terminal penalty dominates


def test_goal_bonus_applied(calc):
    reward, comps = calc.compute_reward(
        1.0, 30.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
        collision=False, goal_reached=True, out_of_lane=False,
    )
    assert comps["r_goal"] == RewardConfig().goal_bonus
    assert comps["r_goal"] == 100.0


def test_out_of_lane_penalty_only_without_collision(calc):
    # Out of lane + no collision -> penalty applied.
    _, c1 = calc.compute_reward(0.0, 10.0, 3.0, 0.0, np.zeros(3, dtype=np.float32), None,
                                False, False, True)
    assert c1["r_out_of_lane"] == RewardConfig().out_of_lane_penalty
    # Out of lane + collision -> collision dominates, lane penalty suppressed.
    _, c2 = calc.compute_reward(0.0, 10.0, 3.0, 0.0, np.zeros(3, dtype=np.float32), None,
                                True, False, True)
    assert c2["r_out_of_lane"] == 0.0
    assert c2["r_collision"] == RewardConfig().collision_penalty


def test_stationary_penalty_when_slow(calc):
    _, c = calc.compute_reward(0.0, 0.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
                               False, False, False)
    assert c["r_stationary"] == RewardConfig().stationary_penalty
    _, c2 = calc.compute_reward(0.0, 5.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
                                False, False, False)
    assert c2["r_stationary"] == 0.0


def test_speed_tracking_peaks_at_target(calc):
    target = RouteConfig().target_speed_kmh  # 30 km/h
    w_speed = RewardConfig().w_speed
    # At target -> full speed reward.
    _, c = calc.compute_reward(0.0, target, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
                               False, False, False)
    assert math.isclose(c["r_speed"], w_speed * 1.0, rel_tol=1e-9)
    # At 0 km/h -> zero speed reward.
    _, c0 = calc.compute_reward(0.0, 0.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
                                False, False, False)
    assert math.isclose(c0["r_speed"], 0.0, abs_tol=1e-9)
    # At 2x target -> zero speed reward.
    _, c2x = calc.compute_reward(0.0, 2 * target, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
                                  False, False, False)
    assert math.isclose(c2x["r_speed"], 0.0, abs_tol=1e-6)


def test_progress_reward_positive_for_forward_negative_for_backward(calc):
    _, c_fwd = calc.compute_reward(2.0, 30.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
                                   False, False, False)
    assert c_fwd["r_progress"] > 0.0
    _, c_bwd = calc.compute_reward(-2.0, 30.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
                                   False, False, False)
    assert c_bwd["r_progress"] < 0.0


def test_lateral_penalty_grows_with_deviation(calc):
    _, c_small = calc.compute_reward(0.0, 30.0, 0.1, 0.0, np.zeros(3, dtype=np.float32), None,
                                      False, False, False)
    _, c_big = calc.compute_reward(0.0, 30.0, 2.0, 0.0, np.zeros(3, dtype=np.float32), None,
                                   False, False, False)
    # Penalty weight is negative, so a larger magnitude -> more negative value.
    assert c_big["r_lateral"] < c_small["r_lateral"] <= 0.0


def test_heading_penalty_grows_with_error(calc):
    _, c_small = calc.compute_reward(0.0, 30.0, 0.0, 0.1, np.zeros(3, dtype=np.float32), None,
                                     False, False, False)
    _, c_big = calc.compute_reward(0.0, 30.0, 0.0, 1.5, np.zeros(3, dtype=np.float32), None,
                                   False, False, False)
    assert c_big["r_heading"] < c_small["r_heading"] <= 0.0


def test_smoothness_penalty_zero_without_previous_action(calc):
    _, c = calc.compute_reward(0.0, 30.0, 0.0, 0.0, np.array([0.5, 0.5, 0.5], dtype=np.float32),
                               None, False, False, False)
    assert c["r_smoothness"] == 0.0


def test_smoothness_penalty_nonzero_with_action_change(calc):
    _, c = calc.compute_reward(0.0, 30.0, 0.0, 0.0,
                               np.array([0.5, 0.5, 0.5], dtype=np.float32),
                               np.array([0.0, 0.5, 0.0], dtype=np.float32),
                               False, False, False)
    # w_action_smoothness is negative, so a jerk cost yields a negative contribution.
    assert c["r_smoothness"] < 0.0


def test_reward_uses_custom_config():
    cfg = RewardConfig(w_progress=10.0, collision_penalty=-999.0)
    calc = CarlaRewardCalculator(reward_config=cfg)
    _, c = calc.compute_reward(1.0, 30.0, 0.0, 0.0, np.zeros(3, dtype=np.float32), None,
                               collision=True, goal_reached=False, out_of_lane=False)
    assert c["r_progress"] == 10.0 * 1.0
    assert c["r_collision"] == -999.0
