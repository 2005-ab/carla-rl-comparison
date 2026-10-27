"""Physics, determinism and safety checks for the independent local backend."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from common.local_env import LocalDrivingEnv, baseline_action


def test_gym_contract():
    check_env(LocalDrivingEnv(), skip_render_check=True)


def test_seed_reproduces_dynamics_and_traffic():
    a, b = LocalDrivingEnv(scenario="traffic"), LocalDrivingEnv(scenario="traffic")
    np.testing.assert_array_equal(a.reset(seed=7)[0], b.reset(seed=7)[0])
    for _ in range(20):
        ra, rb = a.step([.01, .6, 0]), b.step([.01, .6, 0])
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1:] == rb[1:]


def test_throttle_accelerates_brake_stops_and_interlocks():
    env = LocalDrivingEnv(scenario="straight")
    env.reset(seed=4)
    for _ in range(10):
        env.step([0, 1, 0])
    assert env.x > 0 and env.speed > 3
    speed = env.speed
    env.step([0, 1, 1])
    assert env.speed < speed and env.previous_action[1] == 0
    for _ in range(10):
        env.step([0, 0, 1])
    assert env.speed == 0


def test_collision_detected_and_no_goal_bonus():
    env = LocalDrivingEnv(scenario="straight")
    env.traffic = [[env.x + 3, 0, 0]]
    _, _, terminated, _, info = env.step([0, 0, 0])
    assert terminated and info["collision"]
    assert not info["goal_reached"]
    with pytest.raises(RuntimeError):
        env.step([0, 0, 0])


def test_lane_departure_and_timeout():
    env = LocalDrivingEnv(scenario="straight")
    env.y = 4
    assert env.step([0, 0, 0])[4]["out_of_lane"]
    env.reset(seed=0)
    env.step_count = env.config.max_episode_steps - 1
    _, _, terminated, truncated, _ = env.step([0, 0, 0])
    assert truncated and not terminated


def test_collision_at_finish_cannot_earn_goal_bonus():
    env = LocalDrivingEnv(scenario="straight")
    env.x = env.config.route.route_length_m
    env.y = float(env.road_y(env.x))
    env.traffic = [[env.x + 3, 0, 0]]
    info = env.step([0, 0, 0])[4]
    assert info["collision"] and not info["goal_reached"]
    assert info["reward_components"]["r_goal"] == 0


@pytest.mark.parametrize("scenario", ["straight", "curve", "traffic", "two_lane"])
def test_reference_controller_completes_route(scenario):
    env = LocalDrivingEnv(scenario=scenario)
    env.reset(seed=1001)
    while not env.done:
        _, reward, _, _, info = env.step(baseline_action(env))
        assert np.isfinite(reward)
    assert info["goal_reached"] and not info["collision"]


@pytest.mark.parametrize("action", [[0, 1], [np.nan, 0, 0], [0, np.inf, 0]])
def test_invalid_actions_rejected(action):
    with pytest.raises(ValueError):
        LocalDrivingEnv().step(action)


@pytest.mark.parametrize("scenario", ["traffic", "two_lane"])
def test_headless_render(monkeypatch, scenario):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = LocalDrivingEnv(render_mode="rgb_array", scenario=scenario)
    frame = env.render()
    assert frame.shape == (800, 1280, 3) and frame.dtype == np.uint8
    assert frame.std() > 10
    env.close()


@pytest.mark.parametrize("scenario", ["traffic", "two_lane"])
def test_3d_cameras_render_without_changing_simulation(monkeypatch, scenario):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from copy import deepcopy
    from common.local_renderer3d import DrivingRenderer3D
    env = LocalDrivingEnv(scenario=scenario)
    env.reset(seed=1001)
    for _ in range(20):
        env.step(baseline_action(env))
    before = env._observation().copy()
    rng = deepcopy(env.np_random.bit_generator.state)
    traffic = deepcopy(env.traffic)
    renderer = DrivingRenderer3D("rgb_array")
    frames = []
    try:
        for _ in range(3):
            frame = renderer.draw(env)
            assert frame.shape == (800,1280,3) and frame.dtype == np.uint8
            assert frame.std() > 10
            frames.append(frame)
            renderer.cycle_camera()
    finally:
        renderer.close()
    assert not np.array_equal(frames[0],frames[1])
    assert not np.array_equal(frames[0],frames[2])
    np.testing.assert_array_equal(env._observation(),before)
    assert env.traffic == traffic
    assert env.np_random.bit_generator.state == rng


def test_near_plane_clipping_prevents_invalid_perspective():
    from common.local_renderer3d import clip_near
    polygon = np.array([[-1,0,-1],[1,0,2],[0,1,2]], dtype=float)
    clipped = clip_near(polygon)
    assert len(clipped) == 4
    assert np.isfinite(clipped).all()
    assert np.all(clipped[:,2] >= .35-1e-12)
    assert len(clip_near(polygon-np.array([0,0,4]))) == 0


def test_two_lane_gym_contract_and_seeded_opposing_traffic():
    env = LocalDrivingEnv(scenario="two_lane")
    check_env(env,skip_render_check=True)
    a,_ = env.reset(seed=11)
    traffic = [car.copy() for car in env.traffic]
    b,_ = env.reset(seed=11)
    np.testing.assert_array_equal(a,b)
    assert env.traffic == traffic
    assert any(car[2]>0 and car[1]==0 for car in traffic)
    assert any(car[2]<0 and car[1]==3.6 for car in traffic)
    env.step([0,0,0])
    for before,after in zip(traffic,env.traffic):
        assert after[0]-before[0] == pytest.approx(before[2]*.1)


def test_adjacent_oncoming_lane_is_not_a_false_obstacle_or_collision():
    env = LocalDrivingEnv(scenario="two_lane")
    env.amplitude = 0
    env.y,env.yaw = 0,0
    env.traffic = [[0,3.6,-10],[28,0,4]]
    assert env.nearest_traffic() == env.traffic[1]
    assert not env.step([0,0,0])[4]["collision"]


def test_wrong_lane_exposes_signed_oncoming_speed_and_collision():
    env = LocalDrivingEnv(scenario="two_lane")
    env.amplitude = 0
    env.y,env.yaw = 3.6,0
    env.traffic = [[5,3.6,-10]]
    assert env._observation()[8] < 0
    assert env.nearest_traffic() == env.traffic[0]
    _,_,terminated,_,info = env.step([0,0,0])
    assert terminated and info["collision"] and info["wrong_lane"]
    assert info["reward_components"]["r_wrong_lane"] == -2


@pytest.mark.parametrize("offset", [-1.9,5.5])
def test_two_lane_road_edges_terminate(offset):
    env = LocalDrivingEnv(scenario="two_lane")
    env.y = float(env.road_y(env.x))+offset
    env.traffic = []
    info = env.step([0,0,0])[4]
    assert info["out_of_lane"]


def test_two_lane_goal_requires_own_lane():
    env = LocalDrivingEnv(scenario="two_lane")
    env.x = env.config.route.route_length_m
    env.y = float(env.road_y(env.x))+3.6
    env.traffic = []
    info = env.step([0,0,0])[4]
    assert info["wrong_lane"] and not info["goal_reached"]
    assert info["reward_components"]["r_goal"] == 0


def test_original_mixed_distribution_excludes_new_challenge():
    env = LocalDrivingEnv(scenario="mixed")
    for seed in range(20):
        assert env.reset(seed=seed)[1]["scenario"] in ("straight","curve","traffic")
