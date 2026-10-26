#!/usr/bin/env python3
"""
scripts/test_environment.py

Validates the shared CarlaDrivingEnv against a running CARLA simulator:
connects, resets, steps a few times, and runs gymnasium.check_env so the
Gymnasium API contract is verified on the real server.

Fails fast with a clear message if CARLA is unavailable.

Usage:
    python scripts/test_environment.py [--host localhost] [--port 2000] [--steps 20]
"""

import argparse
import os
import sys
import warnings

# Make the workspace root importable when running this script directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from gymnasium.utils.env_checker import check_env

from common.config import EnvConfig, SimulatorConfig
from common.carla_env import CarlaDrivingEnv, CarlaConnectionError


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the CARLA Gymnasium environment")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=20, help="Manual rollout steps")
    parser.add_argument("--no-check-env", action="store_true", help="Skip gymnasium check_env")
    args = parser.parse_args()

    # Build a config pointing at the requested server with a short episode.
    config = EnvConfig()
    config.simulator = SimulatorConfig(host=args.host, port=args.port)
    config.max_episode_steps = max(args.steps + 5, 50)

    print("Connecting to CARLA ...")
    try:
        env = CarlaDrivingEnv(config=config, connect_on_init=True)
    except CarlaConnectionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: unexpected failure constructing environment: {e}", file=sys.stderr)
        return 3

    try:
        print(f"Connected: {env.is_connected}")

        obs, info = env.reset(seed=0)
        print(f"Reset obs ({obs.dtype}, shape={obs.shape}): {obs}")
        print(f"Reset info keys: {sorted(info.keys())}")
        if not env.observation_space.contains(obs):
            print("ERROR: initial observation is outside the observation space.", file=sys.stderr)
            return 4

        total_reward = 0.0
        for i in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            assert np.isfinite(reward), f"non-finite reward at step {i}: {reward}"
            assert env.observation_space.contains(obs), f"obs out of bounds at step {i}"
            if terminated or truncated:
                print(f"Episode ended at step {i}: terminated={terminated}, truncated={truncated}")
                break
        print(f"Manual rollout done. Steps run: {i + 1}, cumulative reward: {total_reward:.3f}")

        if not args.no_check_env:
            print("Running gymnasium.check_env (this resets and steps the env) ...")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                check_env(env, skip_render_check=True)
            print("gymnasium.check_env: PASSED")

        print("Environment validation: OK")
        return 0
    except Exception as e:
        print(f"ERROR: environment validation failed: {e}", file=sys.stderr)
        return 5
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
